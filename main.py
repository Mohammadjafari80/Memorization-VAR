###############################
# 1. Download checkpoints and build models
###############################
import os
import os.path as osp
import torch, torchvision
import random
import numpy as np
import PIL.Image as PImage, PIL.ImageDraw as PImageDraw
from models import VQVAE, build_vae_var
from dataset import CustomImageNet
import torch.distributed as dist
import csv

# Set random seed for reproducibility
seed = 42 
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    
# Only initialize if not already done.
if not dist.is_initialized():
    dist.init_process_group(
        backend='nccl' if torch.cuda.is_available() else 'gloo',
        init_method='tcp://127.0.0.1:29500',
        rank=0, world_size=1
    )
    
# Disable default parameter initialization for faster speed.
setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)

MODEL_DEPTH = 16    # =====> please specify MODEL_DEPTH <=====
assert MODEL_DEPTH in {16, 20, 24, 30}

# Download checkpoint files.
hf_home = 'https://huggingface.co/FoundationVision/var/resolve/main'
vae_ckpt, var_ckpt = 'vae_ch160v4096z32.pth', f'var_d{MODEL_DEPTH}.pth'
if not osp.exists(vae_ckpt):
    os.system(f'wget {hf_home}/{vae_ckpt}')
if not osp.exists(var_ckpt):
    os.system(f'wget {hf_home}/{var_ckpt}')

# Build vae and var.
patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if 'vae' not in globals() or 'var' not in globals():
    vae, var = build_vae_var(
        V=4096, Cvae=32, ch=160, share_quant_resi=4,    # hard-coded VQVAE hyperparameters
        device=device, patch_nums=patch_nums,
        num_classes=1000, depth=MODEL_DEPTH, shared_aln=False,
    )

# Load checkpoints and set models to evaluation mode.
vae.load_state_dict(torch.load(vae_ckpt, map_location='cpu'), strict=True)
var.load_state_dict(torch.load(var_ckpt, map_location='cpu'), strict=True)
vae.eval(), var.eval()
for p in vae.parameters():
    p.requires_grad_(False)
for p in var.parameters():
    p.requires_grad_(False)
print('Prepare finished.')

for name, param in vae.encoder.named_parameters():
    print(name)

###############################
# 2. Register hooks to capture unit average activations
###############################
hook_names = [
    'conv_in',
    'down.0.block.0',
    'down.0.block.1',
    'down.0.downsample',
    'down.1.block.0',
    'down.1.block.1',
    'down.1.downsample',
    'down.2.block.0',
    'down.2.block.1',
    'down.2.downsample',
    'down.3.block.0',
    'down.3.block.1',
    'down.3.downsample',
    'down.4.block.0',
    'down.4.block.1',
    'down.4.attn.0',
    'down.4.attn.1',
    'mid.block_1',
    'mid.attn_1',
    'mid.block_2',
    'norm_out',
    'conv_out',
] # Extracted manually for the model!

# Dictionary to store unit averages for each module per forward pass.
unit_averages = {name: [] for name in hook_names}

def get_hook(name):
    def hook(module, input, output):
        if output.dim() == 4:
            avg = output.mean(dim=(2, 3)).detach().cpu()
        elif output.dim() == 2:
            avg = output.mean(dim=1).detach().cpu()
        else:
            avg = output.view(output.size(0), -1).mean(dim=1).detach().cpu()
        unit_averages[name].append(avg)
    return hook

def get_module(module, dotted_name):
    parts = dotted_name.split('.')
    for part in parts:
        if part.isdigit():
            module = module[int(part)]
        else:
            module = getattr(module, part)
    return module

hooks = {}
encoder = vae.encoder
for name in hook_names:
    submodule = get_module(encoder, name)
    hooks[name] = submodule.register_forward_hook(get_hook(name))
    print(f"Hook registered on module: {name}")

###############################
# 3. Process a dataset and write unit averages to CSV in batches
###############################
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms

# --- Augmentation configuration ---
use_augmentation = False  # Set to True to use augmentation averaging.
n_augmentations =  # increasing this will also probably enhance robustness of the analysis

s = 1
transform_augment = transforms.Compose([
    transforms.RandomApply([transforms.RandomHorizontalFlip()], p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.9 * s, 0.9 * s, 0.9 * s, 0.1 * s)], p=0.9),
    transforms.RandomGrayscale(p=0.1),
    transforms.Resize((256,256)),
    # transforms.ToTensor(),  # outputs tensor in [0,1]
    transforms.Lambda(lambda x: x * 2 - 1)  # normalize to [-1,1]
])
# For non-augmented processing.
transform_fixed = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x * 2 - 1)
])

# --- Dataset Setup ---
dataset = CustomImageNet(root='~/imagenet-100', transform=transform_fixed)
# Use a subset of the dataset.
random.seed(seed)
num_samples = 10000 # Use a bigger number if you have access to resources :) 
indices = random.sample(range(len(dataset)), num_samples)
print(indices[:10]) # to make sure indices are the same for different experiments!
dataset = Subset(dataset=dataset, indices=indices)

dataloader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=4)
print(f"Dataset length is: {len(dataset)}")

# Open CSV file once in write mode.
output_csv = "imagenet_no_aug_unit_averages.csv"
with open(output_csv, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["image_index", "layer_name", "unit_index", "value"])

    with torch.inference_mode():
        for images, image_path in dataloader:
            # If using augmentation, images is a list of PIL images.
            if use_augmentation:
                batch_size = len(images)
                aug_list = []  # will hold augmented tensors for all images.
                # For each image in the batch, apply the same transform n_augmentations times.
                for i in range(batch_size):
                    img = images[i]  # a PIL image.
                    single_aug = []
                    for _ in range(n_augmentations):
                        aug_tensor = transform_augment(img)  # tensor in [-1,1]
                        single_aug.append(aug_tensor.unsqueeze(0))
                    # Stack: shape [n_augmentations, C, H, W]
                    single_aug = torch.cat(single_aug, dim=0)
                    aug_list.append(single_aug)
                    
                # Concatenate: shape [B*n_augmentations, C, H, W]
                big_batch = torch.cat(aug_list, dim=0).to(device)
                
                # Forward pass on augmented batch.
                _ = vae(big_batch)
                
                # For each hooked layer, average activations over augmentations.
                for layer_name in hook_names:
                    if not unit_averages[layer_name]:
                        continue
                    bat = unit_averages[layer_name].pop(0)  # shape: [B*n_augmentations, num_units]
                    bat = bat.view(batch_size, n_augmentations, -1)
                    bat_avg = bat.mean(dim=1)  # shape: [B, num_units]
                    
                    # Write out results: iterate over images in the batch and each unit.
                    for i in range(batch_size):
                        for unit_idx in range(bat_avg.shape[1]):
                            writer.writerow([image_path[i], layer_name, unit_idx, bat_avg[i, unit_idx].item()])
            else:
                # Without augmentation: images is already a tensor.
                images = images.to(device)
                batch_size = images.shape[0]
                _ = vae(images)
                for layer_name in hook_names:
                    if not unit_averages[layer_name]:
                        continue
                    bat = unit_averages[layer_name].pop(0)
                    # bat is expected to have shape [B, num_units]
                    for i in range(batch_size):
                        for unit_idx in range(bat.shape[1]):
                            writer.writerow([image_path[i], layer_name, unit_idx, bat[i, unit_idx].item()])

print(f"Unit averages saved to {output_csv}")

# Remove hooks once processing is complete.
for hook in hooks.values():
    hook.remove()
