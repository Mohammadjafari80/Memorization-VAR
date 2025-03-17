import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
    
class CustomImageNet(Dataset):
    def __init__(self, root, transform=None):
        # Expand user directory if needed
        self.root = os.path.expanduser(root)
        
        # Store transform
        self.transform = transform
        
        # Find all image files
        self.image_paths = self._find_images()
    
    def _find_images(self):
        # Find all .png files, sorted by index
        image_paths = [
            os.path.join(self.root, fname) 
            for fname in sorted(os.listdir(self.root)) 
            if fname.endswith('.png')
        ]
        
        return image_paths
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        # Apply transform if provided
        if self.transform:
            image = self.transform(image)
        
        return image, img_path