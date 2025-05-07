import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, RandomAffine, RandomHorizontalFlip, RandomCrop, RandomVerticalFlip, Grayscale
from PIL import Image
import cv2
import numpy as np
import random

def ImageTransform(loadSize, grayscale=False):
    base_transforms = {
        "train": [
            RandomCrop(loadSize, pad_if_needed=True, padding_mode='constant', fill=0),
            RandomAffine(10, fill=0),
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.5),
        ],
        "test": []
    }
    
    # Add grayscale conversion if needed
    if grayscale:
        base_transforms["train"].append(Grayscale(num_output_channels=1))
        base_transforms["test"].append(Grayscale(num_output_channels=1))
    
    # Add ToTensor at the end
    base_transforms["train"].append(ToTensor())
    base_transforms["test"].append(ToTensor())
    
    return {
        "train": Compose(base_transforms["train"]),
        "test": Compose(base_transforms["test"]),
        "train_gt": Compose(base_transforms["train"])  # Same transforms for GT
    }


class GoProData(Dataset):
    def __init__(self, base_path, split='train', loadSize=(256, 256), mode=1, grayscale=False, use_gamma=False):
        """
        GoPro dataset loader
        
        Args:
            base_path: Base path to the GoPro dataset (e.g., 'dataset/GoPro')
            split: 'train' or 'test'
            loadSize: Image size for loading/cropping
            mode: 1 for training, 0 for testing
            grayscale: Whether to convert images to grayscale
            use_gamma: Whether to use gamma-corrected blur images
        """
        super().__init__()
        self.base_path = base_path
        self.split = split
        self.mode = mode
        self.grayscale = grayscale
        self.use_gamma = use_gamma
        
        # Get all scene folders
        self.scenes = []
        split_path = os.path.join(base_path, split)
        if os.path.exists(split_path):
            self.scenes = sorted([d for d in os.listdir(split_path) 
                                if os.path.isdir(os.path.join(split_path, d))])
        
        # Build image paths
        self.blur_images = []
        self.sharp_images = []
        self.scene_names = []
        
        for scene in self.scenes:
            scene_dir = os.path.join(split_path, scene)
            blur_dir = os.path.join(scene_dir, "blur_gamma" if use_gamma else "blur")
            sharp_dir = os.path.join(scene_dir, "sharp")
            
            if os.path.exists(blur_dir) and os.path.exists(sharp_dir):
                blur_files = sorted([f for f in os.listdir(blur_dir) if f.endswith('.png')])
                sharp_files = sorted([f for f in os.listdir(sharp_dir) if f.endswith('.png')])
                
                # Make sure we have matching pairs
                common_files = set([f for f in blur_files]) & set([f for f in sharp_files])
                common_files = sorted(list(common_files))
                
                for img_file in common_files:
                    self.blur_images.append(os.path.join(blur_dir, img_file))
                    self.sharp_images.append(os.path.join(sharp_dir, img_file))
                    self.scene_names.append(f"{scene}_{img_file}")
        
        # Set up transforms
        if mode == 1:  # Training mode
            self.ImgTrans = (ImageTransform(loadSize, grayscale)["train"], 
                            ImageTransform(loadSize, grayscale)["train_gt"])
        else:  # Testing mode
            self.ImgTrans = ImageTransform(loadSize, grayscale)["test"]

    def __len__(self):
        return len(self.blur_images)

    def __getitem__(self, idx):
        try:
            # Load images
            blur_path = self.blur_images[idx]
            sharp_path = self.sharp_images[idx]
            
            # Read images using PIL for better compatibility with torchvision transforms
            blur_img = Image.open(blur_path).convert('RGB')
            sharp_img = Image.open(sharp_path).convert('RGB')
            
            if self.mode == 1:  # Training mode with augmentations
                # Use the same seed for both transforms to ensure the same random transformations
                seed = torch.random.seed()
                torch.random.manual_seed(seed)
                blur_img = self.ImgTrans[0](blur_img)
                torch.random.manual_seed(seed)
                sharp_img = self.ImgTrans[1](sharp_img)
            else:  # Testing mode
                blur_img = self.ImgTrans(blur_img)
                sharp_img = self.ImgTrans(sharp_img)
                
            name = self.scene_names[idx]
            return blur_img, sharp_img, name
            
        except Exception as e:
            print(f"Error loading image at index {idx}: {e}")
            # Return a placeholder if there's an error
            channels = 1 if self.grayscale else 3
            placeholder = torch.zeros((channels, 256, 256))
            return placeholder, placeholder, "error_loading"
