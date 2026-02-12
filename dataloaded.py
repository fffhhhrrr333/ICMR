import os
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
import albumentations as A
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from config import *

class CustomDataset(Dataset):
    def __init__(self, optical_dir, radar_dir, label_dir, transform=None):
        self.optical_dir = optical_dir
        self.radar_dir = radar_dir
        self.label_dir = label_dir
        self.transform = transform
        self.image_names = os.listdir(optical_dir)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]

        # Load optical image (4 channels)
        optical_path = os.path.join(self.optical_dir, image_name)
        optical_image = Image.open(optical_path)
        
        # Load radar image (1 channel)
        radar_path = os.path.join(self.radar_dir, image_name)
        radar_image = Image.open(radar_path)
        
        # Load label image (binary mask)
        label_path = os.path.join(self.label_dir, image_name)
        label_image = np.array(Image.open(label_path))

        # Stack optical and radar images
        optical_image = np.array(optical_image)
        radar_image = np.expand_dims(np.array(radar_image), axis=-1)  # Reshape to (H, W, 1)
        combined_image = np.concatenate((optical_image, radar_image), axis=-1)  # Shape (H, W, 5)

        # Apply transformations
        if self.transform:
            augmented = self.transform(image=combined_image, mask=label_image)
            combined_image = augmented['image']
            label_image = augmented['mask']

        return combined_image, label_image.unsqueeze(0).long()

optical_dir = OPTICAL_DIR
radar_dir = RADAR_DIR
label_dir = LABEL_DIR

transform = A.Compose([
    A.ToFloat(255),
    A.Resize(height, width),
    A.HorizontalFlip(),
    A.VerticalFlip(),
    ToTensorV2(),
])

# Create dataset
dataset = CustomDataset(optical_dir, radar_dir, label_dir, transform=transform)

# Split dataset into training and testing sets
train_size = int(0.8 * len(dataset))  # 80% for training
test_size = len(dataset) - train_size  # 20% for testing
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

