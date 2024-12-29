from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import torch

# Define a custom dataset class
class CustomDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx, 0]
        label = self.data.iloc[idx, 1]

        # Load image
        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)

        return image, label
    
# Compute mean and standard deviation from your dataset for images
def compute_mean_std(dataset):
    mean = torch.zeros(3)
    std = torch.zeros(3)
    count = 0

    for image, _ in dataset:
        image = transforms.ToTensor()(image)  # Convert PIL image to tensor
        mean += torch.mean(image, dim=(1, 2))  # Compute mean across height and width dimensions
        std += torch.std(image, dim=(1, 2))    # Compute standard deviation across height and width dimensions
        count += 1

    mean /= count
    std /= count

    return mean, std

def normalize_images(dataset):
    
    for image, _ in dataset:
        image = image / 255.0

    return dataset

