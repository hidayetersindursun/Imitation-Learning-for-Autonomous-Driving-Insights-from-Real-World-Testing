import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from _network import CNN_LSTM
from _dataset import CustomDataset, compute_mean_std

train_path = 'data/trainData.csv'

# Define data transformations with Z-score normalization using computed mean and std
print("Computing Mean and Std Deviation...")
mean, std = compute_mean_std(CustomDataset(train_path))
print("Computed Mean:", mean)
print("Computed Std:", std)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

# Create custom dataset instance
train_dataset = CustomDataset(train_path, transform=transform)

# Split the training dataset into training and validation datasets
train_size = int(0.85 * len(train_dataset))
valid_size = len(train_dataset) - train_size
train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [train_size, valid_size])

# Create data loader
batch_size=256
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Initialize the model
model = CNN_LSTM().to(device)


# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Lists to store training loss
train_losses = []

# Training loop
num_epochs = 100
min_valid_loss = np.inf

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        # Move inputs and labels to the device
        inputs, labels = inputs.to(device).float(), labels.to(device).float()  # Ensure float type

        optimizer.zero_grad()
        outputs = model(inputs)
        
        # Ensure target tensor has the same size as the output tensor
        labels = labels.view(-1, 1)  # Reshape labels to [batch_size, 1]
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
    valid_loss = 0.0
    model.eval()

    for inputs, labels in valid_loader:
        inputs, labels = inputs.to(device).float(), labels.to(device).float()
        outputs = model(inputs)
        labels = labels.view(-1, 1)
        loss = criterion(outputs, labels)
        valid_loss += loss.item()
        

    print(f'Epoch {epoch+1} \t\t Training Loss: {running_loss / len(train_loader)} \t\t Validation Loss: {valid_loss / len(valid_loader)} \t\t Validation RMSE: {np.sqrt(valid_loss / len(valid_loader))}')
    if min_valid_loss > valid_loss:
        print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
        min_valid_loss = valid_loss
        # Saving State Dict
        #torch.save(model.state_dict(), 'saved_model.pth')


    # Compute average training loss for the epoch
    train_loss = running_loss / len(train_loader)
    train_losses.append(train_loss)

    # Print training loss for the epoch
    print(f'Epoch [{epoch + 1}/{num_epochs}], '
          f'Training Loss: {train_loss:.4f}, ')
          

    # Plot training loss 
    #plt.plot(train_losses, label='Training Loss')
    #plt.xlabel('Epoch')
    #plt.ylabel('Loss')
    #plt.title('Training Loss')
    #plt.legend()
    #plt.pause(15)




