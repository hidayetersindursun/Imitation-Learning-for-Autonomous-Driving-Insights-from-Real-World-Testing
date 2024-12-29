import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from torchdiffeq import odeint_adjoint as odeint
from _network import ODEFunc, CNN, CNNODE, CNN_LSTM
from _dataset import CustomDataset, compute_mean_std, normalize_images

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Path to the training data
model_save_path = 'torch_models/'

train_path = 'data/trainData.csv'
#train_path = 'data/lstm_train_test_data/trainData.csv'
model_output_name = model_save_path + 'all_data_ode_euler.pth'
print(model_output_name)
"""
print("computing mean and std")
# Compute mean and std deviation of the training dataset
mean, std = compute_mean_std(CustomDataset(train_path))
#mean = torch.tensor([0.4817, 0.4832, 0.4788])
#std = torch.tensor([0.1519, 0.1553, 0.1513])
print("Computed Mean:", mean)
"""

# Define data transformations with Z-score normalization
transform = transforms.Compose([
    transforms.ToTensor()  # normalize to [0, 1] images
])
# Create custom dataset instance
train_dataset = CustomDataset(train_path, transform=transform)
# Split the training dataset into training and validation datasets
train_size = int(0.85 * len(train_dataset))
valid_size = len(train_dataset) - train_size
train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [train_size, valid_size])

# Create data loader
batch_size = 256
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size)

"""
for images, labels in train_loader:
    if not torch.all(images >= 0) or not torch.all(images <= 1):
        print("False")
print("True")
"""

model = CNNODE().to(device)
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'The model has {count_parameters(model):,} trainable parameters')

criterion = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Lists to store training loss
train_losses = []
valid_losses = []

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

        outputs =  model(inputs)
        labels = labels.view(-1, 1)  # Reshape labels to [batch_size, 1]
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    valid_loss = 0.0
    model.eval()

    for inputs, labels in valid_loader:
        inputs, labels = inputs.to(device).float(), labels.to(device).float()
        #features = feature_extractor(inputs)
        next_states = model(inputs)
        #outputs = next_states[:, 1]
        labels = labels.view(-1, 1)
        loss = criterion(next_states, labels)
        valid_loss += loss.item()
        

    print(f'Epoch {epoch+1} \t\t Training Loss: {running_loss / len(train_loader)} \t\t Validation Loss: {valid_loss / len(valid_loader)} \t\t Validation RMSE: {np.sqrt(valid_loss / len(valid_loader))}')
    if min_valid_loss > valid_loss:
        print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
        min_valid_loss = valid_loss
        # Saving State Dict
        torch.save(model.state_dict(), model_output_name)
        #torch.save(fully_connected.state_dict(), 'saved_fully_connected.pth')


    # Compute average training loss for the epoch
    train_loss = running_loss / len(train_loader)
    train_losses.append(train_loss)
    valid_losses.append(valid_loss / len(valid_loader))

    # Print training loss for the epoch
    print(f'Epoch [{epoch + 1}/{num_epochs}], '
        f'Training Loss: {train_loss:.4f}, ')
    
# Plot training and validation loss curves
plt.plot(train_losses, label='Training Loss')
plt.plot(valid_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# save plot as figure
plt.savefig('training_validation_loss.png')
