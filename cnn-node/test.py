import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from _network import CNN_LSTM, CNNODE
from _dataset import CustomDataset, compute_mean_std

"""
1- traindata path ini düzenle
2- model path ini düzenle
3- model tipini düzenle
4- testdata path ini düzenle
"""


# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load train data for computing mean and std deviation
#train_path = 'data/hidayet_data/testData.csv'
train_path = 'data/lstm_train_test_data/trainData.csv'
train_dataset = CustomDataset(train_path)

"""
# Compute mean and std deviation
print("Computing Mean and Std Deviation...")
mean, std = compute_mean_std(train_dataset)
print("Computed Mean:", mean)
print("Computed Std:", std)
"""

label_mean = -0.0214 # cnnlstm4: -0.0214 | all_Data: -0.0207
label_std = 0.1967 # cnnlstm4: 0.1967 | all_Data: 0.2036

# denormalization
def denormalize(x):
    return x * label_std + label_mean
 
 
# Define data transformations with Z-score normalization using computed mean and std
transform = transforms.Compose([
    transforms.ToTensor()  # normalize to [0, 1] images
])

def evaluate_model(model, test_loader, device):
    # Define loss function
    criterion = nn.MSELoss()
    
    # Evaluation mode
    model.eval()
    test_losses = []
    predictions = []
    true_labels = []

    # Iterate over test data
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device).float(), labels.to(device).float()
            
            # Forward pass
            outputs = model(inputs)
            labels = labels.view(-1, 1)
            
            # Compute loss
            loss = criterion(outputs, labels)
            test_losses.append(loss.item())

            # Denormalize predictions and labels
            denorm_outputs = denormalize(outputs.cpu().numpy())
            denorm_labels = denormalize(labels.cpu().numpy())
            
            # Store predictions and true labels
            predictions.extend(denorm_outputs)
            true_labels.extend(denorm_labels)

    # Convert predictions and true labels to numpy arrays
    predictions = np.array(predictions)
    true_labels = np.array(true_labels)

    # Calculate test loss
    mse = np.mean((predictions - true_labels) ** 2)
    rmse = np.sqrt(mse)
    print(f'Test MSE: {mse:.6f}')
    print(f'Test RMSE: {rmse:.6f}')


    return true_labels, predictions


if __name__ == "__main__":
    # Load test data
    test_path = 'data/lstm_train_test_data/testData.csv'
    test_dataset = CustomDataset(test_path, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=256)

    # Initialize the model
    model = CNNODE().to(device)

    # Load trained model weights
    model.load_state_dict(torch.load('torch_models/cnnlstm4_ode_rk4.pth'))

    # Evaluate the model
    true_labels, predictions = evaluate_model(model, test_loader, device)

    # Plot predicted vs true values
    #plt.figure(figsize=(10, 6))
    plt.plot(predictions, label='Prediction',linewidth=0.75)
    plt.plot(true_labels, label='Ground Truth',linewidth=0.75)
    plt.xlabel('Sample')
    plt.ylabel('Steering angle (rad)')
    #plt.tick_params(axis='both', which='major', labelsize=16)
    #plt.title('True vs Predicted Values')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()
