from _network import CNNODE, CNN_LSTM
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

model = CNN_LSTM()

# Visualize the model
writer = SummaryWriter('runs/cnnode_experiment')

# Add model graph to TensorBoard
dummy_input = torch.randn(1, 3, 672, 224)  # Adjust the shape as per your model's input
writer.add_graph(model, dummy_input)
writer.close()

# Example training loop with logging (dummy example)
num_epochs = 5
train_loader = [((torch.randn(1, 3, 672, 224), torch.tensor([0])))]  # Dummy data loader

for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        # Forward pass
        output = model(data)
        loss = nn.CrossEntropyLoss()(output, target)
        
        # Backward pass and optimize (dummy optimizer step)
        # optimizer.zero_grad()
        loss.backward()
        # optimizer.step()
        
        # Log training loss
        writer = SummaryWriter('runs/cnnode_experiment')  # Re-initialize writer to log
        writer.add_scalar('Loss/train', loss.item(), epoch * len(train_loader) + batch_idx)

writer.close()