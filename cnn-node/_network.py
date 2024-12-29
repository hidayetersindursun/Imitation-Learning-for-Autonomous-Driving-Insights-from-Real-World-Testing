import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CNN_LSTM(nn.Module):
    def __init__(self):
        super(CNN_LSTM, self).__init__()
        self.conv_1 = nn.Conv2d(3, 24, kernel_size=(5, 5), stride=(2, 2))
        self.elu_1 = nn.ELU(alpha=1.0)
        self.conv_2 = nn.Conv2d(24, 36, kernel_size=(5, 5), stride=(2, 2))
        self.elu_2 = nn.ELU(alpha=1.0)
        self.conv_3 = nn.Conv2d(36, 48, kernel_size=(5, 5), stride=(2, 2))
        self.elu_3 = nn.ELU(alpha=1.0)
        self.conv_4 = nn.Conv2d(48, 64, kernel_size=(5, 5), stride=(2, 2))
        self.elu_4 = nn.ELU(alpha=1.0)
        self.conv_5 = nn.Conv2d(64, 64, kernel_size=(3, 3))
        self.elu_5 = nn.ELU(alpha=1.0)
        self.conv_6 = nn.Conv2d(64, 64, kernel_size=(3, 3))
        self.elu_6 = nn.ELU(alpha=1.0)
        self.dropout = nn.Dropout(p=0.5)
        self.flatten_1 = nn.Flatten()
        self.elu_7 = nn.ELU(alpha=1.0)
        self.fc_1 = nn.Linear(15680, 100)
        self.elu_8 = nn.ELU(alpha=1.0)
        self.fc_2 = nn.Linear(100, 50)
        self.elu_9 = nn.ELU(alpha=1.0)
        self.flatten_2 = nn.Flatten()
        self.lstm1 = nn.LSTM(input_size=50, hidden_size=64, num_layers=1, batch_first=True)
        self.fc_3 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.elu_1(self.conv_1(x))
        x = self.elu_2(self.conv_2(x))
        x = self.elu_3(self.conv_3(x))
        x = self.elu_4(self.conv_4(x))
        x = self.elu_5(self.conv_5(x))
        x = self.elu_6(self.conv_6(x))
        x = self.dropout(x)
        x = x.permute(0, 3, 2, 1)  # Permute for LSTM input
        x = self.flatten_1(x)
        x = self.elu_7(x)
        x = self.elu_8(self.fc_1(x))
        x = self.elu_9(self.fc_2(x))
        x = self.flatten_2(x)
        x = x.view(x.size(0), 1, -1)  # Reshape for LSTM input
        lstm_out, _ = self.lstm1(x)
        lstm_out = lstm_out[:, -1, :]  # Get output of the last timestep
        x = self.fc_3(lstm_out)
        return x

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=(5, 5), stride=(2, 2)),
            nn.ELU(alpha=1.0),
            nn.Conv2d(24, 36, kernel_size=(5, 5), stride=(2, 2)),
            nn.ELU(alpha=1.0),
            nn.Conv2d(36, 48, kernel_size=(5, 5), stride=(2, 2)),
            nn.ELU(alpha=1.0),
            nn.Conv2d(48, 64, kernel_size=(5, 5), stride=(2, 2)),
            nn.ELU(alpha=1.0),
            nn.Conv2d(64, 64, kernel_size=(3, 3)),
            nn.ELU(alpha=1.0),
            nn.Conv2d(64, 64, kernel_size=(3, 3)),
            nn.ELU(alpha=1.0),
            nn.Dropout(p=0.5),
            nn.Flatten(),
            nn.ELU(alpha=1.0),
            nn.Linear(15680, 100),
            nn.ELU(alpha=1.0),
            nn.Linear(100, 50), 
            nn.ELU(alpha=1.0),
            nn.Flatten()
        )

    def forward(self, x):
        return self.net(x)

class ODEFunc(nn.Module):
    def __init__(self):
        super(ODEFunc, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(50, 100),
            nn.ELU(alpha=1.0),
            nn.Linear(100, 100),
            nn.ELU(alpha=1.0),
            nn.Linear(100, 50),
            nn.ELU(alpha=1.0)
        )

    def forward(self,t, x):
        return self.model(x)

class CNNODE(nn.Module):
    def __init__(self):
        super(CNNODE, self).__init__()
        self.cnn = CNN()
        self.odefunc = ODEFunc()
        self.integration_time = torch.tensor([0, 0.04]).to(device)#.float().to(device)
        self.fc_last1 = nn.Linear(50, 25)
        self.fc_last2 = nn.Linear(25, 1)
        self.elu = nn.ELU(alpha=1.0)

    def forward(self, x):
        features = self.cnn(x)
        integ = odeint(self.odefunc, features, self.integration_time, method="euler")
        out = integ[1]  # Get the output of the last timestep
        out = self.fc_last1(out) # Fully connected layer
        out = self.elu(out) # Activation function
        out = self.fc_last2(out) # Fully connected layer
        return out
