import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=100, kernel_size=3, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(in_channels=100, out_channels=50, kernel_size=3, stride=1, padding=0, bias=False)
        self.fc1 = nn.Linear(in_features=5*5*50, out_features=500)
        self.fc2 = nn.Linear(in_features=500, out_features=10)
    def forward(self, x):
        #x [1, 28, 28]
        x = F.relu(self.conv1(x)) #[100, 26, 26]
        x = F.max_pool2d(x, 2, 2) #[100, 13, 13]
        x = F.relu(self.conv2(x)) #[50, 11, 11]
        x = F.max_pool2d(x, 2, 2) #[50, 5, 5]
        x = x.view(-1, 5*5*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

