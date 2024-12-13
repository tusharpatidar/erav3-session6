import torch
import torch.nn as nn
import torch.nn.functional as F

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        # First Block
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)  # Output 8 channels
        self.bn1 = nn.BatchNorm2d(8)
        
        # Black 2
        self.conv2 = nn.Conv2d(8, 12, 3, padding=1)  # Output 12 channels
        self.bn2 = nn.BatchNorm2d(12)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.1)
        
        # Block 3
        self.conv3 = nn.Conv2d(12, 16, 3, padding=1)  # Output 16 channels
        self.bn3 = nn.BatchNorm2d(16)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout(0.1)
        
        # Fourth Block with parallel paths
        self.conv4_1x1 = nn.Conv2d(16, 16, 1)  # Keep 16 channels
        self.conv4_main = nn.Conv2d(16, 16, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(16)
        
        # Fifth Block with attention
        self.conv5 = nn.Conv2d(16, 16, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(16)
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(16, 8, 1),
            nn.ReLU(),
            nn.Conv2d(8, 16, 1),
            nn.Sigmoid()
        )
        
        # Sixth Block (new)
        self.conv6 = nn.Conv2d(16, 16, 3, padding=1)
        self.bn6 = nn.BatchNorm2d(16)
        self.dropout3 = nn.Dropout(0.1)
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Final fully connected layer
        self.fc = nn.Linear(16, 10)

    def forward(self, x):
        # First Block
        x = F.relu(self.bn1(self.conv1(x)))
        
        # Second Block
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout1(self.pool1(x))
        
        # Third Block
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout2(self.pool2(x))
        
        # Fourth Block with parallel paths
        x_1x1 = self.conv4_1x1(x)
        x_main = F.relu(self.bn4(self.conv4_main(x)))
        x = x_main + F.relu(x_1x1)  # residual connection
        
        # Fifth Block with attention
        x = F.relu(self.bn5(self.conv5(x)))
        att = self.attention(x)
        x = x * att
        
        # Sixth Block (new)
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.dropout3(x)
        
        # GAP and FC
        x = self.gap(x)
        x = x.view(-1, 16)
        x = self.fc(x)
        return F.log_softmax(x, dim=1) 