import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()

        # 输入: [B, 1, 28, 28]
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(128 * 7 * 7, 256)
        self.fc2 = nn.Linear(256, num_classes)

        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        # x: [B, 1, 28, 28]

        x = F.relu(self.bn1(self.conv1(x)))
        # [B, 16, 28, 28]

        x = F.relu(self.bn2(self.conv2(x)))
        # [B, 32, 28, 28]

        x = self.pool(x)
        # [B, 32, 14, 14]

        x = F.relu(self.bn3(self.conv3(x)))
        # [B, 64, 14, 14]

        x = F.relu(self.bn4(self.conv4(x)))
        # [B, 128, 14, 14]

        x = self.pool(x)
        # [B, 128, 7, 7]

        x = x.view(x.size(0), -1)
        # [B, 128 * 7 * 7]

        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        x = self.fc2(x)
        # [B, 10]

        return x