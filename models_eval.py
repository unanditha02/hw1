import torch
import torch.nn as nn
import torch.nn.functional as F


class CaffeNetFC7(nn.Module):
    def __init__(self, num_classes=20, inp_size=224, c_dim=3):
        super().__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(in_channels=c_dim, out_channels=96, kernel_size=11, stride=4, padding='valid')
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding='same')
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding='same')
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding='same')
        self.conv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding='same')
        self.nonlinear = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.flat_dim = 256 * 5 * 5
        self.fc6 = nn.Sequential(nn.Linear(self.flat_dim, 4096), nn.ReLU(), nn.Dropout(0.5))
        self.fc7 = nn.Sequential(nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5))
        self.fc8 = nn.Sequential(nn.Linear(4096, num_classes))
    
    def forward(self, x):
        N = x.size(0)
        x = self.conv1(x)
        x = self.nonlinear(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.nonlinear(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.nonlinear(x)
        x = self.conv4(x)
        x = self.nonlinear(x)

        x = self.conv5(x)
        x = self.nonlinear(x)
        x = self.pool5(x)

        flat_x = x.view(N, self.flat_dim)
        out = self.fc6(flat_x)
        out = self.fc7(out)
        # out = self.fc8(out)

        return out


class CaffeNet53(nn.Module):
    def __init__(self, num_classes=20, inp_size=224, c_dim=3):
        super().__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(in_channels=c_dim, out_channels=96, kernel_size=11, stride=4, padding='valid')
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding='same')
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding='same')
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding='same')
        self.conv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding='same')
        self.nonlinear = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.flat_dim = 256 * 5 * 5
        self.fc6 = nn.Sequential(nn.Linear(self.flat_dim, 4096), nn.ReLU(), nn.Dropout(0.5))
        self.fc7 = nn.Sequential(nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5))
        self.fc8 = nn.Sequential(nn.Linear(4096, num_classes))
    
    def forward(self, x):
        N = x.size(0)
        x = self.conv1(x)
        x = self.nonlinear(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.nonlinear(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.nonlinear(x)
        x = self.conv4(x)
        x = self.nonlinear(x)

        x = self.conv5(x)
        x = self.nonlinear(x)
        x = self.pool5(x)

        flat_x = x.view(N, self.flat_dim)
        out = self.fc6(flat_x)
        out = self.fc7(out)
        out = self.fc8(out)

        return out