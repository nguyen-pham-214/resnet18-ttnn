# !pip install torch torchvision matplotlib -q
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms


class Config:
    # model structure
    in_channels = 64
    num_classes = 10
    
    # architecture
    block = "BasicBlock"
    layers = [2, 2, 2, 2]   # ResNet18
    
    # convolution settings
    conv1_kernel = 3
    conv1_stride = 1
    conv1_padding = 1
    
    # pooling
    avgpool_output = (1,1)

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # The shortcut is an identity mapping that adds the original input to the block output.
        # For element-wise addition, both tensors must have the same shape.
        # If the input and output shapes differ (due to stride or channel change),
        # a 1x1 convolution is used to project the input to the required shape.
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(x)
        out = self.relu(out)

        return out


class ResNet18(nn.Module):

    def __init__(self, config=Config):
        super().__init__()

        self.config = config
        self.in_channels = config.in_channels

        self.conv1 = nn.Conv2d(
            3,
            config.in_channels,
            kernel_size=config.conv1_kernel,
            stride=config.conv1_stride,
            padding=config.conv1_padding,
            bias=False
        )

        self.bn1 = nn.BatchNorm2d(config.in_channels)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(BasicBlock, 64, config.layers[0], stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, config.layers[1], stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, config.layers[2], stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, config.layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(config.avgpool_output)
        self.fc = nn.Linear(512, config.num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):

        strides = [stride] + [1]*(num_blocks-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels

        return nn.Sequential(*layers)

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)

        out = out.view(out.size(0), -1)

        out = self.fc(out)

        return out


def create_torch_model(device):
    torch.manual_seed(0)

    config = Config()
    model = ResNet18(config)

    model.to(device)
    model.eval()

    return model