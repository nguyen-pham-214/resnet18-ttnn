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
    num_classes = 1000   # ImageNet

    # architecture
    block = "BasicBlock"
    layers = [2, 2, 2, 2]   # ResNet18

    # ImageNet stem
    conv1_kernel = 7
    conv1_stride = 2
    conv1_padding = 3

    # max pooling after stem
    maxpool_kernel = 3
    maxpool_stride = 2
    maxpool_padding = 1

    # pooling
    avgpool_output = (1, 1)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels * self.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class ResNet18(nn.Module):
    def __init__(self, config=Config):
        super().__init__()

        self.config = config
        self.in_channels = config.in_channels

        # ImageNet-style stem
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=config.in_channels,
            kernel_size=config.conv1_kernel,
            stride=config.conv1_stride,
            padding=config.conv1_padding,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(config.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(
            kernel_size=config.maxpool_kernel,
            stride=config.maxpool_stride,
            padding=config.maxpool_padding
        )

        # ResNet-18 stages
        self.layer1 = self._make_layer(BasicBlock, 64,  config.layers[0], stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, config.layers[1], stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, config.layers[2], stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, config.layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(config.avgpool_output)
        self.fc = nn.Linear(512 * BasicBlock.expansion, config.num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        acts = {}
        shapes = {}

        acts["input"] = x
        shapes["input"] = tuple(x.shape)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        acts["stem"] = out
        shapes["stem"] = tuple(out.shape)

        
        # acts["maxpool"] = out
        # shapes["maxpool"] = tuple(out.shape)

        out = self.layer1(out)
        acts["layer1"] = out
        shapes["layer1"] = tuple(out.shape)

        out = self.layer2(out)
        acts["layer2"] = out
        shapes["layer2"] = tuple(out.shape)

        out = self.layer3(out)
        acts["layer3"] = out
        shapes["layer3"] = tuple(out.shape)

        out = self.layer4(out)
        acts["layer4"] = out
        shapes["layer4"] = tuple(out.shape)

        # acts["prepool"] = out
        # shapes["prepool"] = tuple(out.shape)

        out = self.avgpool(out)
        # acts["avgpool"] = out
        # shapes["avgpool"] = tuple(out.shape)

        out = torch.flatten(out, 1)
        # acts["flatten"] = out
        # shapes["flatten"] = tuple(out.shape)

        out = self.fc(out)
        acts["head"] = out
        shapes["head"] = tuple(out.shape)

        return out, acts, shapes


def create_torch_model(device):
    torch.manual_seed(0)

    config = Config()
    model = ResNet18(config)

    model.to(device)
    model.eval()

    return model