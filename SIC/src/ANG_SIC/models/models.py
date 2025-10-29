from typing import Callable, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

def _norm_name(name: str) -> str:
    return (name or "").strip().lower()

def model_expected_in_channels(name: str, dataset: str | None = None) -> int:
    n = _norm_name(name)
    if n == "mojocifarcnn":
        return 3
    if dataset and _norm_name(dataset) == "cifar10":
        return 3
    return 1

def list_available_models() -> List[str]:
    return sorted(
        [
            "ResNet8",
            "ResNet16",
            "SmallMNISTCNN",
            "LeNet5Tanh",
            "LeNet5ELU",
            "SeedCNN",
            "LeNet5Seed",
            "MojoCIFARCNN",
        ]
    )

class BasicBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = (
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
            if stride != 1 or in_channels != out_channels
            else nn.Sequential()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)

class ResNet(nn.Module):
    def __init__(self, block: Callable, num_blocks: tuple[int, int, int], num_classes: int = 10, in_ch: int = 1):
        super().__init__()
        self.in_channels = 16
        self.conv1 = nn.Conv2d(in_ch, 16, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, block: Callable, channels: int, num_blocks: int, stride: int) -> nn.Sequential:
        layers = []
        strides = [stride] + [1] * (num_blocks - 1)
        for s in strides:
            layers.append(block(self.in_channels, channels, s))
            self.in_channels = channels
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return self.fc(out)

def ResNet8(in_ch: int = 1, num_classes: int = 10) -> ResNet:
    return ResNet(BasicBlock, (1, 1, 1), num_classes=num_classes, in_ch=in_ch)

def ResNet16(in_ch: int = 1, num_classes: int = 10) -> ResNet:
    return ResNet(BasicBlock, (2, 2, 2), num_classes=num_classes, in_ch=in_ch)

class MojoCIFARCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 50, 3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(50, 100, 2, stride=1, bias=False)
        self.conv3 = nn.Conv2d(100, 150, 2, stride=1, bias=False)
        self.conv4 = nn.Conv2d(150, 200, 2, stride=1, bias=False)
        self.conv5 = nn.Conv2d(200, 250, 2, stride=1, bias=False)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(250, num_classes, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

class LeNet5Tanh(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.pool1 = nn.AvgPool2d(2)
        self.conv2 = nn.Conv2d(20, 50, 5)
        self.pool2 = nn.AvgPool2d(2)
        self.fc1 = nn.Linear(50 * 4 * 4, 500)
        self.fc2 = nn.Linear(500, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.tanh(self.conv1(x))
        x = self.pool1(x)
        x = torch.tanh(self.conv2(x))
        x = self.pool2(x)
        x = x.view(-1, 50 * 4 * 4)
        x = torch.tanh(self.fc1(x))
        return self.fc2(x)

class LeNet5ELU(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.pool1 = nn.AvgPool2d(2)
        self.conv2 = nn.Conv2d(20, 50, 5)
        self.pool2 = nn.AvgPool2d(2)
        self.fc1 = nn.Linear(50 * 4 * 4, 500)
        self.fc2 = nn.Linear(500, num_classes)
        self.elu = nn.ELU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.elu(self.conv1(x))
        x = self.pool1(x)
        x = self.elu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(-1, 50 * 4 * 4)
        x = self.elu(self.fc1(x))
        return self.fc2(x)

class SmallMNISTCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class SeedCNN(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 7, stride=2, padding=1, bias=False)
        self.conv2 = nn.Conv2d(6, 50, 7, stride=4, padding=2, bias=False)
        self.classifier = nn.Linear(50 * 3 * 3, num_classes, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        return self.classifier(x)

class LeNet5Seed(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5, stride=1, padding=0, bias=True)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(20, 50, kernel_size=5, stride=1, padding=0, bias=True)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.tanh(self.conv1(x))
        x = self.pool1(x)
        x = torch.tanh(self.conv2(x))
        x = self.pool2(x)
        return x.view(x.size(0), -1)

def build_model(name: str, dataset: str | None = None, num_classes: int = 10) -> nn.Module:
    n = _norm_name(name)
    in_ch = model_expected_in_channels(name, dataset=dataset)
    registry: Dict[str, Callable[[], nn.Module]] = {
        "smallmnistcnn": lambda: SmallMNISTCNN(num_classes=num_classes),
        "lenet5tanh": lambda: LeNet5Tanh(num_classes=num_classes),
        "lenet5elu": lambda: LeNet5ELU(num_classes=num_classes),
        "seedcnn": lambda: SeedCNN(num_classes=num_classes),
        "lenet5seed": lambda: LeNet5Seed(),
        "resnet8": lambda: ResNet8(in_ch=in_ch, num_classes=num_classes),
        "resnet16": lambda: ResNet16(in_ch=in_ch, num_classes=num_classes),
        "mojocifarcnn": lambda: MojoCIFARCNN(num_classes=num_classes),
    }
    if n not in registry:
        raise ValueError(f"Unknown model '{name}'. Options: {list_available_models()}")
    return registry[n]()
