import torch.nn as nn
import torchvision.models as models

def get_resnet18_cifar():
    model = models.resnet18(weights=None)

    # Adatta la prima conv per immagini 32x32
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()  # rimuove il maxpool iniziale
    model.fc = nn.Linear(512, 10)  # CIFAR-10 ha 10 classi

    return model