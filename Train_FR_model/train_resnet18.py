import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn.functional as F

EMB = 128 

torch.manual_seed(42)
torch.backends.cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device : ",torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

transform_train = transforms.Compose([
    transforms.Resize((112,112)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
    transforms.RandomRotation(degrees=7),
    transforms.RandomGrayscale(p=0.1),  # helpful for IR-style robustness
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])

])

transform_test = transforms.Compose([
    transforms.Resize(112,112),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])



train_dataset = datasets.ImageFolder(root='./train_data',transform=transform_train)

test_dataset = datasets.ImageFolder(root='./test_data',transform=transform_test)

print(len(train_dataset))
print(len(train_dataset.classes))

print(train_dataset.class_to_idx)# print mindex mapping

train_DataLoader = DataLoader(train_dataset, batch_size=32, shuffle=True)

test_DataLoader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class model_def(nn.Module):
    def __init__(self, ntrain):
        super().__init__()
        base = resnet18(weights = ResNet18_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(base.children())[:-1])
        self.classifier = nn.Sequential(
            nn.Linear(512,EMB)
        )
    
    def forward(self,x):
        x = self.features(x).flatten(1)
        x = self.classifier(x)
        return F.normalize(x , p=2 , dim=1)
    


    
