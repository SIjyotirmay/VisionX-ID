import torch
from torchvision import datasets, transforms














torch.manual_seed(42)
torch.backends.cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device : ",torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

transform_train = transforms.Compose([

])

transform_test = transforms.Compose({

})

train_dataset = datasets.ImageFolder(root='',transform=transform_train)

train_dataset = datasets.ImageFolder(root='',transform=transform_test)