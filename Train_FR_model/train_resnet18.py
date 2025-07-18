# import torch
# from torchvision import datasets, transforms, models
# from torch.utils.data import DataLoader
# import torch.nn as nn
# from torchvision.models import resnet18, ResNet18_Weights
# import torch.nn.functional as F
# import torch.optim as optim
# EMB = 128 

# torch.manual_seed(42)
# torch.backends.cudnn.benchmark = True

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("Device : ",torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

# transform_train = transforms.Compose([
#     transforms.Resize((112,112)),
#     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
#     transforms.RandomRotation(degrees=7),
#     transforms.RandomGrayscale(p=0.1),  # helpful for IR-style robustness
#     transforms.ToTensor(),
#     transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])

# ])

# transform_test = transforms.Compose([
#     transforms.Resize(112,112),
#     transforms.ToTensor(),
#     transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
# ])



# train_dataset = datasets.ImageFolder(root='./train_data',transform=transform_train)

# test_dataset = datasets.ImageFolder(root='./test_data',transform=transform_test)

# print(len(train_dataset))
# print(len(train_dataset.classes))

# print(train_dataset.class_to_idx)# print mindex mapping

# train_Loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# test_Loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# class model_def(nn.Module):
#     def __init__(self, ntrain):
#         super().__init__()
#         base = resnet18(weights = ResNet18_Weights.IMAGENET1K_V1)
#         self.features = nn.Sequential(*list(base.children())[:-1])
#         self.classifier = nn.Sequential(
#             nn.Linear(512,EMB)
#         )
    
#     def forward(self,x):
#         x = self.features(x).flatten(1)
#         x = self.classifier(x)
#         return F.normalize(x , p=2 , dim=1)
    


# class ArcFaceLoss(nn.Module):
#     def __init__(self, num_classes, emb_dim=512, s=64.0, m=0.50):
#         super().__init__()
#         self.W = nn.Parameter(torch.randn(emb_dim, num_classes))
#         nn.init.xavier_uniform_(self.W)
#         self.s, self.m = s, m
#         self.num_classes = num_classes

#     def forward(self, emb, labels):
#         W = F.normalize(self.W, p=2, dim=0)         
#         logits = emb @ W                            
#         theta  = torch.acos(torch.clamp(logits, -1+1e-7, 1-1e-7))
#         target_logits = torch.cos(theta + self.m)   

#         one_hot = F.one_hot(labels, self.num_classes).to(emb.dtype)
#         logits  = logits * (1 - one_hot) + target_logits * one_hot
#         logits *= self.s
#         return F.cross_entropy(logits, labels)
    


# model = model_def(EMB=512).to(device)
# loss_fn = ArcFaceLoss(num_classes=len(train_dataset.classes), emb_dim=512).to(device)
# optimizer = optim.Adam(list(model.parameters()) + list(loss_fn.parameters()), lr=1e-3)

# for epoch in range(20):
#     model.train()
#     total_loss = 0

#     for imgs, labels in train_Loader:
#         imgs, labels = imgs.to(device), labels.to(device)
        
#         optimizer.zero_grad()
#         embeddings = model(imgs)
#         loss = loss_fn(embeddings, labels)
#         loss.backward()
#         optimizer.step()

#         total_loss += loss.item()
    
#     avg_loss = total_loss / len(train_Loader)
#     print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")


# model.eval()
# with torch.no_grad():
#     img_tensor = transform_test(single_face_image).unsqueeze(0).to(device)
#     embedding = model(img_tensor)  # shape: [1, 512]


# F.cosine_similarity(embedding1, embedding2)  # Output: similarity score



import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image

# ---- Config ----
EMB_DIM = 128  # embedding size
BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-3

torch.manual_seed(42)
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

# ---- Transforms ----
transform_train = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
    transforms.RandomRotation(degrees=7),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

transform_test = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

# ---- Dataset ----
train_dataset = datasets.ImageFolder(root='./train_data', transform=transform_train)
test_dataset = datasets.ImageFolder(root='./test_data', transform=transform_test)

n_classes = len(train_dataset.classes)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print("Number of classes:", n_classes)
print("Class-to-index mapping:", train_dataset.class_to_idx)

# ---- Model ----
class FaceNet(nn.Module):
    def __init__(self, emb_dim=EMB_DIM):
        super().__init__()
        base = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(base.children())[:-1])  # Remove FC
        self.fc = nn.Linear(512, emb_dim)

    def forward(self, x):
        x = self.features(x).flatten(1)
        x = self.fc(x)
        return F.normalize(x, p=2, dim=1)  # L2-normalized embeddings

# ---- ArcFace Loss ----
class ArcFaceLoss(nn.Module):
    def __init__(self, num_classes, emb_dim=EMB_DIM, s=64.0, m=0.50):
        super().__init__()
        self.W = nn.Parameter(torch.randn(emb_dim, num_classes))
        nn.init.xavier_uniform_(self.W)
        self.s, self.m = s, m
        self.num_classes = num_classes

    def forward(self, emb, labels):
        W = F.normalize(self.W, p=2, dim=0)
        logits = emb @ W
        theta = torch.acos(torch.clamp(logits, -1 + 1e-7, 1 - 1e-7))
        target_logits = torch.cos(theta + self.m)
        one_hot = F.one_hot(labels, self.num_classes).to(emb.dtype)
        logits = logits * (1 - one_hot) + target_logits * one_hot
        logits *= self.s
        return F.cross_entropy(logits, labels)

# ---- Initialize ----
model = FaceNet().to(device)
loss_fn = ArcFaceLoss(num_classes=n_classes, emb_dim=EMB_DIM).to(device)
optimizer = optim.Adam(list(model.parameters()) + list(loss_fn.parameters()), lr=LR)

# ---- Training Loop ----
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        embeddings = model(imgs)
        loss = loss_fn(embeddings, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch + 1}/{EPOCHS}], Loss: {avg_loss:.4f}")

# ---- Inference Example ----
def get_embedding(model, image_path):
    model.eval()
    image = Image.open(image_path).convert("RGB")
    tensor = transform_test(image).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model(tensor)
    return emb

# Example (replace with your own image paths)
emb1 = get_embedding(model, "test_data/person1/img1.jpg")
emb2 = get_embedding(model, "test_data/person1/img2.jpg")

# Cosine similarity between embeddings
similarity = F.cosine_similarity(emb1, emb2).item()
print("Cosine similarity:", similarity)

