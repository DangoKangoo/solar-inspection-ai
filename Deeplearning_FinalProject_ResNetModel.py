import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from PIL import Image
from pathlib import Path

import numpy as np
from glob import glob
import matplotlib.pyplot as plt

IMAGE_SIZE = [224, 224]

trainFolder = Path("./train")
validateFolder = Path("./val")

resNet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1) 
resNet.fc = nn.Identity() 

print(resNet)

for param in resNet.parameters(): 
    param.requires_grad = False

#for param in resNet.layer4.parameters():
    #param.requires_grad = True

# Classes
classes = glob(str(trainFolder / "*"))
print(classes)

classesNum = len(classes)
print(classesNum)

# Continue with the next layers of the model:

class Model(nn.Module):
    def __init__(self, backbone, numClasses):
        super().__init__()
        self.backbone = backbone
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(2048, numClasses)

    def forward(self, x):
        x = self.backbone(x) # outputs [N, 2048]
        x = self.flatten(x)
        x = self.classifier(x) # logits

        return x

# Create the model and add new layers
model = Model(resNet, classesNum)
#print(model)

# Pytorch replacement for compile (loss/optimizer/metrics)
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam([
    #{"params": model.backbone.layer4.parameters(), "lr": 1e-5},
    {"params": model.classifier.parameters(), "lr": 1e-3}
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


# Image Augmentation

trainTransform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]) 
])

testTransform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

trainingDataset = datasets.ImageFolder(trainFolder, transform=trainTransform)
testDataset = datasets.ImageFolder(validateFolder, transform=testTransform)

trainingSet = DataLoader(trainingDataset, batch_size=32, shuffle=True, num_workers=0)
testSet = DataLoader(testDataset, batch_size=32, shuffle=False, num_workers=0)


# Fit the model
epoches = 47 # Change as needed
history = {"accuracy": [], "val_accuracy": [], "loss": [], "val_loss": []}

for epoch in range(epoches):
    model.train()
    running_loss = 0.0
    running_correct = 0
    running_total = 0

    for images, labels in trainingSet:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = torch.argmax(logits, dim=1)
        running_correct += (preds == labels).sum().item()
        running_total += labels.size(0)
        
    train_loss = running_loss / max(1, running_total)
    train_acc = running_correct / max(1, running_total)

    model.eval()
    val_loss_sum = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in testSet:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            val_loss_sum += loss.item() * images.size(0)
            preds = torch.argmax(logits, dim=1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)
        
    val_loss = val_loss_sum / max(1, val_total)
    val_acc = val_correct / max(1, val_total)

    history["loss"].append(train_loss)
    history["accuracy"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_accuracy"].append(val_acc)

    print(
        f"Epoch: {epoch+1:02d} | "
        f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
        f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
    )


class Result:
    def __init__(self, history_dict):
        self.history = history_dict

result = Result(history)

# Plot the result

# Plot the accuracy
plt.plot(result.history["accuracy"], label="trainAcc")
plt.plot(result.history["val_accuracy"], label="valAcc")
plt.legend()
plt.show()

# Plot the loss
plt.plot(result.history["loss"], label="trainLoss")
plt.plot(result.history["val_loss"], label="valLoss")
plt.legend()
plt.show()

# Save the model
torch.save(model.state_dict(), "./TestModel.pth")