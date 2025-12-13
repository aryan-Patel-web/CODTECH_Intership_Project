"""
Task 2: Simple image classification (PyTorch)
- Expects: data/images/train/ and data/images/val/ structured by class folders
- Uses pretrained resnet18 for transfer learning
- Saves best model to models/saved_model.pth
"""

import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'images')
model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
os.makedirs(model_dir, exist_ok=True)

transform = {
    'train': transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ]),
    'val': transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])
}

train_ds = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transform['train'])
val_ds = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=transform['val'])
train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(train_ds.classes))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

def train(epochs=3):
    best_acc = 0.0
    history = {'train_loss':[], 'val_loss':[], 'val_acc':[]}
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
        train_loss = train_loss / len(train_loader.dataset)

        # validation
        model.eval()
        val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = correct / len(val_loader.dataset)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        print(f"Epoch {epoch+1}/{epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(model_dir, 'saved_model.pth'))
    # simple plot
    plt.figure(figsize=(8,4))
    plt.plot(history['train_loss'], label='train_loss')
    plt.plot(history['val_loss'], label='val_loss')
    plt.legend(); plt.title('Loss')
    plt.savefig(os.path.join(model_dir, 'loss_plot.png'))
    plt.figure()
    plt.plot(history['val_acc'], label='val_acc')
    plt.legend(); plt.title('Val Accuracy')
    plt.savefig(os.path.join(model_dir, 'acc_plot.png'))
    print("Training complete. Model and plots saved to models/")





if __name__ == "__main__":
    if not os.path.exists(os.path.join(data_dir, 'train')):
        print("Create sample images under data/images/train/<class> and data/images/val/<class>")
    else:
        train(epochs=3)
