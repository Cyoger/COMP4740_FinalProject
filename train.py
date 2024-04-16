from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch_geometric.nn import GATConv, global_mean_pool
from torchvision import datasets, transforms
from torchvision.transforms import RandomHorizontalFlip, RandomCrop, RandomRotation, ColorJitter, RandomAffine, RandomPerspective, RandomResizedCrop, GaussianBlur
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader 
from torch.optim import Adam
from tqdm import tqdm

from architecture.gat2 import HybridGATModel

train_transform = transforms.Compose([
    RandomHorizontalFlip(),
    RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

batch_size = 32

trainset = CIFAR10(root='./data', train=True, download=True, transform=train_transform)
train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

testset = CIFAR10(root='./data', train=False, download=True, transform=test_transform)
test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = HybridGATModel().to(device)
optimizer = Adam(model.parameters(), lr=0.001)
criterion = nn.NLLLoss()

losses = []
accuracies = []

def train(epoch):
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1} [TRAIN]', total=len(train_loader))
    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())
    avg_loss = total_loss / len(train_loader)
    losses.append(avg_loss)
    return avg_loss

def test(epoch):
    model.eval()
    correct = 0
    total = 0
    progress_bar = tqdm(test_loader, desc=f'Epoch {epoch+1} [TEST]', total=len(test_loader))
    with torch.no_grad():
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            pred = output.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
            progress_bar.set_postfix(accuracy=100. * correct / total)
    accuracy = 100. * correct / total
    accuracies.append(accuracy)
    return accuracy


best_accuracy = 0
for epoch in range(50):
    train_loss = train(epoch)
    test_acc = test(epoch)
    if test_acc > best_accuracy:
        best_accuracy = test_acc
        torch.save(model.state_dict(), 'best_model_GAT_cnn_power_adam.pth')
    print(f'Epoch: {epoch+1}, Training Loss: {train_loss:.4f}, Test Accuracy: {test_acc:.2f}%')


plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(losses, label='Training Loss')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(accuracies, label='Test Accuracy')
plt.title('Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.show()