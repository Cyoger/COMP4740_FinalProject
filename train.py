import numpy as np
import torch 
from torch_geometric.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from util import image_to_superpixel_graph	
from architecture.GCN import GCN


transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: (x * 255).byte().numpy())])
cifar10_train = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
cifar10_test = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_data_list = [image_to_superpixel_graph(img[0], img[1]) for img in cifar10_train]
test_data_list = [image_to_superpixel_graph(img[0], img[1]) for img in cifar10_test]

train_loader = DataLoader(train_data_list, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data_list, batch_size=32, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN(in_channels=3, hidden_channels=64).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

def train():
    model.train()
    total_loss = 0
    for data in tqdm(train_loader, desc="Training", leave=False):
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def test():
    model.eval()
    correct = 0
    for data in tqdm(test_loader, desc="Testing", leave=False):
        data = data.to(device)
        with torch.no_grad():
            pred = model(data).max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(test_loader.dataset)

epochs = 10
for epoch in range(epochs):
    print(f'Epoch {epoch+1}/{epochs}')
    train_loss = train()
    test_acc = test()
    print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Test Acc: {test_acc:.4f}')