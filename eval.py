import itertools
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

from matplotlib import pyplot as plt
import torch
from torch_geometric.loader import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from architecture.gat2 import HybridGATModel
from torch.optim.lr_scheduler import ReduceLROnPlateau

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HybridGATModel().to(device)
model.load_state_dict(torch.load('best_model_GAT_cnn_power_adam.pth'))
model.eval()


test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
test_loader = DataLoader(testset, batch_size=32, shuffle=False)


all_preds = []
all_labels = []
misclassified_images = []

for images, labels in tqdm(test_loader, desc="Evaluating model"):
    images, labels = images.to(device), labels.to(device)
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)
    all_preds.extend(predicted.view(-1).cpu().numpy())
    all_labels.extend(labels.view(-1).cpu().numpy())

    misclass_mask = predicted != labels
    misclassified_images.extend([(images[i], predicted[i], labels[i]) for i in range(images.size(0)) if misclass_mask[i]])


fig = plt.figure(figsize=(10, 5))
for i, (img, pred, true) in enumerate(misclassified_images[:10]):
    img = img.cpu().numpy().transpose((1, 2, 0))
    img = img * 0.5 + 0.5 
    ax = fig.add_subplot(2, 5, i + 1, xticks=[], yticks=[])
    ax.imshow(img)
    ax.set_title(f"{pred.item()} ({true.item()})", color=("green" if pred == true else "red"))

plt.show()

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=testset.classes, yticklabels=testset.classes)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()