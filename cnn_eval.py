import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F

poolSize = 2
poolStride = 2

conv1KSize = 3
conv1Pad = 0
conv1Stride = 1    

conv1FMSize = (32 - conv1KSize + (2*conv1Pad)) / conv1Stride + 1
conv1AfterPool = (conv1FMSize - poolSize) / poolStride + 1
if not (conv1FMSize).is_integer() or not (conv1AfterPool).is_integer():
    print("WARNING: VALUES SELECTED FOR CONV1 LAYER ARE INVALID")
conv1FMSize = int(conv1FMSize)
conv1AfterPool = int(conv1AfterPool)

conv2KSize = 4
conv2Pad = 0
conv2Stride = 1

conv2FMSize = (conv1AfterPool - conv2KSize + (2*conv2Pad)) / conv2Stride + 1 
conv2AfterPool = (conv2FMSize - poolSize) / poolStride + 1
if not (conv2FMSize).is_integer() or not (conv2AfterPool).is_integer():
    print("WARNING: VALUES SELECTED FOR CONV2 LAYER ARE INVALID")
conv2FMSize = int(conv2FMSize)
conv2AfterPool = int(conv2AfterPool)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, conv1KSize)
        self.pool = nn.MaxPool2d(poolSize, poolStride)
        self.conv2 = nn.Conv2d(16, 64, conv2KSize)
        self.fc1 = nn.Linear(64 * conv2AfterPool * conv2AfterPool, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Load the trained model
model = Net()
model.load_state_dict(torch.load('models/69_cnn_cq.pth'))
model.eval()

# Define data transforms for the CIFAR-10 dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize input images
])

# Load the CIFAR-10 test dataset
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define the loss function
criterion = nn.CrossEntropyLoss()

# Evaluate the model
correct = 0
total = 0
with torch.no_grad():  # Disable gradient tracking during evaluation
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

# Print the accuracy
print('Accuracy on the test set: {:.2f}%'.format(100 * correct / total))

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# prepare to count predictions for each class
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

# again no gradients needed
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predicted):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1

print(f"|{'|'.join(classes)}|") 
a = "".join(['|---' for i in range(len(classes))])
print(f"{a}|")

# print accuracy for each class
scores = []
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    scores.append(str(accuracy) + "%")

print(f"|{'|'.join(scores)}|")