import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        nn.init.kaiming_uniform_(self.conv1.weight)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        nn.init.kaiming_uniform_(self.conv2.weight)
        self.bn2 = nn.BatchNorm2d(32)

        self.pool1 = nn.MaxPool2d(2, 2)
        self.drop1 = nn.Dropout2d(0.2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        nn.init.kaiming_uniform_(self.conv3.weight)
        self.bn3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        nn.init.kaiming_uniform_(self.conv4.weight)
        self.bn4 = nn.BatchNorm2d(64)

        self.pool2 = nn.MaxPool2d(2, 2)
        self.drop2 = nn.Dropout2d(0.3)

        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        nn.init.kaiming_uniform_(self.conv5.weight)
        self.bn5 = nn.BatchNorm2d(128)

        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        nn.init.kaiming_uniform_(self.conv6.weight)
        self.bn6 = nn.BatchNorm2d(128)

        self.pool3 = nn.MaxPool2d(2, 2)
        self.drop3 = nn.Dropout2d(0.4)

        self.conv7 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        nn.init.kaiming_uniform_(self.conv7.weight)
        self.bn7 = nn.BatchNorm2d(128)

        self.conv8 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        nn.init.kaiming_uniform_(self.conv8.weight)
        self.bn8 = nn.BatchNorm2d(128)

        self.drop4 = nn.Dropout2d(0.4)

        self.fc1 = nn.Linear(128 * 4 * 4, 128)
        nn.init.kaiming_uniform_(self.fc1.weight)
        self.bn9 = nn.BatchNorm1d(128)
        self.drop5 = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(128, 64)
        nn.init.kaiming_uniform_(self.fc2.weight)
        self.bn10 = nn.BatchNorm1d(64)
        self.drop6 = nn.Dropout(0.5)

        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.drop1(self.pool1(x))

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.drop2(self.pool2(x))

        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.drop3(self.pool3(x))

        x = F.relu(self.bn7(self.conv7(x)))
        x = F.relu(self.bn8(self.conv8(x)))
        x = self.drop4(x)

        x = x.view(-1, 128 * 4 * 4)

        x = F.relu(self.bn9(self.fc1(x)))
        x = self.drop5(x)

        x = F.relu(self.bn10(self.fc2(x)))
        x = self.drop6(x)

        x = self.fc3(x)
        return x

# Load the trained model
model = Net()
model.load_state_dict(torch.load('models/82_cifar_cnn_cq.pth'))
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