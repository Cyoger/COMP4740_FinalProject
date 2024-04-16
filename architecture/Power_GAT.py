import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool


class CNNFeatureExtractor(nn.Module):
    def __init__(self):
        super(CNNFeatureExtractor, self).__init__()
        # Add more convolutional layers
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # Additional conv layer
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),  # Additional conv layer
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256, -1)  # Change to match new conv layer output
        x = x.permute(0, 2, 1)
        return x

class HybridGATModel(nn.Module):
    def __init__(self):
        super(HybridGATModel, self).__init__()
        self.cnn = CNNFeatureExtractor()
        self.gat1 = GATConv(in_channels=256, out_channels=256, heads=1, concat=True)
        self.gat2 = GATConv(in_channels=256, out_channels=128, heads=1, concat=True)
        self.out = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.6)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, images):
        x = self.cnn(images)  # Process images through the CNN
        batch_size, num_nodes, in_channels = x.shape
        edge_index = self.create_complete_graph(num_nodes).to(x.device)
        
        # Use reshape to correctly handle contiguous memory layout requirements
        x = x.reshape(-1, in_channels)  # Flatten to fit GAT input requirements
        
        x = self.dropout(self.relu(self.gat1(x, edge_index)))
        x = self.dropout(self.relu(self.gat2(x, edge_index)))
        
        # Correctly compute the batch index for global mean pooling
        batch_index = torch.arange(batch_size * num_nodes, device=x.device) // num_nodes
        
        x = global_mean_pool(x, batch_index)  # Pool features across the graph
        x = self.out(x)
        return F.log_softmax(x, dim=1)

    @staticmethod
    def create_complete_graph(num_nodes):
        edge_index = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    edge_index.append([i, j])
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        return edge_index
