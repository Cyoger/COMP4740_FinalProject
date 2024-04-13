import numpy as np
from skimage.segmentation import slic
import torch
from torch_geometric.data import Data

def clean_edge_index(edge_index, num_nodes):
    mask = (edge_index[0] < num_nodes) & (edge_index[1] < num_nodes) & (edge_index[0] >= 0) & (edge_index[1] >= 0)
    return edge_index[:, mask]


def image_to_superpixel_graph(image, label, num_segments=75):
    if not isinstance(image, torch.Tensor):
        image = torch.tensor(image) 


    image_np = (image.permute(1, 2, 0) * 255).byte().numpy()  

    segments = slic(image_np, n_segments=num_segments, compactness=20, sigma=1)
    segment_ids = np.unique(segments)

    features = np.zeros((len(segment_ids), 3)) 
    for i, seg_id in enumerate(segment_ids):
        mask = segments == seg_id
        mean_color = image_np[mask].mean(axis=0)
        features[i, :] = mean_color 

    edges = []
    for i in range(segments.shape[0]-1):
        for j in range(segments.shape[1]-1):
            current = segments[i, j]
            right = segments[i, j+1]
            bottom = segments[i+1, j]
            if current != right:
                edges.append([current, right])
                edges.append([right, current]) 
            if current != bottom:
                edges.append([current, bottom])
                edges.append([bottom, current]) 

    edge_index = clean_edge_index(torch.tensor(edges).t().contiguous(), len(features))  

    return Data(x=torch.tensor(features, dtype=torch.float), edge_index=edge_index, y=torch.tensor([label]))
