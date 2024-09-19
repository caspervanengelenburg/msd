import os
import torch
import torch.nn as nn
import torch_geometric as pyg
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from torchvision import transforms
from utils_unet import load_pickle
from models_unet import GraphFloorplanUNet
from PIL import Image

def analysis_array(array):
    flattened_arr = array.flatten()
    unique_elements = np.unique(flattened_arr)

    return unique_elements, len(unique_elements)

def visualize_result(boundary_image_np, gt_image_np, predicted_np):
    # Plot the ground truth and prediction
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    axs[0].imshow(boundary_image_np, cmap='gray')
    axs[0].set_title('Boundary Image')

    axs[1].imshow(gt_image_np, cmap='viridis')
    axs[1].set_title('Ground Truth')

    im = axs[2].imshow(predicted_np, cmap='viridis')
    axs[2].set_title('Prediction')

    # add a color bar
    fig.colorbar(im, ax=axs.ravel().tolist(), orientation='vertical')
    plt.show()


def infer(model, boundary_image, graph_pyg):
    model.eval()

    with torch.no_grad():
        # Perform inference
        output = model(boundary_image.unsqueeze(0).cuda(), graph_pyg.to('cuda'))
        predicted = torch.argmax(output, dim=1)

        # Convert tensors to numpy arrays for visualization
        predicted_np = predicted.cpu().numpy()[0]

    return predicted_np


def load_sample(graph_path, struct_path):
    transform = transforms.ToTensor()

    # Load struct image
    struct_in = np.load(struct_path)
    boundary_image = struct_in.astype(np.uint8) 
    boundary_image = transform(boundary_image)

    # Load graph
    num_room_types = 4 
    num_connection_types = 3
    connection_dic = {'door': 0, 'entrance': 1, 'passage': 2}
    
    graph_nx = load_pickle(graph_path)
    graph_nx.graph['struct'] = struct_in[np.newaxis, ...]
    node_features = []
    for _, node_data in graph_nx.nodes(data=True):
        node_type = node_data['zoning_type']
        node_feature = [0]*num_room_types
        node_feature[node_type] = 1
        node_features.append(node_feature)
    
    # Add node features if not present
    edge_features = []
    for _, _, edge_data in graph_nx.edges(data=True):
        connection_type = connection_dic[edge_data['connectivity']]
        edge_feature = [0]*num_connection_types
        edge_feature[connection_type] = 1
        edge_features.append(edge_feature)

    # Convert to PyG graph
    graph_pyg = pyg.utils.from_networkx(graph_nx)
    graph_pyg.x = torch.tensor(node_features, dtype=torch.float)  # node features
    graph_pyg.edge_attr = torch.tensor(edge_features, dtype=torch.float)  # edge features
    
    return boundary_image, graph_pyg

# Create the model
num_node_features = 4
input_nc = 3
output_nc = 11

model = GraphFloorplanUNet(num_node_features, input_nc, output_nc, features=[64, 128, 256, 512]).cuda()

# Load model weights if needed
model.load_state_dict(torch.load('checkpoints/model_checkpoint_epoch_99.pt'))

graph_path = './dataset_processed/test/graph_in/'
filenames = [os.path.splitext(f)[0] for f in os.listdir(graph_path)]

dest_dir = 'submission_unet'

# Load sample
for filename in filenames:
    graph_path = './dataset_processed/test/graph_in/'+filename+'.pickle'
    struct_path = './dataset_processed/test/struct_in/'+filename+'.npy'
    boundary_image, graph_pyg = load_sample(graph_path, struct_path)
    boundary_image = boundary_image.cuda()
    graph_pyg = graph_pyg.cuda()

    predicted = infer(model, boundary_image, graph_pyg)
    
    # Make a copy of the array
    array_copy = predicted.copy()
    
    # Change pixels in the copy
    array_copy[array_copy == 10] = 13
    
    array_copy = array_copy.astype('uint8')
    image = Image.fromarray(array_copy)
    image.save(os.path.join(dest_dir, filename + '.png'))
    