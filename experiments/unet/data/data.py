import os
import torch
import torch_geometric as pyg
import numpy as np
import networkx as nx
from utils import load_pickle
from torchvision import transforms

transform = transforms.ToTensor()
target_transform = lambda x: torch.from_numpy(np.array(x, dtype=np.int64))

class msdDataset(torch.utils.data.Dataset):
    def __init__(self, path, graph_type = 'zoning'):
        self.graph_path = os.path.join(path, 'graph_in' if 'zoning' in graph_type else 'graph_out')
        self.struct_path = os.path.join(path, 'struct_in')
        self.full_path = os.path.join(path, 'full_out')
        self.transform = transform
        self.target_transform = target_transform
        
        self.filenames = [os.path.splitext(f)[0] for f in os.listdir(self.graph_path)] # add this line

    def __getitem__(self, index):
        filename = self.filenames[index] # use this filename instead of the index
        
        graph_nx = load_pickle(os.path.join(self.graph_path, f'{filename}.pickle'))
        struct_in = np.load(os.path.join(self.struct_path, f'{filename}.npy'))
        full_out = np.load(os.path.join(self.full_path, f'{filename}.npy'))
        graph_nx.graph['struct'] = struct_in[np.newaxis, ...]
        graph_nx.graph['full'] = full_out[np.newaxis, ...]

        # Load boundary image
        boundary_image = struct_in.astype(np.uint8) 
        boundary_image = self.transform(boundary_image)
        
        # Load ground truth image
        gt_image = full_out[..., 0].astype(np.uint8)
        gt_image = self.target_transform(gt_image)
        
        num_room_types = 4  
        num_connection_types = 3 
        connection_dic = {'door': 0, 'entrance': 1, 'passage': 2}
        
        node_features = []
        for _, node_data in graph_nx.nodes(data=True):
            node_type = node_data['zoning_type']
            node_feature = [0]*num_room_types
            node_feature[node_type] = 1
            node_features.append(node_feature)

        # For edge attributes
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

        return boundary_image, graph_pyg, gt_image

    def __len__(self):
        return len(self.filenames)
