import torch
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class GraphFloorplanUNet(nn.Module):
    def __init__(self, num_node_features, in_channels, out_channels, features=[64, 128, 256, 512]):
        super(GraphFloorplanUNet, self).__init__()
        self.encoder = Encoder(in_channels, features)
        self.reduce_dim = nn.Conv2d(512, 256, kernel_size=1)
        
        self.decoder = Decoder(out_channels, features)
        
        self.floorplan_gnn = FloorPlanGNN(num_node_features)
        
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x, graph_data):
        input_img = x.clone()

        # Sizes you want
        floorplan_info = []
        resize_sizes = [(256, 256), (128, 128), (64, 64), (32, 32)]
        for size in resize_sizes:
            resized = F.interpolate(input_img, size=size)  # We only need the spatial dimensions here (H, W)
            floorplan_info.append(resized)
        
        skips = []
        for layer in self.encoder.encode:
            x = layer(x)
            skips.append(x)
        
        x = self.reduce_dim(x)
        # # Forward pass through the graph neural network
        graph_features = self.floorplan_gnn(graph_data)
        
        x = torch.cat([x, graph_features], 1)  # concatenate along channel dimension
        
        # Decoder with skip connections
        skips = skips[::-1]
        floorplan_info = floorplan_info[::-1]
        for skip, info, layer in zip(skips, floorplan_info, self.decoder.decode):
            x = layer(torch.cat([x, skip, info], 1))

        return self.final_conv(x) # to 11 classes


# Graph Neural Network
class FloorPlanGNN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_dim=64):
        super(FloorPlanGNN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, 1024)
        self.fc = nn.Linear(1024, 512*512)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)
        x = self.fc(x).view(-1, 256, 32, 32)
        return x

    
class Encoder(nn.Module):
    def __init__(self, in_channels, features):
        super(Encoder, self).__init__()
        layers = []
        for feature in features:
            layers.append(self._block(in_channels, feature))
            in_channels = feature
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)

    def _block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

class Decoder(nn.Module):
    def __init__(self, out_channels, features):
        super(Decoder, self).__init__()
        
        layers = []
        in_ch = features[-1]
        
        # Iterate through reversed features but exclude the last feature 
        # (since it's the starting point)
        for feature in reversed(features[:-1]):
            layers.append(self._block(in_ch*2+3, feature))
            in_ch = feature
        
        # Final convolution to match out_channels ******************************
        layers.append(nn.ConvTranspose2d(128+3, 64, kernel_size=4, stride=2, padding=1))
        
        self.decode = nn.Sequential(*layers)

    def forward(self, x):
        return self.decode(x)

    def _block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1), # Upsampling
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
