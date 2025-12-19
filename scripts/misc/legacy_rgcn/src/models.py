import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, to_hetero

class GNNEncoder(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        # SAGEConv is good for inductive learning
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

class EdgeDecoder(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        # Simple MLP to predict edge label from src_emb + dst_emb + edge_attr
        # Input dim = NodeEmb + NodeEmb + EdgeAttr
        # Edge Attr is NOW 3 dims (Amount, FilingGap, TimeDecay) - 'Type' is structural
        self.lin1 = torch.nn.Linear(2 * hidden_channels + 3, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, 1) # Binary classification

    def forward(self, z_dict, edge_label_index, edge_attr):
        # z_dict: Dictionary of node embeddings
        # edge_label_index: The edges we want to predict
        
        row, col = edge_label_index
        
        # Get embeddings for source (Politician) and target (Company)
        z_src = z_dict['politician'][row]
        z_dst = z_dict['company'][col]
        
        # Concatenate: [Src, Dst, EdgeFeatures]
        z = torch.cat([z_src, z_dst, edge_attr], dim=-1)
        
        z = self.lin1(z).relu()
        z = self.lin2(z)
        return z.view(-1)

class Model(torch.nn.Module):
    def __init__(self, hidden_channels, metadata):
        super().__init__()
        # Encoder: Heterogeneous GNN
        self.encoder = GNNEncoder(hidden_channels, hidden_channels)
        self.encoder = to_hetero(self.encoder, metadata, aggr='sum')
        
        # Decoder: Link Prediction
        self.decoder = EdgeDecoder(hidden_channels)

    def encode(self, x_dict, edge_index_dict):
        """
        Computes Node Embeddings for the entire graph.
        """
        return self.encoder(x_dict, edge_index_dict)

    def decode(self, z_dict, edge_label_index, edge_attr):
        """
        Predicts logits for specific edges using pre-computed embeddings.
        """
        return self.decoder(z_dict, edge_label_index, edge_attr)

    def forward(self, x_dict, edge_index_dict):
        # Just Encode. Decoding must be called explicitly for specific edge types.
        return self.encode(x_dict, edge_index_dict)
