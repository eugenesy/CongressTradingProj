import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TGNMemory, TransformerConv
from torch_geometric.nn.models.tgn import (
    IdentityMessage,
    MeanAggregator,
)

class GraphAttentionEmbedding(torch.nn.Module):
    def __init__(self, in_channels, out_channels, msg_dim, time_enc):
        super().__init__()
        self.time_enc = time_enc
        edge_dim = msg_dim + time_enc.out_channels + 2
        
        self.conv1 = TransformerConv(in_channels, out_channels // 4, heads=4,
                                    dropout=0.1, edge_dim=edge_dim)
        self.conv2 = TransformerConv(out_channels, out_channels // 4, heads=4,
                                    dropout=0.1, edge_dim=edge_dim)
        self.dropout = torch.nn.Dropout(0.1)
        self.relu = torch.nn.ReLU()

    def forward(self, x, edge_index, edge_attr):
        x = self.conv1(x, edge_index, edge_attr)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index, edge_attr)
        return x

class LinkPredictor(torch.nn.Module):
    """
    Decodes node embeddings into logits for disjoint intervals.
    Output dim = num_classes (which is num_thresholds + 1).
    """
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes) # Outputs logits for each Interval
        )

    def forward(self, z_src, z_dst, s_src, s_dst, p_src, p_dst):
        h_src = torch.cat([z_src, s_src, p_src], dim=-1)
        h_dst = torch.cat([z_dst, s_dst, p_dst], dim=-1)
        h_pair = torch.cat([h_src, h_dst], dim=-1)
        return self.net(h_pair)

class PriceEncoder(torch.nn.Module):
    def __init__(self, input_dim=14, hidden_dim=32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
    def forward(self, x):
        return self.mlp(x)

class TGN(torch.nn.Module):
    def __init__(self, num_nodes, raw_msg_dim, memory_dim, time_dim, embedding_dim, 
                 num_parties, num_states, num_classes):
        super().__init__()
        self.num_nodes = num_nodes
        self.memory_dim = memory_dim
        self.time_dim = time_dim
        self.num_classes = num_classes
        
        # Price Encoder
        self.price_emb_dim = 32
        self.price_encoder = PriceEncoder(input_dim=14, hidden_dim=self.price_emb_dim)
        
        self.augmented_msg_dim = raw_msg_dim + self.price_emb_dim
        
        # Memory Module
        self.memory = TGNMemory(
            num_nodes=num_nodes,
            raw_msg_dim=self.augmented_msg_dim,
            memory_dim=memory_dim,
            time_dim=time_dim,
            message_module=IdentityMessage(self.augmented_msg_dim, memory_dim, time_dim),
            aggregator_module=MeanAggregator(),
        )
        
        # Embedding Module
        self.gnn = GraphAttentionEmbedding(
            in_channels=memory_dim,
            out_channels=embedding_dim,
            msg_dim=self.augmented_msg_dim,
            time_enc=self.memory.time_enc,
        )
        
        # Static Embeddings
        self.emb_party = nn.Embedding(num_parties + 1, 8) 
        self.emb_state = nn.Embedding(num_states + 1, 8)
        static_dim = 16
        
        # Link Predictor (Interval Classification)
        self.predictor = LinkPredictor(
            in_channels=embedding_dim + static_dim + self.price_emb_dim, 
            num_classes=num_classes
        )

    def encode_static(self, x_static_idx):
        p = self.emb_party(x_static_idx[:, 0])
        s = self.emb_state(x_static_idx[:, 1])
        return torch.cat([p, s], dim=-1)

    def forward(self, n_id, edge_index, edge_attr, price_seq=None):
        memory = self.memory.memory[n_id]
        z = self.gnn(memory, edge_index, edge_attr)
        return z
        
    def get_price_embedding(self, price_seq):
        return self.price_encoder(price_seq)

    def predict_cumulative_probs(self, logits):
        """
        Converts raw logits (for disjoint intervals) into cumulative probabilities.
        This guarantees monotonicity: P(>0%) >= P(>4%).
        
        Args:
            logits: [Batch, Num_Classes]
        Returns:
            probs: [Batch, Num_Thresholds] (Binary label probabilities)
        """
        # 1. Softmax to get probability of being in each specific interval
        # interval_probs: [Batch, N+1]
        interval_probs = F.softmax(logits, dim=-1)
        
        # 2. Compute Reverse Cumulative Sum
        # If intervals are: [<0, 0-4, 4-8, >8]
        # P(>0) = P(0-4) + P(4-8) + P(>8)
        # We assume the last class is the "highest" bin.
        # We flip, cumsum, and flip back.
        
        # We exclude the 0-th index from the final output because
        # the 0-th cumulative sum would be P(Total) which is always 1.0.
        # We want P(>Threshold_1), P(>Threshold_2)...
        
        # cumsum from right to left
        cum_probs = torch.cumsum(interval_probs.flip(dims=[-1]), dim=-1).flip(dims=[-1])
        
        # The output at index 0 is P(>= Threshold_0).
        # The output at index 1 is P(>= Threshold_1).
        # The last element is P(>= Threshold_Last).
        
        # We drop the very first element if it corresponds to "Everything above -infinity" (which is 1.0)
        # But wait, our classes correspond to thresholds.
        # Class 0: < T0
        # Class 1: T0 <= x < T1
        # ...
        # P(x >= T0) = sum(Class 1 ... Class N)
        # P(x >= T1) = sum(Class 2 ... Class N)
        
        # Our cum_probs currently:
        # Index 0: sum(Class 0...N) = 1.0
        # Index 1: sum(Class 1...N) = P(x >= T0) -> This corresponds to TARGET_COLUMNS[0]
        # Index N: sum(Class N)     = P(x >= TN) -> This corresponds to TARGET_COLUMNS[last]
        
        # So we just slice off the first column (which represents the "Below Lowest Threshold" prob included)
        return cum_probs[:, 1:]