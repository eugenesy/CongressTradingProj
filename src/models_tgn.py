import torch
import torch.nn as nn
from torch_geometric.nn import TGNMemory, TransformerConv
from torch_geometric.nn.models.tgn import (
    LastNeighborLoader,
    IdentityMessage,
    MeanAggregator,  # Changed from LastAggregator for better message aggregation
)

class GraphAttentionEmbedding(torch.nn.Module):
    def __init__(self, in_channels, out_channels, msg_dim, time_enc):
        super().__init__()
        self.time_enc = time_enc
        edge_dim = msg_dim + time_enc.out_channels + 2
        
        # Layer 1
        self.conv1 = TransformerConv(in_channels, out_channels // 4, heads=4,
                                    dropout=0.1, edge_dim=edge_dim)
        # Layer 2 (Deep TGN)
        self.conv2 = TransformerConv(out_channels, out_channels // 4, heads=4,
                                    dropout=0.1, edge_dim=edge_dim)
        self.dropout = torch.nn.Dropout(0.1)
        self.relu = torch.nn.ReLU()

    def forward(self, x, edge_index, edge_attr):
        # 1-Hop
        x = self.conv1(x, edge_index, edge_attr)
        x = self.relu(x)
        x = self.dropout(x)
        # 2-Hop
        x = self.conv2(x, edge_index, edge_attr)
        return x

class LinkPredictor(torch.nn.Module):
    """
    Deep Interaction Decoder.
    Concatenates Src and Dst embeddings to learn non-linear interactions
    (e.g., specific politician types trading specific stock types).
    """
    def __init__(self, in_channels):
        super().__init__()
        # Input size is 2 * in_channels (Src + Dst)
        self.net = nn.Sequential(
            nn.Linear(in_channels * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, z_src, z_dst, s_src, s_dst, p_src, p_dst):
        # 1. Construct Node Embeddings
        # h: [Batch, In_Channels]
        h_src = torch.cat([z_src, s_src, p_src], dim=-1)
        h_dst = torch.cat([z_dst, s_dst, p_dst], dim=-1)
        
        # 2. Concatenate Pair (Interaction)
        # h_pair: [Batch, In_Channels * 2]
        h_pair = torch.cat([h_src, h_dst], dim=-1)
        
        # 3. Decode
        return self.net(h_pair)


class PriceEncoder(torch.nn.Module):
    """
    MLP Encoder for engineered price features.
    Input: 14 dims (7 Stock + 7 SPY features)
    Output: 32 dims (Price Embedding)
    """
    def __init__(self, input_dim=14, hidden_dim=32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.BatchNorm1d(input_dim),  # Normalize features (Crucial!)
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),            # Regularization
            nn.Linear(hidden_dim, hidden_dim),
        )
        
    def forward(self, x):
        # x: [Batch, 14] (NOT a sequence anymore!)
        return self.mlp(x)

class TGN(torch.nn.Module):
    def __init__(self, num_nodes, raw_msg_dim, memory_dim, time_dim, embedding_dim, num_parties, num_states):
        super().__init__()
        self.num_nodes = num_nodes
        self.memory_dim = memory_dim
        self.time_dim = time_dim
        
        # 0. Price Encoder (MLP - for engineered features)
        # Input: 14 dims (7 Stock + 7 SPY)
        # Output: 32 dims (Embedding)
        self.price_emb_dim = 32
        self.price_encoder = PriceEncoder(input_dim=14, hidden_dim=self.price_emb_dim)
        
        # Message dimension increases by Price Embedding
        # Old msg: [Amt, Buy, Gap] = 3
        # New msg: [Amt, Buy, Gap] + [PriceEmb] = 3 + 32 = 35
        self.augmented_msg_dim = raw_msg_dim + self.price_emb_dim
        
        # 1. Memory Module
        # Updates state S_i(t) based on interactions
        self.memory = TGNMemory(
            num_nodes=num_nodes,
            raw_msg_dim=self.augmented_msg_dim, # INCREASED
            memory_dim=memory_dim,
            time_dim=time_dim,
            message_module=IdentityMessage(self.augmented_msg_dim, memory_dim, time_dim), # INCREASED
            aggregator_module=MeanAggregator(),
        )
        
        # 2. Embedding Module
        # Computes Z_i(t) from S_i(t) and spatial neighbors
        self.gnn = GraphAttentionEmbedding(
            in_channels=memory_dim,
            out_channels=embedding_dim,
            msg_dim=self.augmented_msg_dim, # INCREASED
            time_enc=self.memory.time_enc,
        )
        
        # 3. Static Embeddings (Politician Meta)
        self.emb_party = nn.Embedding(num_parties + 1, 8) 
        self.emb_state = nn.Embedding(num_states + 1, 8)
        
        static_dim = 8 + 8
        
        # 4. Link Predictor
        # Takes (Dynamic_Emb[100] + Static_Emb[16] + Price_Emb[32])
        # We also feed the current price context to the decoder
        self.predictor = LinkPredictor(embedding_dim + static_dim + self.price_emb_dim)

    def encode_static(self, x_static_idx):
        # x_static_idx: [Batch, 2] (Party, State)
        p = self.emb_party(x_static_idx[:, 0])
        s = self.emb_state(x_static_idx[:, 1])
        return torch.cat([p, s], dim=-1)

    def forward(self, n_id, edge_index, edge_attr, price_seq=None):
        """
        Compute Embeddings for nodes 'n_id'
        
        Args:
            n_id: Node IDs
            edge_index: Graph Connectivity
            edge_attr: Edge Features (MUST BE AUGMENTED with PriceEmb before calling this if used in GNN)
            price_seq: [Num_Events, 60, 20] - Sequence data for the *events* in this batch.
                       Note: TGN.forward is usually called *after* memory update.
                       The `edge_attr` passed here usually comes from `TemporalData`.
                       
        Wait, TGN flow in PyG:
        1. Batch of events -> Update Memory (using msg).
        2. Get Memory -> Run GNN (using memory + edges).
        
        In `train_rolling.py`, we construct the message *manually* to update memory?
        Let's check `train_rolling.py`.
        Usually:
        `memory.update_state(src, dst, t, msg)`
        Then:
        `z = model(n_id, edge_index, edge_attr)`
        
        The `edge_attr` in `model()` is the features on the edges used for GNN aggregation.
        If we want the neighbor's price context to influence my embedding, `edge_attr` must include PriceEmb.
        
        This means the caller (`train_rolling.py`) is responsible for:
        1. Encoding `price_seq` -> `price_emb`.
        2. Concatenating `raw_msg` + `price_emb` -> `augmented_msg`.
        3. Passing `augmented_msg` to `memory.update_state`.
        4. Passing `augmented_msg` as `edge_attr` to `model`.
        
        So `TGN.forward` doesn't strictly need to run the LSTM.
        BUT, if we want end-to-end gradient, the LSTM must be part of the model.
        So `train_rolling.py` will call `model.encode_price(seq)` -> `emb`.
        Then pass that emb around.
        
        I will allow `forward` to take `edge_attr` which is ALREADY augmented.
        I will add a helper method `encode_price`.
        And update `LinkPredictor` usage.
        """
        
        # Retrieve current memory
        memory = self.memory.memory[n_id]
        
        # Compute Embeddings using GAT
        # edge_attr should already be augmented (Msg + PriceEmb)
        z = self.gnn(memory, edge_index, edge_attr)
        
        return z
        
    def get_price_embedding(self, price_seq):
        return self.price_encoder(price_seq)

