"""
models_basic.py
===============
Enterprise Module.

Refactored/Audited: 2026-03-20
"""

import torch
import torch.nn as nn
from torch_geometric.nn import TGNMemory, TransformerConv
from torch_geometric.nn.models.tgn import (
    IdentityMessage,
    MeanAggregator,
)

class PriceEncoder(torch.nn.Module):
    """Encodes price/market features into embeddings."""
    def __init__(self, input_dim=14, hidden_dim=64, output_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.1),  # LeakyReLU for gradient flow
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
            nn.LeakyReLU(0.1)
        )
    
    def forward(self, price_seq):
        # price_seq: (batch_size, 14) - [stock_feats(7), spy_feats(7)]
        return self.net(price_seq)

class GraphAttentionEmbedding(torch.nn.Module):
    def __init__(self, in_channels, out_channels, msg_dim, time_enc):
        super().__init__()
        self.time_enc = time_enc
        edge_dim = msg_dim + time_enc.out_channels + 1 # +1 for Label (masked)
        
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
    def __init__(self, in_channels, price_dim=32):
        super().__init__()
        # Input: (z + static + price) for src and dst = 2 * in_channels + 2 * price_dim
        total_input = in_channels * 2 + price_dim * 2
        self.net = nn.Sequential(
            nn.Linear(total_input, 128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, z_src, z_dst, s_src, s_dst, p_src, p_dst):
        # Concatenate: z + static + price for each node
        h_src = torch.cat([z_src, s_src, p_src], dim=-1)
        h_dst = torch.cat([z_dst, s_dst, p_dst], dim=-1)
        h_pair = torch.cat([h_src, h_dst], dim=-1)
        return self.net(h_pair)

class BasicTGN(torch.nn.Module):
    def __init__(self, num_nodes, raw_msg_dim, memory_dim, time_dim, embedding_dim, num_parties, num_states, num_chambers, use_price=True):
        super().__init__()
        self.num_nodes = num_nodes
        self.memory_dim = memory_dim
        self.time_dim = time_dim
        self.use_price = use_price
        
        # BASIC: With optional Price Encoder
        # Msg: [Amt, Buy, Gap, WinRate] = 4
        self.msg_dim = raw_msg_dim 
        
        # Price Encoder (Market Context)
        if self.use_price:
            self.price_encoder = PriceEncoder(input_dim=14, hidden_dim=64, output_dim=32)  # Scaled up
            price_dim = 32
        else:
            price_dim = 0 
        
        
        self.memory = TGNMemory(
            num_nodes=num_nodes,
            raw_msg_dim=self.msg_dim + price_dim,  # Augmented message includes price embedding
            memory_dim=memory_dim,
            time_dim=time_dim,
            message_module=IdentityMessage(self.msg_dim + price_dim, memory_dim, time_dim), 
            aggregator_module=MeanAggregator(),
        )
        
        
        self.gnn = GraphAttentionEmbedding(
            in_channels=memory_dim,
            out_channels=embedding_dim,
            msg_dim=self.msg_dim + price_dim,  # Edge features include price
            time_enc=self.memory.time_enc,
        )
        
        self.emb_party = nn.Embedding(num_parties + 1, 8) 
        self.emb_state = nn.Embedding(num_states + 1, 8)
        self.emb_chamber = nn.Embedding(num_chambers + 1, 8)
        
        # IDENTITY SHORTCUT REMOVED (Inductive Capability)
        # self.node_emb = nn.Embedding(num_nodes, 16)
        
        static_dim = 24 # 8+8+8
        
        # Predictor: takes (embedding_dim + static_dim) for z+static, plus price_dim
        self.predictor = LinkPredictor(embedding_dim + static_dim, price_dim=price_dim)

    def encode_static(self, x_static_idx):
        p = self.emb_party(x_static_idx[:, 0])
        s = self.emb_state(x_static_idx[:, 1])
        c = self.emb_chamber(x_static_idx[:, 2])
        return torch.cat([p, s, c], dim=-1)

    def forward(self, n_id, edge_index, edge_attr, price_seq=None):
        # DETACH memory to prevent inplace modification error during update_state
        # This stops gradients from flowing back into the memory buffer itself
        memory = self.memory.memory[n_id].clone().detach()
        
        # Augment edge_attr with price embeddings if available
        if self.use_price and price_seq is not None:
            # price_seq: (num_edges, 14)
            price_emb = self.price_encoder(price_seq)  # (num_edges, 16)
            # Concatenate with existing edge features
            edge_attr = torch.cat([edge_attr, price_emb], dim=-1)
            
        z = self.gnn(memory, edge_index, edge_attr)
        return z
