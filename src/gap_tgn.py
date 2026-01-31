import torch
import torch.nn as nn
from torch_geometric.nn import TGNMemory, TransformerConv
from torch_geometric.nn.models.tgn import MeanAggregator

class LearnableMessage(torch.nn.Module):
    def __init__(self, raw_msg_dim, memory_dim, time_dim):
        super().__init__()
        self.out_channels = memory_dim
        input_dim = memory_dim * 2 + raw_msg_dim + time_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, memory_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(memory_dim, memory_dim),
        )

    def forward(self, z_src, z_dst, raw_msg, t_enc):
        h = torch.cat([z_src, z_dst, raw_msg, t_enc], dim=-1)
        return self.mlp(h)

class GraphAttentionEmbedding(torch.nn.Module):
    def __init__(self, in_channels, out_channels, msg_dim, time_enc):
        super().__init__()
        self.time_enc = time_enc
        # FIX: Explicitly match dimensions. 
        # msg_dim includes (Edge_Feats + Price_Emb)
        # edge_dim = (Edge_Feats + Price_Emb) + Time_Emb
        edge_dim = msg_dim + time_enc.out_channels
        
        self.conv1 = TransformerConv(in_channels, out_channels // 4, heads=4, dropout=0.1, edge_dim=edge_dim)
        self.conv2 = TransformerConv(out_channels, out_channels // 4, heads=4, dropout=0.1, edge_dim=edge_dim)
        self.dropout = torch.nn.Dropout(0.1)
        self.relu = torch.nn.ReLU()

    def forward(self, x, edge_index, edge_attr):
        x = self.relu(self.conv1(x, edge_index, edge_attr))
        x = self.dropout(x)
        x = self.conv2(x, edge_index, edge_attr)
        return x

class Decoder(torch.nn.Module):
    def __init__(self, in_channels, price_dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(in_channels * 2 + price_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        self.net = nn.Sequential(
            nn.BatchNorm1d(in_channels * 2 + price_dim),
            nn.Linear(in_channels * 2 + price_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),                 
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, z_src, z_dst, d_src, d_dst, p_context):
        # z = Graph Embedding (Memory + Neighbors)
        # d = Dynamic Node Features (Bio, Econ, Financials, etc.)
        
        # Fuse Graph + Dynamic
        h_src = torch.cat([z_src, d_src], dim=-1)
        h_dst = torch.cat([z_dst, d_dst], dim=-1)
        h_graph = torch.cat([h_src, h_dst], dim=-1)
        
        # Gating with Price Context
        s_gate = self.gate(torch.cat([h_graph, p_context], dim=-1))
        
        h_graph_w = h_graph * s_gate
        h_mkt_w = p_context * (1 - s_gate)
        
        return self.net(torch.cat([h_graph_w, h_mkt_w], dim=-1))

class GAPTGN(torch.nn.Module):
    def __init__(self, num_nodes, edge_feat_dim, pol_feat_dim, comp_feat_dim, 
                 memory_dim=100, time_dim=100, embedding_dim=100, device='cpu'):
        super().__init__()
        self.device = device
        self.price_emb_dim = 32
        
        # 1. Price Encoder
        self.price_encoder = nn.Sequential(
            nn.BatchNorm1d(14),
            nn.Linear(14, self.price_emb_dim),
            nn.ReLU(),
            nn.Linear(self.price_emb_dim, self.price_emb_dim),
        )
        
        self.augmented_msg_dim = edge_feat_dim + self.price_emb_dim
        
        # 2. Memory Module (Standard TGNMemory)
        self.memory = TGNMemory(
            num_nodes=num_nodes,
            raw_msg_dim=self.augmented_msg_dim, 
            memory_dim=memory_dim,
            time_dim=time_dim,
            message_module=LearnableMessage(self.augmented_msg_dim, memory_dim, time_dim),
            aggregator_module=MeanAggregator(),
        ).to(device)
        
        # 3. Graph Attention (TransformerConv)
        self.gnn = GraphAttentionEmbedding(
            in_channels=memory_dim,
            out_channels=embedding_dim,
            msg_dim=self.augmented_msg_dim,
            time_enc=self.memory.time_enc,
        ).to(device)
        
        # 4. Dynamic Feature Projections
        # These handle the DYNAMIC node sizes from config
        self.pol_proj = nn.Linear(pol_feat_dim, 16) 
        self.comp_proj = nn.Linear(comp_feat_dim, 16)
        
        static_dim = 16 
        
        # 5. Decoder
        self.predictor = Decoder(embedding_dim + static_dim, self.price_emb_dim).to(device)

    def forward(self, n_id, edge_index, edge_attr):
        """
        Compute Graph Embeddings (z) for a set of nodes (n_id) and their sampled subgraph.
        """
        memory = self.memory.memory[n_id]
        
        # Handle case with no edges (e.g. start of timeline)
        if edge_index.numel() == 0:
            # Fallback: Just return projected memory if no edges exist to convolve over
            # Since TransformerConv might error or return zeros, we can just pass memory
            # But normally TransformerConv handles empty edges by returning bias/zeros.
            # We'll stick to the standard call.
            pass

        z = self.gnn(memory, edge_index, edge_attr)
        return z

    def encode_dynamic(self, x_pol, x_comp, src_indices, dst_indices, num_pols):
        """
        Projects the raw dynamic features into the embedding space.
        """
        # Politician Features
        p_feat = self.pol_proj(x_pol[src_indices])
        
        # Company Features
        # Map global dst indices (which are offset) back to 0-indexed company array
        comp_idx = dst_indices - num_pols
        # Safety clamp
        comp_idx = torch.clamp(comp_idx, 0, x_comp.shape[0]-1)
        c_feat = self.comp_proj(x_comp[comp_idx])
        
        return p_feat, c_feat

    def get_price_embedding(self, price_seq):
        return self.price_encoder(price_seq)