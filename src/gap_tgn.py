import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TGNMemory, TransformerConv
# NEW: Import the specific Aggregator class
from torch_geometric.nn.models.tgn import LastAggregator

class IdentityMessage(torch.nn.Module):
    """
    A message module that concatenates memory, edge features, and time encoding.
    Used by TGNMemory to construct the input for the memory updater.
    """
    def __init__(self, raw_msg_dim, memory_dim, time_dim):
        super(IdentityMessage, self).__init__()
        # The output dimension is the concatenation of:
        # Source Memory (memory_dim) + Dest Memory (memory_dim) + Raw Message (raw_msg_dim) + Time Encoding (time_dim)
        self.out_channels = raw_msg_dim + 2 * memory_dim + time_dim

    def forward(self, z_src, z_dst, raw_msg, t_enc):
        return torch.cat([z_src, z_dst, raw_msg, t_enc], dim=-1)

class GAPTGN(nn.Module):
    def __init__(self, edge_feat_dim, price_seq_dim, hidden_dim, num_nodes, num_parties, num_states):
        """
        GAP-TGN Model Definition.
        """
        super(GAPTGN, self).__init__()
        
        self.hidden_dim = hidden_dim
        
        # 1. TGN Memory
        self.memory = TGNMemory(
            num_nodes=num_nodes,
            raw_msg_dim=edge_feat_dim,
            memory_dim=hidden_dim,
            time_dim=hidden_dim,
            message_module=IdentityMessage(edge_feat_dim, hidden_dim, hidden_dim),
            # FIX: Pass the instantiated class, not a string
            aggregator_module=LastAggregator()
        )
        
        # 2. Graph Embedding (GNN)
        self.gnn = TransformerConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            heads=2,
            dropout=0.1,
            edge_dim=edge_feat_dim
        )
        
        # 3. Static Node Embeddings
        self.party_emb = nn.Embedding(num_parties, 16)
        self.state_emb = nn.Embedding(num_states, 16)
        self.static_proj = nn.Linear(32, hidden_dim)
        
        # 4. Price Sequence Encoder
        self.price_lstm = nn.LSTM(input_size=1, hidden_size=32, batch_first=True)
        self.price_proj = nn.Linear(32, hidden_dim)
        
        # 5. Prediction Head
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim * 3, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, src, dst, t, msg, price_seq, trade_t, x_static):
        """
        Forward pass for training/inference.
        """
        
        # A. Update Memory
        self.memory.update_state(src, dst, t, msg)
        
        # 2. Retrieve updated memory
        mem_src = self.memory.memory[src]
        mem_dst = self.memory.memory[dst]
        
        # B. Static Features
        party_idx = x_static[src, 0]
        state_idx = x_static[src, 1]
        
        static_feat = torch.cat([
            self.party_emb(party_idx),
            self.state_emb(state_idx)
        ], dim=1)
        src_static = self.static_proj(static_feat)
        
        # C. Price Encoder
        p_seq = price_seq.unsqueeze(-1)
        _, (h_n, _) = self.price_lstm(p_seq)
        price_emb = self.price_proj(h_n[-1])
        
        # D. Feature Fusion
        pol_ctx = mem_src + src_static
        comp_ctx = mem_dst + price_emb
        
        combined = torch.cat([pol_ctx, comp_ctx, price_emb], dim=1)
        
        # E. Predict
        out = self.predictor(combined)
        
        return out.squeeze(-1)