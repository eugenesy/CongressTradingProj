import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TGNMemory, TransformerConv
from torch_geometric.nn.models.tgn import LastAggregator

class IdentityMessage(torch.nn.Module):
    def __init__(self, raw_msg_dim, memory_dim, time_dim):
        super(IdentityMessage, self).__init__()
        self.out_channels = raw_msg_dim + 2 * memory_dim + time_dim

    def forward(self, z_src, z_dst, raw_msg, t_enc):
        return torch.cat([z_src, z_dst, raw_msg, t_enc], dim=-1)

class GAPTGN(nn.Module):
    def __init__(self, edge_feat_dim, pol_dim, comp_dim, price_seq_dim, hidden_dim, num_nodes):
        """
        GAP-TGN Model with Dynamic Node Features.
        pol_dim: Dimension of dynamic politician features (Bio + Ideology etc.)
        comp_dim: Dimension of dynamic company features (SIC + Financials)
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
        
        # 3. Dynamic Node Feature Encoders (Replacing Static Embeddings)
        # We project the raw feature vectors into the hidden space
        self.pol_encoder = nn.Linear(pol_dim, hidden_dim)
        self.comp_encoder = nn.Linear(comp_dim, hidden_dim)
        
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

    def forward(self, src, dst, t, msg, price_seq, trade_t, x_pol, x_comp):
        """
        x_pol: Tensor [Batch_Size, Pol_Feature_Dim] - Dynamic features for source nodes
        x_comp: Tensor [Batch_Size, Comp_Feature_Dim] - Dynamic features for dest nodes
        """
        
        # A. Update Memory (using Edge Features only)
        self.memory.update_state(src, dst, t, msg)
        
        # Retrieve updated memory
        mem_src = self.memory.memory[src]
        mem_dst = self.memory.memory[dst]
        
        # B. Encode Dynamic Node Features
        # These features naturally update because the data loader passes the 
        # specific x_pol/x_comp row corresponding to the event time.
        pol_emb = self.pol_encoder(x_pol)
        comp_emb = self.comp_encoder(x_comp)
        
        # C. Price Encoder
        p_seq = price_seq.unsqueeze(-1)
        _, (h_n, _) = self.price_lstm(p_seq)
        price_emb = self.price_proj(h_n[-1])
        
        # D. Feature Fusion
        # Context = Memory (History) + Dynamic Features (Current State)
        pol_ctx = mem_src + pol_emb
        comp_ctx = mem_dst + comp_emb + price_emb
        
        combined = torch.cat([pol_ctx, comp_ctx, price_emb], dim=1)
        
        # E. Predict
        out = self.predictor(combined)
        
        return out.squeeze(-1)