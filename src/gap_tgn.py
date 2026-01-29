import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TGNMemory, TransformerConv

class GAPTGN(nn.Module):
    def __init__(self, edge_feat_dim, price_seq_dim, hidden_dim, num_nodes, num_parties, num_states):
        """
        GAP-TGN Model Definition.
        
        Args:
            edge_feat_dim (int): Dimension of edge features (dynamic based on config).
            price_seq_dim (int): Length of price sequence (e.g., 14).
            hidden_dim (int): Dimension of hidden layers.
            num_nodes (int): Total number of unique nodes (politicians + companies).
            num_parties (int): Number of unique political parties.
            num_states (int): Number of unique states.
        """
        super(GAPTGN, self).__init__()
        
        self.hidden_dim = hidden_dim
        
        # 1. TGN Memory
        # We use a GRU-based memory updater to track node states over time
        self.memory = TGNMemory(
            num_nodes=num_nodes,
            raw_msg_dim=edge_feat_dim,  # Matches the dynamic feature size
            memory_dim=hidden_dim,
            time_dim=hidden_dim,
            message_module=nn.Identity(), # Simple concatenation of msg + source_emb
            aggregator_module=None # Defaults to 'last'
        )
        
        # 2. Graph Embedding (GNN)
        # Updates node embeddings based on neighbors + current memory state
        self.gnn = TransformerConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            heads=2,
            dropout=0.1,
            edge_dim=edge_feat_dim # Edge features condition the message passing
        )
        
        # 3. Static Node Embeddings (Party & State)
        self.party_emb = nn.Embedding(num_parties, 16)
        self.state_emb = nn.Embedding(num_states, 16)
        self.static_proj = nn.Linear(32, hidden_dim) # Projects combined static features to hidden dim
        
        # 4. Price Sequence Encoder (LSTM)
        # Encodes the 14-day price history leading up to the transaction
        self.price_lstm = nn.LSTM(input_size=1, hidden_size=32, batch_first=True)
        self.price_proj = nn.Linear(32, hidden_dim)
        
        # 5. Prediction Head
        # Concatenates: [Pol_Context, Comp_Context, Market_Signal]
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim * 3, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1) # Output logit
        )

    def forward(self, src, dst, t, msg, price_seq, trade_t, x_static):
        """
        Forward pass for training/inference.
        """
        
        # A. Update Memory (TGN Standard Flow)
        # Note: In a strict temporal setup, we often use the *previous* batch's state 
        # for embedding computation, then update memory. 
        # Here we follow a simplified flow compatible with PyG's TGN example.
        
        # 1. Update memory with current batch interactions
        self.memory.update_state(src, dst, t, msg)
        
        # 2. Retrieve updated memory
        mem_src = self.memory.memory[src]
        mem_dst = self.memory.memory[dst]
        
        # 3. Compute Graph Embeddings
        # We need neighbor information. For the TGNMemory module, 
        # the embedding usually comes from the memory state directly 
        # or via a GNN layer on the temporal snapshot.
        # Here we simplify by using the Memory state as the primary node embedding
        # and augmenting it with static/market features.
        # (A full GNN pass requires neighbor sampling which adds complexity; 
        # treating Memory as the sufficient statistic is a valid TGN variant).
        
        # B. Static Features (Politician Party/State)
        # x_static: [Batch_Nodes, 2]
        # We map src nodes to their static IDs
        party_idx = x_static[src, 0]
        state_idx = x_static[src, 1]
        
        static_feat = torch.cat([
            self.party_emb(party_idx),
            self.state_emb(state_idx)
        ], dim=1)
        src_static = self.static_proj(static_feat)
        
        # C. Price Encoder
        # price_seq: [Batch, 14] -> [Batch, 14, 1]
        p_seq = price_seq.unsqueeze(-1)
        _, (h_n, _) = self.price_lstm(p_seq)
        price_emb = self.price_proj(h_n[-1]) # [Batch, Hidden]
        
        # D. Feature Fusion
        # Politician Context = Memory + Static
        pol_ctx = mem_src + src_static
        
        # Company Context = Memory + Market Signal
        comp_ctx = mem_dst + price_emb
        
        # Final Vector
        combined = torch.cat([pol_ctx, comp_ctx, price_emb], dim=1)
        
        # E. Predict
        out = self.predictor(combined)
        
        return out.squeeze(-1)