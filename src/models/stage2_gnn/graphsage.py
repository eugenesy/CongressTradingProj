import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, MessagePassing

class AuxEdgeSAGEConv(MessagePassing):
    """
    Custom GraphSAGE Convolution that natively incorporates multidimensional 
    edge attributes (like numerosity, amounts, and recency) during message passing.
    """
    def __init__(self, in_channels, out_channels, edge_dim):
        # 'mean' aggregation matches the GraphSAGE algorithm
        super().__init__(aggr='mean') 
        self.lin_l = nn.Linear(in_channels[0], out_channels)
        self.lin_r = nn.Linear(in_channels[1], out_channels)
        self.lin_msg = nn.Linear(in_channels[0] + edge_dim, out_channels)
        
    def forward(self, x, edge_index, edge_attr):
        # x is a tuple: (x_source, x_target)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        # Add the target node's own projected representation
        out = self.lin_r(x[1]) + out
        return out
        
    def message(self, x_j, edge_attr):
        # x_j is the source node features
        # Concatenate source node features and edge features before projecting
        msg = torch.cat([x_j, edge_attr], dim=-1)
        return F.relu(self.lin_msg(msg))


class BipartiteSAGEExtended(nn.Module):
    def __init__(self, pol_in_dim, comp_in_dim, hidden_dim=64, out_dim=32, 
                 num_states=60, num_sectors=20, num_industries=150):
        super(BipartiteSAGEExtended, self).__init__()
        
        # --- Categorical Embeddings ---
        self.state_emb = nn.Embedding(num_states, 8)
        self.sector_emb = nn.Embedding(num_sectors, 8)
        self.ind_emb = nn.Embedding(num_industries, 16)
        
        # --- Input Projections ---
        # Pol dim = continuous pol features + 8 (State embedding)
        self.pol_proj = nn.Linear(pol_in_dim + 8, hidden_dim)
        # Comp dim = continuous comp features + 8 (Sector) + 16 (Industry)
        self.comp_proj = nn.Linear(comp_in_dim + 8 + 16, hidden_dim)
        
        # --- Primary Trading Edge GNN ---
        self.sage_pol_to_comp = SAGEConv((hidden_dim, hidden_dim), hidden_dim)
        self.sage_comp_to_pol = SAGEConv((hidden_dim, hidden_dim), hidden_dim)
        
        # --- Auxiliary Edge GNNs ---
        # Edge dims based on graph_builder.py features:
        # lobby_strong/weak: [Interaction Count, Days Since] -> dim 2
        # campaign: [Log Amount, Donation Count, Days Since] -> dim 3
        self.sage_lobby_strong = AuxEdgeSAGEConv((hidden_dim, hidden_dim), hidden_dim, edge_dim=2)
        self.sage_lobby_weak = AuxEdgeSAGEConv((hidden_dim, hidden_dim), hidden_dim, edge_dim=2)
        self.sage_campaign = AuxEdgeSAGEConv((hidden_dim, hidden_dim), hidden_dim, edge_dim=3)
        
        # --- Output Heads ---
        self.out_pol = nn.Linear(hidden_dim, out_dim)
        self.out_comp = nn.Linear(hidden_dim, out_dim)

    def forward(self, x_pol, pol_state, x_comp, comp_sec, comp_ind, edge_index, **aux_edges):
        # 1. Base Node Embeddings
        emb_state = self.state_emb(pol_state)
        emb_sec = self.sector_emb(comp_sec)
        emb_ind = self.ind_emb(comp_ind)
        
        h_pol = torch.cat([x_pol, emb_state], dim=-1)
        h_pol = F.relu(self.pol_proj(h_pol))
        
        h_comp = torch.cat([x_comp, emb_sec, emb_ind], dim=-1)
        h_comp = F.relu(self.comp_proj(h_comp))
        
        # 2. Primary Message Passing (Transactions)
        # edge_index direction: [src=Pol, dst=Comp]
        h_comp_trade = F.relu(self.sage_pol_to_comp((h_pol, h_comp), edge_index))
        
        # Reverse edge index for Comp to Pol messages
        edge_index_rev = torch.stack([edge_index[1], edge_index[0]], dim=0)
        h_pol_trade = F.relu(self.sage_comp_to_pol((h_comp, h_pol), edge_index_rev))
        
        # 3. Auxiliary Message Passing (Lobbying & Campaign)
        # These are generally Company -> Politician influences
        h_pol_aux = torch.zeros_like(h_pol)
        
        if 'lobby_strong' in aux_edges:
            e_idx = aux_edges['lobby_strong']['edge_index']
            e_feat = aux_edges['lobby_strong']['features']
            # Reverse direction to pass message from Comp -> Pol
            e_idx_rev = torch.stack([e_idx[1], e_idx[0]], dim=0)
            h_pol_aux += F.relu(self.sage_lobby_strong((h_comp, h_pol), e_idx_rev, e_feat))
            
        if 'lobby_weak' in aux_edges:
            e_idx = aux_edges['lobby_weak']['edge_index']
            e_feat = aux_edges['lobby_weak']['features']
            e_idx_rev = torch.stack([e_idx[1], e_idx[0]], dim=0)
            h_pol_aux += F.relu(self.sage_lobby_weak((h_comp, h_pol), e_idx_rev, e_feat))
            
        if 'campaign' in aux_edges:
            e_idx = aux_edges['campaign']['edge_index']
            e_feat = aux_edges['campaign']['features']
            e_idx_rev = torch.stack([e_idx[1], e_idx[0]], dim=0)
            h_pol_aux += F.relu(self.sage_campaign((h_comp, h_pol), e_idx_rev, e_feat))
            
        # 4. Combine Representations
        # Politician node gets influenced by both standard trades and auxiliary influences
        h_pol_final = h_pol_trade + h_pol_aux
        
        # Companies are currently only influenced by the trades in this schema 
        # (Could expand auxiliary edges to be bidirectional later if needed)
        h_comp_final = h_comp_trade 
        
        # 5. Output Projection
        out_pol = self.out_pol(h_pol_final)
        out_comp = self.out_comp(h_comp_final)
        
        return out_pol, out_comp