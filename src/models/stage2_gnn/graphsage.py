import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class BipartiteSAGEExtended(nn.Module):
    def __init__(self, 
                 pol_in_dim: int, 
                 comp_in_dim: int, 
                 hidden_dim: int, 
                 out_dim: int, 
                 num_states: int, 
                 num_sectors: int, 
                 num_industries: int,
                 state_emb_dim: int = 8,
                 sector_emb_dim: int = 8,
                 ind_emb_dim: int = 8,
                 dropout: float = 0.2):
        """
        Args:
            pol_in_dim: Dynamically calculated sum of the active politician feature dimensions 
                        (e.g., performance stats + chamber + party + ideology + committees).
            comp_in_dim: Dynamically calculated sum of active company feature dimensions 
                         (e.g., SIC one-hot + SEC financials).
            hidden_dim: The standard hidden dimension size for the GNN layers.
            out_dim: The output dimension (e.g., for link prediction or node classification).
        """
        super().__init__()
        
        # --- Learned Embeddings ---
        # We preserve these as learned embeddings as requested
        self.state_emb = nn.Embedding(num_states, state_emb_dim)
        self.sector_emb = nn.Embedding(num_sectors, sector_emb_dim)
        self.industry_emb = nn.Embedding(num_industries, ind_emb_dim)
        
        # --- Projection Layers ---
        # Calculate total input dimensions after concatenating the static embeddings 
        # with the dynamic continuous/one-hot features
        self.pol_total_dim = pol_in_dim + state_emb_dim
        self.comp_total_dim = comp_in_dim + sector_emb_dim + ind_emb_dim
        
        # Project heterogeneous features to the common hidden_dim
        self.pol_proj = nn.Linear(self.pol_total_dim, hidden_dim)
        self.comp_proj = nn.Linear(self.comp_total_dim, hidden_dim)
        
        # --- Message Passing Layers ---
        # Standard SAGEConv. Since we unify the node representations via projection, 
        # we can process them jointly in the bipartite graph.
        self.conv1 = SAGEConv((hidden_dim, hidden_dim), hidden_dim)
        self.conv2 = SAGEConv((hidden_dim, hidden_dim), hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.predictor = nn.Linear(hidden_dim * 2, out_dim)

    def forward(self, 
                x_pol_dyn: torch.Tensor, pol_state_idx: torch.Tensor, 
                x_comp_dyn: torch.Tensor, comp_sector_idx: torch.Tensor, comp_ind_idx: torch.Tensor, 
                edge_index: torch.Tensor):
        """
        x_pol_dyn: Shape (num_pols, pol_in_dim) - Concatenated dynamic attributes & bio features.
        pol_state_idx: Shape (num_pols,) - Categorical index for State.
        x_comp_dyn: Shape (num_comps, comp_in_dim) - Concatenated financial/SIC attributes.
        comp_sector_idx: Shape (num_comps,) - Categorical index for Sector.
        comp_ind_idx: Shape (num_comps,) - Categorical index for Industry.
        edge_index: Shape (2, num_edges) - Bipartite connectivity.
        """
        
        # 1. Look up categorical embeddings
        state_embeds = self.state_emb(pol_state_idx)
        sector_embeds = self.sector_emb(comp_sector_idx)
        ind_embeds = self.industry_emb(comp_ind_idx)
        
        # 2. Concatenate the dynamic attributes with the learned embeddings
        pol_features = torch.cat([x_pol_dyn, state_embeds], dim=-1)
        comp_features = torch.cat([x_comp_dyn, sector_embeds, ind_embeds], dim=-1)
        
        # 3. Project to the common hidden dimension
        h_pol = F.relu(self.pol_proj(pol_features))
        h_comp = F.relu(self.comp_proj(comp_features))
        
        h_pol = self.dropout(h_pol)
        h_comp = self.dropout(h_comp)
        
        # 4. Joint Message Passing
        # PyG handles bipartite graphs natively if we stack the nodes and align the edge_index 
        # (assuming node indices 0 to N-1 are pols, and N to N+M-1 are companies)
        h_total = torch.cat([h_pol, h_comp], dim=0)
        
        h_out = F.relu(self.conv1(h_total, edge_index))
        h_out = self.dropout(h_out)
        h_out = self.conv2(h_out, edge_index)
        
        # 5. Split back into respective representations
        num_pols = h_pol.size(0)
        h_pol_out = h_out[:num_pols]
        h_comp_out = h_out[num_pols:]
        
        return h_pol_out, h_comp_out