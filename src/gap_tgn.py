import torch
import torch.nn as nn
from torch_geometric.nn import TGNMemory,TransformerConv

class PriceEncoder(nn.Module):
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

class TimeEncoder(nn.Module):
    def __init__(self, dimension):
        super(TimeEncoder, self).__init__()
        self.dimension = dimension
        self.w = nn.Linear(1, dimension)
        self.p = nn.Linear(1, dimension)
        
    def forward(self, t):
        t = t.unsqueeze(1).float()
        return torch.cos(self.w(t)) + torch.sin(self.p(t))

class LearnableMessage(torch.nn.Module):
    """
    Complex Message Function from models_tgn.py.
    Fuses Source Mem, Dest Mem, Time, and Edge Features into a message.
    """
    def __init__(self, raw_msg_dim, memory_dim, time_dim):
        super().__init__()
        input_dim = memory_dim * 2 + raw_msg_dim + time_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, memory_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(memory_dim, memory_dim),
        )

    def forward(self, mem_src, mem_dst, raw_msg, t_enc):
        h = torch.cat([mem_src, mem_dst, raw_msg, t_enc], dim=-1)
        return self.mlp(h)
    
class GraphAttentionEmbedding(torch.nn.Module):
    def __init__(self, in_channels, out_channels, msg_dim, time_dim):
        super().__init__()
        # Edge dim = raw_msg + price_emb + time_encoding + (optional padding)
        edge_dim = msg_dim + time_dim 
        self.conv1 = TransformerConv(in_channels, out_channels // 4, heads=4, dropout=0.1, edge_dim=edge_dim)
        self.conv2 = TransformerConv(out_channels, out_channels // 4, heads=4, dropout=0.1, edge_dim=edge_dim)
        self.dropout = torch.nn.Dropout(0.1)
        self.relu = torch.nn.ReLU()

    def forward(self, x, edge_index, edge_attr):
        x = self.relu(self.conv1(x, edge_index, edge_attr))
        x = self.dropout(x)
        x = self.conv2(x, edge_index, edge_attr)
        return x

class LinkPredictor(torch.nn.Module):
    """
    Deep Interaction Decoder with Feature Gating.
    Adapted to use x_pol (Politician) and x_comp (Company/Market) features.
    """
    def __init__(self, memory_dim, pol_dim, comp_dim):
        super().__init__()
        
        # 1. Project inputs to common embedding space
        self.pol_proj = nn.Linear(pol_dim, memory_dim)
        self.comp_proj = nn.Linear(comp_dim, memory_dim)
        
        # Dimensions after projection
        emb_dim = memory_dim 
        
        # 2. Gating module (Takes Pol_Emb + Comp_Emb + Market_Context)
        # We treat x_comp (company data) as the "Market Price" context for the gate
        gate_input_dim = (emb_dim * 2) + memory_dim # (Z_src + Z_dst) + Comp_Feats(Projected)
        
        self.gate = nn.Sequential(
            nn.Linear(gate_input_dim * 2, 32), # *2 because we concat mem + static
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # 3. Main Decoder
        # Input: (Mem_Src + Pol_Static) + (Mem_Dst + Comp_Static)
        decoder_input_dim = (memory_dim + memory_dim) * 2 
        
        self.net = nn.Sequential(
            nn.BatchNorm1d(decoder_input_dim),
            nn.Linear(decoder_input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),                 
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, src, dst, t, msg, x_pol, x_comp, edge_index, edge_attr, price_seq=None):
        
        # 1. Apply Graph Attention on the current Memory
        # We pass the whole memory table to the GNN to update embeddings based on graph structure
        
        z = self.gnn(self.memory, edge_index, edge_attr)
        
        # 2. Extract specific node embeddings for the target batch
        z_src = z[src]
        z_dst = z[dst]
        
        # 3. Pass GNN-refined embeddings + Dynamic Features to Predictor
        # Note: Price usage here is optional if you want it in the decoder, 
        # but the critical part is putting it in the memory update (step 4 below).
        logits = self.predictor(z_src, z_dst, x_pol, x_comp)
        return torch.sigmoid(logits), None
    
class GAPTGN(nn.Module):
    def __init__(self, num_nodes, edge_feat_dim, pol_feat_dim, comp_feat_dim, 
                 time_dim=100, memory_dim=100, embedding_dim=100, device='cpu'):
        super(GAPTGN, self).__init__()
        self.num_nodes = num_nodes
        self.device = device
        self.memory_dim = memory_dim
        self.embedding_dim = embedding_dim
        self.time_dim = time_dim

        # 1. Encoders
        self.time_encoder = TimeEncoder(time_dim)
        self.price_encoder = PriceEncoder(input_dim=14, hidden_dim=32) 
        
        # Note: edge_feat_dim must match the dimension of (raw_edge_features + time_embedding)
        self.gnn = GraphAttentionEmbedding(
            in_channels=memory_dim,
            out_channels=embedding_dim,
            msg_dim=edge_feat_dim + 32, # +32 accounts for the price embedding we will add later
            time_dim=time_dim
        )
        
        # 2. Message Module
        # We increase raw_msg_dim by 32 because we will concatenate price embeddings to messages
        self.msg_module = LearnableMessage(edge_feat_dim + 32, memory_dim, time_dim)
        
        # 3. Memory Updater (Manual GRU)
        self.memory_updater = nn.GRUCell(memory_dim, memory_dim)
        
        # 4. Predictor (Decoder)
        # Input is now embedding_dim (from GNN) instead of memory_dim
        self.predictor = LinkPredictor(
            memory_dim=embedding_dim, 
            pol_dim=pol_feat_dim, 
            comp_dim=comp_feat_dim
        )
        
        self.memory = None
        self.last_update = None
        self.reset_memory()

    def reset_memory(self):
        self.memory = torch.zeros(self.num_nodes, self.memory_dim).to(self.device)
        self.last_update = torch.zeros(self.num_nodes).to(self.device)
        
    def detach_memory(self):
        self.memory = self.memory.detach()
        
    def forward(self, src, dst, t, msg, x_pol, x_comp):
        """
        Returns probabilities using current memory and batch features.
        Does NOT update memory (call update_memory for that).
        """
        mem_src = self.memory[src]
        mem_dst = self.memory[dst]
        
        # Use the Deep Gated Predictor
        logits = self.predictor(mem_src, mem_dst, x_pol, x_comp)
        
        return torch.sigmoid(logits), None

    def update_memory(self, src, dst, t, msg, price_seq):
        # 1. Embed Price
        # price_seq: (Batch, 14) -> (Batch, 32)
        price_emb = self.price_encoder(price_seq)
        
        # 2. Concatenate Message + Price
        # msg: (Batch, edge_dim)
        # augmented_msg: (Batch, edge_dim + 32)
        augmented_msg = torch.cat([msg, price_emb], dim=-1)

        # 3. Standard TGN Update
        mem_src = self.memory[src]
        mem_dst = self.memory[dst]
        t_enc = self.time_encoder(t)
        
        # Use the augmented message
        encoded_msg = self.msg_module(mem_src, mem_dst, augmented_msg, t_enc)
        
        updated_mem_src = self.memory_updater(encoded_msg, mem_src)
        updated_mem_dst = self.memory_updater(encoded_msg, mem_dst)
        
        self.memory[src] = updated_mem_src.detach()
        self.memory[dst] = updated_mem_dst.detach()
        self.last_update[src] = t.detach()
        self.last_update[dst] = t.detach()
        """
        Updates the memory state using the provided batch of interactions.
        Should be called AFTER forward/loss computation for the batch.
        """
        # 1. Get Memory (Detached for truncated BPTT)
        mem_src = self.memory[src]
        mem_dst = self.memory[dst]
        
        # 2. Time Encoding
        t_enc = self.time_encoder(t)
        
        # 3. Compute Complex Messages
        # Note: We use the same message logic for both Src and Dst 
        # (assuming undirected information flow or shared context)
        encoded_msg = self.msg_module(mem_src, mem_dst, msg, t_enc)
        
        # 4. Update Memory with GRU
        updated_mem_src = self.memory_updater(encoded_msg, mem_src)
        updated_mem_dst = self.memory_updater(encoded_msg, mem_dst)
        
        # 5. Commit updates
        self.memory[src] = updated_mem_src.detach()
        self.memory[dst] = updated_mem_dst.detach()
        self.last_update[src] = t.detach()
        self.last_update[dst] = t.detach()
    
    def encode_dynamic(self, x_pol_batch, x_comp_batch, src, dst, num_pols):
        """
        Since x_pol and x_comp are now correctly batched by the TemporalDataLoader,
        we simply return them. The src/dst args are kept for compatibility with 
        the script call signature but are not needed for lookup anymore.
        """
        return x_pol_batch, x_comp_batch