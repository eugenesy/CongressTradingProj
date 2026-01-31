import torch
import torch.nn as nn
import pandas as pd
import numpy as np

class TimeEncoder(nn.Module):
    def __init__(self, dimension):
        super(TimeEncoder, self).__init__()
        self.dimension = dimension
        self.w = nn.Linear(1, dimension)
        self.p = nn.Linear(1, dimension)
        
    def forward(self, t):
        t = t.unsqueeze(1).float()
        return torch.cos(self.w(t)) + torch.sin(self.p(t))

class GAPTGN(nn.Module):
    def __init__(self, num_nodes, edge_feat_dim, pol_feat_dim, comp_feat_dim, 
                 time_dim=100, memory_dim=100, embedding_dim=100, device='cpu'):
        super(GAPTGN, self).__init__()
        self.num_nodes = num_nodes
        self.device = device
        self.memory_dim = memory_dim
        self.embedding_dim = embedding_dim
        
        # Time Encoder
        self.time_encoder = TimeEncoder(time_dim)
        
        # Memory
        self.memory = None
        self.last_update = None
        
        # Message Function
        self.msg_dim = memory_dim * 2 + time_dim + edge_feat_dim
        self.msg_encoder = nn.Sequential(
            nn.Linear(self.msg_dim, memory_dim),
            nn.ReLU(),
            nn.Linear(memory_dim, memory_dim)
        )
        
        # Memory Updater (GRU)
        self.memory_updater = nn.GRUCell(memory_dim, memory_dim)
        
        # Embedding / Fusion
        # Dynamic Projection Layers
        self.pol_proj = nn.Linear(pol_feat_dim, embedding_dim)
        self.comp_proj = nn.Linear(comp_feat_dim, embedding_dim)
        
        self.fusion = nn.Linear(memory_dim + embedding_dim, embedding_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        self.reset_memory()
        
    def reset_memory(self):
        self.memory = torch.zeros(self.num_nodes, self.memory_dim).to(self.device)
        self.last_update = torch.zeros(self.num_nodes).to(self.device)
        
    def detach_memory(self):
        self.memory = self.memory.detach()
        
    def forward(self, src, dst, t, msg, x_pol, x_comp):
        # 1. Embeddings
        mem_src = self.memory[src]
        mem_dst = self.memory[dst]
        
        # Project Politician Features
        # x_pol shape: [Num_Politicians, Pol_Feat_Dim]
        feat_src = self.pol_proj(x_pol[src])
            
        # Project Company Features
        # x_comp shape: [Num_Companies, Comp_Feat_Dim]
        # dst indices are global (OFFSET by num_pols)
        offset = x_pol.shape[0]
        comp_idx = dst - offset
        
        # Safety clamp
        comp_idx = torch.clamp(comp_idx, 0, x_comp.shape[0]-1)
        feat_dst = self.comp_proj(x_comp[comp_idx])
            
        # Fusion: Combine Memory + Static Features
        emb_src = self.fusion(torch.cat([mem_src, feat_src], dim=1))
        emb_dst = self.fusion(torch.cat([mem_dst, feat_dst], dim=1))
        
        # 2. Decode
        combined = torch.cat([emb_src, emb_dst], dim=1)
        logits = self.decoder(combined)
        
        return torch.sigmoid(logits), None

    def update_memory(self, src, dst, t, msg):
        """
        Updates the memory of source and destination nodes using the latest interaction.
        This is a simplified TGN memory update (Last Message Only).
        """
        # 1. Get current memory
        mem_src = self.memory[src]
        mem_dst = self.memory[dst]
        
        # 2. Compute Time Encoding
        t_enc = self.time_encoder(t)
        
        # 3. Construct Raw Message: [Memory_Src || Memory_Dst || Time || Edge_Feat]
        # Note: We use the memory *before* the update to compute the message
        raw_msg = torch.cat([mem_src, mem_dst, t_enc, msg], dim=1)
        
        # 4. Encode Message
        encoded_msg = self.msg_encoder(raw_msg)
        
        # 5. Update Memory using GRU
        # GRU expects input (msg) and hidden state (current_memory)
        # We update src and dst separately
        
        # Update Source Memory
        updated_mem_src = self.memory_updater(encoded_msg, mem_src)
        
        # Update Destination Memory
        updated_mem_dst = self.memory_updater(encoded_msg, mem_dst)
        
        # 6. Detach to prevent backprop through entire history (Truncated BPTT)
        self.memory[src] = updated_mem_src.detach()
        self.memory[dst] = updated_mem_dst.detach()
        
        # Update last update time
        self.last_update[src] = t
        self.last_update[dst] = t