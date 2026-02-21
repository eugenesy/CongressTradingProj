import torch
import torch.nn as nn
from torch_geometric.nn import TGNMemory, TransformerConv
from torch_geometric.nn.models.tgn import MeanAggregator

class LearnableMessage(torch.nn.Module):
    def __init__(self, raw_msg_dim, memory_dim, time_dim, num_edge_types=4, edge_type_dim=8):
        super().__init__()
        self.out_channels = memory_dim
        
        # NEW: Embedding layer for Edge Types (0=Trade, 1=Sponsor, 2=Vote, 3=Donation)
        self.edge_type_emb = nn.Embedding(num_edge_types, edge_type_dim)
        
        # Calculate new dimension: Replace 1 categorical feature with an 8-dim embedding
        self.processed_msg_dim = (raw_msg_dim - 1) + edge_type_dim
        input_dim = memory_dim * 2 + self.processed_msg_dim + time_dim
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, memory_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(memory_dim, memory_dim),
        )

    def forward(self, z_src, z_dst, raw_msg, t_enc):
        # raw_msg schema: [Amt, is_buy, gap, edge_type, ...price_features]
        cont_features = raw_msg[:, :3]
        edge_type = raw_msg[:, 3].long()
        rest_features = raw_msg[:, 4:] if raw_msg.shape[1] > 4 else torch.empty((raw_msg.shape[0], 0), device=raw_msg.device)
        
        # Encode the edge type
        e_emb = self.edge_type_emb(edge_type)
        
        # Re-assemble the message with the rich embedding
        processed_msg = torch.cat([cont_features, rest_features, e_emb], dim=-1)
        
        h = torch.cat([z_src, z_dst, processed_msg, t_enc], dim=-1)
        return self.mlp(h)

class GraphAttentionEmbedding(torch.nn.Module):
    def __init__(self, in_channels, out_channels, msg_dim, time_enc, num_edge_types=4, edge_type_dim=8):
        super().__init__()
        self.time_enc = time_enc
        
        # NEW: Embedding layer for Edge Types
        self.edge_type_emb = nn.Embedding(num_edge_types, edge_type_dim)
        
        self.processed_msg_dim = (msg_dim - 1) + edge_type_dim
        # The +2 accounts for downstream concatenations from the training script
        edge_dim = self.processed_msg_dim + time_enc.out_channels + 2 
        
        self.conv1 = TransformerConv(in_channels, out_channels // 4, heads=4, dropout=0.1, edge_dim=edge_dim)
        self.conv2 = TransformerConv(out_channels, out_channels // 4, heads=4, dropout=0.1, edge_dim=edge_dim)
        self.dropout = torch.nn.Dropout(0.1)
        self.relu = torch.nn.ReLU()

    def forward(self, x, edge_index, edge_attr):
        # edge_attr schema: [Amt, is_buy, gap, edge_type, ...price_features, ...time_enc, ...etc]
        cont_features = edge_attr[:, :3]
        edge_type = edge_attr[:, 3].long()
        rest_features = edge_attr[:, 4:] if edge_attr.shape[1] > 4 else torch.empty((edge_attr.shape[0], 0), device=edge_attr.device)
        
        # Encode the edge type
        e_emb = self.edge_type_emb(edge_type)
        
        # Re-assemble the edge attributes for the attention convolution
        processed_attr = torch.cat([cont_features, rest_features, e_emb], dim=-1)
        
        x = self.relu(self.conv1(x, edge_index, processed_attr))
        x = self.dropout(x)
        x = self.conv2(x, edge_index, processed_attr)
        return x

class Decoder(torch.nn.Module):
    def __init__(self, in_channels, price_dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(in_channels, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        self.net = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.Linear(in_channels, 128),
            nn.ReLU(),
            nn.Dropout(0.3),                 
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, z_src, z_dst, s_src, s_dst, p_context):
        # s_src/s_dst now contain the FUSED node features (Static + Dynamic)
        h_src = torch.cat([z_src, s_src], dim=-1)
        h_dst = torch.cat([z_dst, s_dst], dim=-1)
        h_graph = torch.cat([h_src, h_dst], dim=-1)
        
        # Concat graph representation with price context
        full_input = torch.cat([h_graph, p_context], dim=-1)
        
        s_gate = self.gate(full_input)
        
        h_graph_w = h_graph * s_gate
        h_mkt_w = p_context * (1 - s_gate)
        
        return self.net(torch.cat([h_graph_w, h_mkt_w], dim=-1))

class ResearchTGN(torch.nn.Module):
    def __init__(self, num_nodes, raw_msg_dim, memory_dim, time_dim, embedding_dim, 
                 num_parties, num_states, 
                 pol_dim=0, comp_dim=0):
        super().__init__()
        self.embedding_dim = embedding_dim  # STORE THIS for forward usage
        self.price_emb_dim = 32
        
        self.price_encoder = nn.Sequential(
            nn.BatchNorm1d(14),
            nn.Linear(14, self.price_emb_dim),
            nn.ReLU(),
            nn.Linear(self.price_emb_dim, self.price_emb_dim),
        )
        
        self.augmented_msg_dim = raw_msg_dim + self.price_emb_dim
        
        self.memory = TGNMemory(
            num_nodes=num_nodes,
            raw_msg_dim=self.augmented_msg_dim, 
            memory_dim=memory_dim,
            time_dim=time_dim,
            message_module=LearnableMessage(self.augmented_msg_dim, memory_dim, time_dim),
            aggregator_module=MeanAggregator(),
        )
        
        self.gnn = GraphAttentionEmbedding(
            in_channels=memory_dim,
            out_channels=embedding_dim,
            msg_dim=self.augmented_msg_dim,
            time_enc=self.memory.time_enc,
        )
        
        # --- Static Embeddings (Always present) ---
        self.emb_party = nn.Embedding(num_parties + 1, 8) 
        self.emb_state = nn.Embedding(num_states + 1, 8)
        static_dim = 16
        
        # --- Dynamic Projections (Optional) ---
        self.pol_dim = pol_dim
        self.comp_dim = comp_dim
        
        has_dynamic = False
        if pol_dim > 0:
            self.pol_proj = nn.Linear(pol_dim, 16)
            has_dynamic = True
            
        if comp_dim > 0:
            self.comp_proj = nn.Linear(comp_dim, 16)
            has_dynamic = True

        if has_dynamic:
            static_dim += 16

        # Decoder input
        decoder_input_dim = (embedding_dim + static_dim) * 2 + self.price_emb_dim
        self.predictor = Decoder(decoder_input_dim, self.price_emb_dim)

    def encode_node_features(self, x_static_batch, x_dynamic=None, mode='pol'):
        # 1. Static Embeddings
        p = self.emb_party(x_static_batch[:, 0])
        s = self.emb_state(x_static_batch[:, 1])
        s_emb = torch.cat([p, s], dim=-1)
        
        # 2. Dynamic Features
        if self.pol_dim > 0 or self.comp_dim > 0:
            if mode == 'pol' and self.pol_dim > 0 and x_dynamic is not None:
                d_emb = self.pol_proj(x_dynamic)
                s_emb = torch.cat([s_emb, d_emb], dim=-1)
            elif mode == 'comp' and self.comp_dim > 0 and x_dynamic is not None:
                d_emb = self.comp_proj(x_dynamic)
                s_emb = torch.cat([s_emb, d_emb], dim=-1)
            else:
                pad = torch.zeros((s_emb.size(0), 16), device=s_emb.device)
                s_emb = torch.cat([s_emb, pad], dim=-1)

        return s_emb

    def forward(self, n_id, edge_index, edge_attr):
        memory = self.memory.memory[n_id]
        
        # FIX: Correctly handle empty edges by returning proper embedding dim
        if edge_index.numel() == 0:
            return torch.zeros((len(n_id), self.embedding_dim), device=memory.device)
            
        z = self.gnn(memory, edge_index, edge_attr)
        return z

    def get_price_embedding(self, price_seq):
        return self.price_encoder(price_seq)