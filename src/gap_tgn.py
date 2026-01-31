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
        edge_dim = msg_dim + time_enc.out_channels + 2
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

    def forward(self, z_src, z_dst, s_src, s_dst, p_context):
        h_src = torch.cat([z_src, s_src], dim=-1)
        h_dst = torch.cat([z_dst, s_dst], dim=-1)
        h_graph = torch.cat([h_src, h_dst], dim=-1)
        
        s_gate = self.gate(torch.cat([h_graph, p_context], dim=-1))
        
        h_graph_w = h_graph * s_gate
        h_mkt_w = p_context * (1 - s_gate)
        
        return self.net(torch.cat([h_graph_w, h_mkt_w], dim=-1))

class ResearchTGN(torch.nn.Module):
    def __init__(self, num_nodes, raw_msg_dim, memory_dim, time_dim, embedding_dim, num_parties, num_states):
        super().__init__()
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
        
        self.emb_party = nn.Embedding(num_parties + 1, 8) 
        self.emb_state = nn.Embedding(num_states + 1, 8)
        static_dim = 16
        
        self.predictor = Decoder(embedding_dim + static_dim, self.price_emb_dim)

    def encode_static(self, x_static_idx):
        p = self.emb_party(x_static_idx[:, 0])
        s = self.emb_state(x_static_idx[:, 1])
        return torch.cat([p, s], dim=-1)

    def forward(self, n_id, edge_index, edge_attr):
        memory = self.memory.memory[n_id]
        z = self.gnn(memory, edge_index, edge_attr)
        return z

    def get_price_embedding(self, price_seq):
        return self.price_encoder(price_seq)