import torch
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx
import pandas as pd
import numpy as np

# Try to set a non-interactive backend if possible, or just save to file
import matplotlib
matplotlib.use('Agg') 

class ChocolateVisualizer:
    def __init__(self, data_loader, graph_builder):
        self.loader = data_loader
        self.builder = graph_builder
        
    def visualize_subgraph(self, hetero_data, center_node_type, center_node_id, output_path="graph_viz.png", hop=1):
        """
        Visualizes a subgraph around a specific node.
        
        Args:
            hetero_data (HeteroData): The pyG graph object.
            center_node_type (str): 'politician' or 'company'.
            center_node_id (int or str): Index or BioGuideID/Ticker.
            output_path (str): Path to save the image.
        """
        print(f"Generating visualization for {center_node_type} {center_node_id}...")
        
        # Resolve ID if string
        node_idx = center_node_id
        if isinstance(center_node_id, str):
            if center_node_type == 'politician':
                node_idx = self.builder.pol_id_map.get(center_node_id)
            elif center_node_type == 'company':
                node_idx = self.builder.company_id_map.get(center_node_id)
                
        if node_idx is None:
            print(f"Node {center_node_id} not found in this graph snippet.")
            return

        # Convert to homogeneous for easier NetworkX visualization (or construct NX manually)
        # Since HeteroData to_networkx is tricky with attributes, let's build a small NX graph manually
        # by looking at edges connected to our node.
        
        G = nx.DiGraph()
        
        # Add center
        center_label = str(center_node_id)
        G.add_node(center_label, type=center_node_type, color='red' if center_node_type=='politician' else 'blue')
        
        # Edges
        edge_index = hetero_data['politician', 'trades', 'company'].edge_index
        src, dst = edge_index
        
        # Find connections
        # If Politician is center
        if center_node_type == 'politician':
            connected_indices = (src == node_idx).nonzero(as_tuple=True)[0]
            for idx in connected_indices:
                company_idx = int(dst[idx])
                # Reverse lookup Company Ticker
                # (Ideally we'd have a reverse map, iterating for now is okay for viz)
                ticker = [k for k, v in self.builder.company_id_map.items() if v == company_idx][0]
                
                # Edge Info
                attr = hetero_data['politician', 'trades', 'company'].edge_attr[idx]
                label = int(hetero_data['politician', 'trades', 'company'].y[idx])
                
                # Add Company Node
                G.add_node(ticker, type='company', color='blue')
                G.add_edge(center_label, ticker, weight=1, label=f"L:{label}")
                
        # If Company is center
        elif center_node_type == 'company':
            connected_indices = (dst == node_idx).nonzero(as_tuple=True)[0]
            for idx in connected_indices:
                pol_idx = int(src[idx])
                pid = [k for k, v in self.builder.pol_id_map.items() if v == pol_idx][0]
                
                attr = hetero_data['politician', 'trades', 'company'].edge_attr[idx]
                label = int(hetero_data['politician', 'trades', 'company'].y[idx])
                
                G.add_node(pid, type='politician', color='red')
                G.add_edge(pid, center_label, weight=1, label=f"L:{label}")

        # Draw
        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(G)
        
        # Colors
        colors = [nx.get_node_attributes(G, 'color').get(node, 'gray') for node in G.nodes()]
        
        nx.draw(G, pos, with_labels=True, node_color=colors, node_size=1500, font_size=10, font_weight='bold')
        
        # Edge Labels
        edge_labels = nx.get_edge_attributes(G, 'label')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
        
        plt.title(f"Transaction Neighborhood for {center_node_id}")
        plt.savefig(output_path)
        print(f"Saved visualization to {output_path}")
        plt.close()

    def print_ascii_subgraph(self, hetero_data, center_node_type, center_node_id):
        """
        Prints a text-based representation of the subgraph for quick inspection.
        """
        print(f"\n--- ASCII Visualization for {center_node_type} {center_node_id} ---")
        
        # Resolve ID
        node_idx = center_node_id
        if isinstance(center_node_id, str):
            if center_node_type == 'politician':
                node_idx = self.builder.pol_id_map.get(center_node_id)
            elif center_node_type == 'company':
                node_idx = self.builder.company_id_map.get(center_node_id)
                
        if node_idx is None:
            print(f"Node {center_node_id} not found.")
            return

        edge_index = hetero_data['politician', 'trades', 'company'].edge_index
        src, dst = edge_index
        edge_attr = hetero_data['politician', 'trades', 'company'].edge_attr
        y = hetero_data['politician', 'trades', 'company'].y
        
        if center_node_type == 'politician':
            print(f"[{center_node_type.upper()}: {center_node_id}]")
            connected_indices = (src == node_idx).nonzero(as_tuple=True)[0]
            
            if len(connected_indices) == 0:
                print("  (No transactions in this window)")
                return
                
            print("  |")
            for i, idx in enumerate(connected_indices):
                company_idx = int(dst[idx])
                ticker = [k for k, v in self.builder.company_id_map.items() if v == company_idx][0]
                
                # Decode Edge Features (approx)
                # attr: [amt, is_purchase, price]
                amt = float(edge_attr[idx][0])
                is_buy = float(edge_attr[idx][1]) > 0.5
                price = float(edge_attr[idx][2])
                label = int(y[idx])
                
                type_str = "BUY " if is_buy else "SELL"
                connector = "  +--"
                if i == len(connected_indices) - 1:
                    connector = "  +--" 
                
                print(f"{connector} [{type_str} ${amt:,.0f} @ {price:.2f}] --> [CO: {ticker}] (Label: {label})")

                print(f"{connector} [{type_str} ${amt:,.0f} @ {price:.2f}] --> [CO: {ticker}] (Label: {label})")

    def visualize_full_graph(self, hetero_data, output_path="full_graph.png"):
        """
        Visualizes the entire graph topology.
        Red = Politicians, Blue = Companies.
        """
        num_edges = hetero_data['politician', 'trades', 'company'].num_edges
        print(f"Generating full graph visualization with {num_edges} edges...")
        
        G = nx.DiGraph()
        
        # Reverse maps for labels
        rev_pol_map = {v: k for k, v in self.builder.pol_id_map.items()}
        rev_comp_map = {v: k for k, v in self.builder.company_id_map.items()}
        
        edge_index = hetero_data['politician', 'trades', 'company'].edge_index
        src, dst = edge_index
        y = hetero_data['politician', 'trades', 'company'].y
        
        # Add edges (and implicitly nodes)
        for i in range(num_edges):
            pol_idx = int(src[i])
            comp_idx = int(dst[i])
            # label = int(y[i])
            
            pol_id = rev_pol_map.get(pol_idx, f"P_{pol_idx}")
            comp_id = rev_comp_map.get(comp_idx, f"C_{comp_idx}")
            
            G.add_node(pol_id, color='red', type='politician')
            G.add_node(comp_id, color='blue', type='company')
            G.add_edge(pol_id, comp_id)
            
        plt.figure(figsize=(20, 20))
        # Use spring layout with k parameter to spread nodes out
        # k=None uses 1/sqrt(n), we might want slightly looser
        pos = nx.spring_layout(G, k=0.05, iterations=50) 
        
        colors = [nx.get_node_attributes(G, 'color').get(node, 'gray') for node in G.nodes()]
        sizes = [300 if nx.get_node_attributes(G, 'type').get(node) == 'politician' else 50 for node in G.nodes()]
        
        nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=sizes, alpha=0.7)
        nx.draw_networkx_edges(G, pos, alpha=0.2, arrowsize=5)
        
        # Only label politicians to avoid clutter
        labels = {n: n for n in G.nodes() if G.nodes[n].get('type') == 'politician'}
        nx.draw_networkx_labels(G, pos, labels, font_size=8, font_color='black')
        
        plt.title(f"Full Transaction Graph: {num_edges} Trades\n(Red=Politician, Blue=Company)")
        plt.axis('off')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved full graph to {output_path}")
        plt.close()

if __name__ == "__main__":
    pass
