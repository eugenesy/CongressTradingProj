import torch
import numpy as np
import sys
import os

# Setup path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '..'))

from clean_tgn.src.models_basic import BasicTGN
from torch_geometric.nn import TGNMemory

def verify_memory():
    print("="*50)
    print("MEMORY UPDATE VERIFICATION")
    print("="*50)
    
    # 1. Setup Dummy Model
    device = torch.device('cpu') # CPU is enough for this check
    num_nodes = 10
    msg_dim = 4
    memory_dim = 8 # Small for visualization
    
    model = BasicTGN(
        num_nodes=num_nodes,
        raw_msg_dim=msg_dim,
        memory_dim=memory_dim,
        time_dim=100,
        embedding_dim=8,
        num_parties=2,
        num_states=2,
        num_chambers=2
    ).to(device)
    
    # 2. Inspect Initial State of Node 0 (Pelosi)
    # The memory is initialized to zeros
    node_idx = 0
    mem_before = model.memory.memory[node_idx].clone().detach().numpy()
    
    print(f"\n[Step 1] Initial Memory for Node {node_idx}:")
    print(mem_before)
    
    # 3. Create a Transaction (Event)
    # Node 0 buys Node 1 at time t=100
    src = torch.tensor([0], dtype=torch.long)
    dst = torch.tensor([1], dtype=torch.long)
    t = torch.tensor([100], dtype=torch.long)
    # Msg: [Amt=5.0, Buy=1.0, Gap=3.0, WinRate=0.75]
    msg = torch.tensor([[5.0, 1.0, 3.0, 0.75]], dtype=torch.float)
    
    print(f"\n[Step 2] Event Occurs: Node {node_idx} buys Node 1")
    print(f"Message: {msg.numpy()}")
    
    # 4. update State
    print("\n[Step 3] Updating Memory...")
    model.memory.update_state(src, dst, t, msg)
    
    # 5. Inspect New State
    mem_after = model.memory.memory[node_idx].clone().detach().numpy()
    
    print(f"\n[Step 4] Updated Memory for Node {node_idx}:")
    print(mem_after)
    
    # 6. Verify Change
    diff = np.linalg.norm(mem_after - mem_before)
    print(f"\n[Conclusion] Memory Delta (L2 Norm): {diff:.6f}")
    
    if diff > 0.0001:
        print("✅ SUCCESS: The Node Memory definitely changed.")
        print("The 'Network Effect' is active.")
    else:
        print("❌ FAILURE: Memory did not change.")

if __name__ == "__main__":
    verify_memory()
