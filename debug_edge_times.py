
import torch
from torch_geometric.data import HeteroData

data = torch.load("output/progression/temporal_graph.pt", weights_only=False)

print("Checking edge times...")
problems = []
for et in data.edge_types:
    if 'edge_time' in data[et]:
        times = data[et].edge_time
        min_t = times.min().item()
        max_t = times.max().item()
        print(f"{et}: min={min_t}, max={max_t}")
        
        # Check for garbage (e.g. extremely negative due to long conversion of NaN? or 0?)
        if min_t < 1900 and min_t != 0: # Assuming years like 2000+
             print(f"⚠️ SUSPICIOUS TIME DETECTED in {et}")
             problems.append(et)
    else:
        print(f"{et}: No edge_time")

if problems:
    print("\n❌ Found problems with edge times. Modification needed.")
else:
    print("\n✅ Edge times look reasonable.")
