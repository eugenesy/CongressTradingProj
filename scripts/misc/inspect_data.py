
import torch
from torch_geometric.data import TemporalData
import datetime

try:
    data = torch.load("data/temporal_data.pt", weights_only=False)
    print("Keys:", data)
    print("Msg shape:", data.msg.shape)
    print("Time shape:", data.t.shape)
    print("Static shape:", getattr(data, 'x_static', 'Not Found'))
    
    # Check time range
    t_min = data.t.min().item()
    t_max = data.t.max().item()
    print(f"Time range: {t_min} to {t_max}")
    
    # Guess units (seconds? days?)
    # 2012 to 2024 is approx 12 years.
    # If seconds: 12 * 365 * 24 * 3600 = ~378,000,000
    # If days: ~4380
    
    # Assuming standard unix timestamp if large, or days if small.
    # Let's try to convert t_min to date assuming it's a timestamp.
    try:
        d_min = datetime.datetime.fromtimestamp(t_min)
        d_max = datetime.datetime.fromtimestamp(t_max)
        print(f"Date range (if timestamp): {d_min} to {d_max}")
    except:
        print("Could not convert to date from timestamp")

except Exception as e:
    print(e)
