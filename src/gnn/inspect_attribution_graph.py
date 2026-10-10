import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

GRAPH_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "graph"
    / "attribution"
    / "pyg_attribution_graph.pt"
)

if not GRAPH_FILE.exists():
    print("ERROR: Attribution graph file not found.")
else:
    data = torch.load(
        GRAPH_FILE,
        map_location="cpu",
        weights_only=False
    )

    print("\nATTRIBUTION GRAPH INSPECTION")
    print("=" * 40)

    print("Object type:", type(data))

    if hasattr(data, "x"):
        print("Node feature shape:", tuple(data.x.shape))

    if hasattr(data, "edge_index"):
        print("Edge index shape:", tuple(data.edge_index.shape))

    if hasattr(data, "edge_type"):
        print("Edge type shape:", tuple(data.edge_type.shape))

    if hasattr(data, "num_nodes"):
        print("Number of nodes:", data.num_nodes)

    if hasattr(data, "num_edges"):
        print("Number of edges:", data.num_edges)

    print("\nAvailable attributes:")
    print(data)

    print("\nInspection completed.")