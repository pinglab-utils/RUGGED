import pandas as pd
import torch
from torch_geometric.data import Data
from torch_sparse import SparseTensor

import os.path as osp

import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.explain import Explainer, GNNExplainer, ModelConfig
from torch_geometric.nn import GCNConv

if torch.cuda.is_available():
    device = torch.device('cuda')
    print("Using CUDA: NVIDIA GPU")
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
    print("Using MPS: Apple Silicon GPU")
else:
    device = torch.device('cpu')
    print("Using CPU")

edge_list = "grape_pred_merged_edge_list.tsv"
node_list = "grape_pred_merged_node_list.tsv"

# Load the triples
triples_df = pd.read_csv(edge_list, sep='\t', header=0, names=['h', 'r', 't', 'w'])
triples_df = triples_df[['h', 'r', 't']]

# Load the node to node type mapping
node_types_df = pd.read_csv(node_list, sep='\t', header=0, names=['node_name', 'node_type'])

unique_nodes = pd.concat([triples_df['h'], triples_df['t']]).unique()
node_to_index = {node: idx for idx, node in enumerate(unique_nodes)}

edge_index = torch.tensor([(node_to_index[h], node_to_index[t]) for h, t in zip(triples_df['h'], triples_df['t'])], dtype=torch.long).t().contiguous()

n = len(unique_nodes)
indices = torch.arange(0, n).unsqueeze(0).repeat(2, 1)
values = torch.ones(n)
node_features = SparseTensor(row=indices[0], col=indices[1], value=values, sparse_sizes=(n, n))


data = Data(edge_index=edge_index)
data.num_nodes = n
data.x = node_features

print("Summary of the Knowledge Graph:")
print(f"Number of nodes: {data.num_nodes}")
print(f"Number of edges: {data.edge_index.size(1)}")  # edge_index.shape[1] gives the number of edges
print(f"Average node degree: {data.edge_index.size(1) / data.num_nodes:.2f}")
print(f"Number of node features: {data.num_node_features if data.x is not None else 0}")
print(f"Contains isolated nodes: {data.has_isolated_nodes()}")
print(f"Contains self-loops: {data.has_self_loops()}")
print(f"Is undirected: {data.is_undirected()}")

# Trying to load in smaller chunks
from torch_geometric.loader import DataLoader, ClusterData, ClusterLoader
import pdb; pdb.set_trace()
# Step 1: Create ClusterData
cluster_data = ClusterData(data, num_parts=1000, recursive=False, save_dir='./cluster_data')

# Step 2: Create ClusterLoader to load mini-batches
cluster_loader = ClusterLoader(cluster_data, batch_size=20, shuffle=True, num_workers=4)

# Now, `cluster_loader` can be used similarly to a standard DataLoader in PyTorch



rint("DONE!!")
