edge_list = "../JoVE_LLM/data/knowledge_graph/textmining_k2bio_knowledge_graph_filtered_k2_edges.csv"
node_list = "../JoVE_LLM/data/knowledge_graph/textmining_k2bio_knowledge_graph_filtered_k2_nodes.csv"

# Load triples

import pandas as pd
import torch
from torch_geometric.data import Data

# Load the triples
triples_df = pd.read_csv(edge_list, sep=',', header=0, names=['h', 'r', 't','w'])
triples_df = triples_df[['h','r','t']]

# Load the node to node type mapping
node_types_df = pd.read_csv(node_list, sep=',', header=0, names=['node_name', 'node_type'])

# Assuming node names are unique and can be used directly as node identifiers
# If not, you may need to map node names to unique integer IDs
# Convert to PyG data format

# Map node names to indices
unique_nodes = pd.concat([triples_df['h'], triples_df['t']]).unique()
node_to_index = {node: idx for idx, node in enumerate(unique_nodes)}
# Convert edges to tensor format
edge_index = torch.tensor([(node_to_index[h], node_to_index[t]) for h, t in zip(triples_df['h'], triples_df['t'])], dtype=torch.long).t().contiguous()
# Create a PyTorch Geometric Data object
data = Data(x=node_features, edge_index=edge_index)

disease_list = ['ARR','CCD','CHD','CM','IHD','OTH','VD','VOO']

node_types_df.loc[node_types_df['node_name'].isin(disease_list), 'node_type'] = "Disease"

node_types_df[node_types_df['node_name'].isin(disease_list)]
# TODO need to fix above bug, diseae list not showing up as the right node type


# Get unique node types and create a mapping
node_types = node_types_df['node_type'].unique()
type_to_index = {type_: idx for idx, type_ in enumerate(node_types)}

# One-hot encode the node types
def encode_node_type(node_type):
    encoding = [0] * len(node_types)
    encoding[type_to_index[node_type]] = 1
    return encoding

# Apply encoding to the DataFrame
node_types_df['type_vector'] = node_types_df['node_type'].apply(encode_node_type)

# Convert to PyTorch tensor
node_features = torch.tensor(node_types_df['type_vector'].tolist()).float()

print(node_features)

import pandas as pd
import torch
from torch_geometric.data import Data
from torch_sparse import SparseTensor

# Create a PyTorch Geometric Data object with sparse node features
data = Data(edge_index=edge_index)
data.num_nodes = node_features.shape[0]  # It's important to set the number of nodes explicitly
data.x = node_features  # This is our sparse node features tensor

print("Summary of the Knowledge Graph:")
print(f"Number of nodes: {data.num_nodes}")
print(f"Number of edges: {data.edge_index.size(1)}")  # edge_index.shape[1] gives the number of edges
print(f"Average node degree: {data.edge_index.size(1) / data.num_nodes:.2f}")
print(f"Number of node features: {data.num_node_features if data.x is not None else 0}")
print(f"Contains isolated nodes: {data.has_isolated_nodes()}")
print(f"Contains self-loops: {data.has_self_loops()}")
print(f"Is undirected: {data.is_undirected()}")

from torch_geometric.utils import to_undirected

# Make the graph undirected
edge_index_undirected = to_undirected(data.edge_index, num_nodes=data.num_nodes)

# Update the Data object
data.edge_index = edge_index_undirected

# Optionally, you might want to remove self-loops if they're not intended in your undirected graph
from torch_geometric.utils import remove_self_loops
data.edge_index, _ = remove_self_loops(data.edge_index)

# Verifying changes
print(f"Is undirected: {data.is_undirected()}")

print("Summary of the Knowledge Graph:")
print(f"Number of nodes: {data.num_nodes}")
print(f"Number of edges: {data.edge_index.size(1)}")  # edge_index.shape[1] gives the number of edges
print(f"Average node degree: {data.edge_index.size(1) / data.num_nodes:.2f}")
print(f"Number of node features: {data.num_node_features if data.x is not None else 0}")
print(f"Contains isolated nodes: {data.has_isolated_nodes()}")
print(f"Contains self-loops: {data.has_self_loops()}")
print(f"Is undirected: {data.is_undirected()}")

# Verify all edge weights are 1
if data.edge_attr is not None:
    all_weights_one = (data.edge_attr == 1.0).all().item()
    print(f"All edge weights are 1: {all_weights_one}")
    # Example: Inspect unique edge weights
    unique_edge_weights = data.edge_attr.unique()
    print("Unique edge weights:", unique_edge_weights)
else:
    print("No edge weights found.")

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

import torch
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.data import Data

# Assuming your data loading code is here

# Determine the computing device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
# source: https://github.com/pyg-team/pytorch_geometric/blob/master/examples/explain/gnn_explainer_link_pred.py

import os.path as osp

import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.explain import Explainer, GNNExplainer, ModelConfig
from torch_geometric.nn import GCNConv

transform = T.Compose([
    T.NormalizeFeatures(),
    T.ToDevice(device),
    T.RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=True),
])

train_data, val_data, test_data = transform(data)

edge_to_test = (53007,29805)

edge_label_index = torch.tensor(edge_to_test, dtype=torch.long)
edge_label_index=edge_label_index.to('cuda:0')
edge_label_index

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

    def decode(self, z, edge_label_index):
        src, dst = edge_label_index
        return (z[src] * z[dst]).sum(dim=-1)

    def forward(self, x, edge_index, edge_label_index):
        z = model.encode(x, edge_index)
        return model.decode(z, edge_label_index).view(-1)


model = GCN(10, 128, 64).to(device)
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)


def train():
    model.train()
    optimizer.zero_grad()

    out = model(train_data.x, train_data.edge_index,
                train_data.edge_label_index)
    loss = F.binary_cross_entropy_with_logits(out, train_data.edge_label)
    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def test(data):
    model.eval()
    out = model(data.x, data.edge_index, data.edge_label_index).sigmoid()
    return roc_auc_score(data.edge_label.cpu().numpy(), out.cpu().numpy())


for epoch in range(1, 201):
    loss = train()
    if epoch % 20 == 0:
        val_auc = test(val_data)
        test_auc = test(test_data)
        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Val: {val_auc:.4f}, '
              f'Test: {test_auc:.4f}')

model_config = ModelConfig(
    mode='binary_classification',
    task_level='edge',
    return_type='raw',
)

# Explain model output for a single edge:
edge_label_index = val_data.edge_label_index[:, 0]

explainer = Explainer(
    model=model,
    explanation_type='model',
    algorithm=GNNExplainer(epochs=200),
    node_mask_type='attributes',
    edge_mask_type='object',
    model_config=model_config,
)
explanation = explainer(
    x=train_data.x,
    edge_index=train_data.edge_index,
    edge_label_index=edge_label_index,
)
print(f'Generated model explanations in {explanation.available_explanations}')

# Explain a selected target (phenomenon) for a single edge:
edge_label_index = val_data.edge_label_index[:, 0]
target = val_data.edge_label[0].unsqueeze(dim=0).long()

explainer = Explainer(
    model=model,
    explanation_type='phenomenon',
    algorithm=GNNExplainer(epochs=200),
    node_mask_type='attributes',
    edge_mask_type='object',
    model_config=model_config,
)
explanation = explainer(
    x=train_data.x,
    edge_index=train_data.edge_index,
    target=target,
    edge_label_index=edge_label_index,
)
available_explanations = explanation.available_explanations
print(f'Generated phenomenon explanations in {available_explanations}')class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

    def decode(self, z, edge_label_index):
        src, dst = edge_label_index
        return (z[src] * z[dst]).sum(dim=-1)

    def forward(self, x, edge_index, edge_label_index):
        z = model.encode(x, edge_index)
        return model.decode(z, edge_label_index).view(-1)


model = GCN(10, 128, 64).to(device)
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)


def train():
    model.train()
    optimizer.zero_grad()

    out = model(train_data.x, train_data.edge_index,
                train_data.edge_label_index)
    loss = F.binary_cross_entropy_with_logits(out, train_data.edge_label)
    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def test(data):
    model.eval()
    out = model(data.x, data.edge_index, data.edge_label_index).sigmoid()
    return roc_auc_score(data.edge_label.cpu().numpy(), out.cpu().numpy())


for epoch in range(1, 201):
    loss = train()
    if epoch % 20 == 0:
        val_auc = test(val_data)
        test_auc = test(test_data)
        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Val: {val_auc:.4f}, '
              f'Test: {test_auc:.4f}')

model_config = ModelConfig(
    mode='binary_classification',
    task_level='edge',
    return_type='raw',
)

# Explain model output for a single edge:
edge_label_index = val_data.edge_label_index[:, 0]

explainer = Explainer(
    model=model,
    explanation_type='model',
    algorithm=GNNExplainer(epochs=200),
    node_mask_type='attributes',
    edge_mask_type='object',
    model_config=model_config,
)
explanation = explainer(
    x=train_data.x,
    edge_index=train_data.edge_index,
    edge_label_index=edge_label_index,
)
print(f'Generated model explanations in {explanation.available_explanations}')

# Explain a selected target (phenomenon) for a single edge:
edge_label_index = val_data.edge_label_index[:, 0]
target = val_data.edge_label[0].unsqueeze(dim=0).long()

explainer = Explainer(
    model=model,
    explanation_type='phenomenon',
    algorithm=GNNExplainer(epochs=200),
    node_mask_type='attributes',
    edge_mask_type='object',
    model_config=model_config,
)
explanation = explainer(
    x=train_data.x,
    edge_index=train_data.edge_index,
    target=target,
    edge_label_index=edge_label_index,
)
available_explanations = explanation.available_explanations
print(f'Generated phenomenon explanations in {available_explanations}')

# Inverting the node_to_index mapping to get index_to_node
index_to_node = {idx: node for node, idx in node_to_index.items()}


explainer = Explainer(
    model=model,
    explanation_type='model',
    algorithm=GNNExplainer(epochs=200),
    node_mask_type='attributes',
    edge_mask_type='object',
    model_config=model_config,
)
explanation = explainer(
    x=train_data.x,
    edge_index=train_data.edge_index,
    edge_label_index=edge_label_index,
)
print(f'Generated model explanations in {explanation.available_explanations}')

# Assuming 'explanation.edge_mask' contains the importance scores of each edge
edge_mask = explanation.edge_mask.detach().cpu().numpy()

# Select top n edges by importance for visualization
n = 10
indices = np.argsort(-edge_mask)[:n]
top_n_edges = edge_index[indices]
top_n_importances = edge_mask[indices]

# Create subgraph with only the top n important edges
H = G.edge_subgraph([(index_to_node[e[0]], index_to_node[e[1]]) for e in top_n_edges])

# Draw the subgraph
pos = nx.spring_layout(H)  # positions for all nodes

# Draw nodes
nx.draw_networkx_nodes(H, pos, node_size=700)

# Draw edges with width proportional to their importance
edge_widths = [5 * w / max(top_n_importances) for w in top_n_importances]  # Scale by 5 for visibility
nx.draw_networkx_edges(H, pos, edgelist=[(index_to_node[e[0]], index_to_node[e[1]]) for e in top_n_edges], width=edge_widths)

# Labels
nx.draw_networkx_labels(H, pos, font_size=20, font_family="sans-serif")

plt.title(f"Top {n} Important Edges Weighted by Importance")
plt.show()

# Assuming edge_index and index_to_node are defined as per your dataset and G is your original graph
edge_to_test = (53007, 29805)

edge_label_index = torch.tensor(edge_to_test, dtype=torch.long).to('cuda:0')

# Setup the explainer with appropriate configurations
explainer = Explainer(
    model=model,
    explanation_type='model',
    algorithm=GNNExplainer(epochs=200),
    node_mask_type='attributes',
    edge_mask_type='object',
    model_config=model_config,
)
explanation = explainer(
    x=train_data.x,
    edge_index=train_data.edge_index,
    edge_label_index=edge_label_index,
)

# Extract edge importance scores and ensure data is on CPU
edge_mask = explanation.edge_mask.detach().cpu().numpy()
edge_indices = train_data.edge_index.cpu().numpy()

# Select top n edges by importance
n = 10
indices = np.argsort(-edge_mask)[:n]
top_n_edges = edge_indices[:, indices]
top_n_importances = edge_mask[indices]

# Ensure edge_to_test is included in top_n_edges and check existence in index_to_node
if all(node in index_to_node for node in edge_to_test):
    if edge_to_test not in [(index_to_node[e[0]], index_to_node[e[1]]) for e in top_n_edges.T]:
        top_n_edges = np.hstack([top_n_edges, np.array(edge_to_test).reshape(2, 1)])
        top_n_importances = np.append(top_n_importances, 1)  # assuming importance as 1 for visibility

# Create subgraph including edge_to_test
H = G.edge_subgraph([(index_to_node[e[0]], index_to_node[e[1]]) for e in top_n_edges.T if e[0] in index_to_node and e[1] in index_to_node])

# Draw the subgraph
pos = nx.spring_layout(H)  # Node positions
nx.draw_networkx_nodes(H, pos, node_size=700)
edges = [(index_to_node[e[0]], index_to_node[e[1]]) for e in top_n_edges.T if e[0] in index_to_node and e[1] in index_to_node]
edge_colors = ['red' if e == (index_to_node[edge_to_test[0]], index_to_node[edge_to_test[1]]) else 'black' for e in edges]
edge_widths = [5 * w / max(top_n_importances) for w in top_n_importances]
nx.draw_networkx_edges(H, pos, edgelist=edges, width=edge_widths, edge_color=edge_colors)
nx.draw_networkx_labels(H, pos, font_size=20, font_family="sans-serif")

plt.title(f"Top {n} Important Edges Weighted by Importance")
plt.show()
