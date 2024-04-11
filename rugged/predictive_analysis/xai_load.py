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


from torch_geometric.loader import LinkNeighborLoader
import torch_geometric.transforms as T

def random_link_split(data, val_neg_sampling_ratio):
  """Splits the data into train, test, and val data."""
  transform = T.RandomLinkSplit(
      num_val=0.1,
      num_test=0.1,
      disjoint_train_ratio=0.3,
      neg_sampling_ratio=val_neg_sampling_ratio,
      # We don't want to add negative edges for the training set here because we
      # want them to vary for every epoch. Hence, we let the negative sampling
      # happen at the loader level for the training set. See below.
      add_negative_train_samples=False,
      edge_types=("user", "rates", "movie"),
      rev_edge_types=("movie", "rev_rates", "user"), 
  )
  train_data, val_data, test_data = transform(T.ToUndirected()(data))
  return train_data, val_data, test_data


def free_memory():
    """Clears the GPU cache and triggers garbage collection, to reduce OOMs."""
    cuda.empty_cache()
    gc.collect()

def get_masked(target, batch, split_name):
    """
    Applies the mask for a given split but no-ops if the mask isn't present.

    This is useful for shared models where the data may or may not be masked.
    """
    mask_name = f'{split_name}_mask'
    return target if mask_name not in batch else target[batch[mask_name]]

def evaluate_model(model, loader, frac=1.0):
    """
    Main model evaluation loop for validation/testing.

    This is almost identical to the equivalent function in the previous Colab,
    except we calculate the ROC AUC instead of the raw accuracy.
    """
    model.eval()

    y_true_tensors = []
    y_pred_tensors = []

    loader = split_loaders[split_name]
    num_batches = round(frac * len(loader))

    for i, batch in enumerate(loader):
        batch_num = i+1
        print(f'\r{split_name} batch {batch_num} / {num_batches}', end='')

        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch)

            # only evaluate the predictions from the split we care about
            relevant_pred = get_masked(pred, batch, split_name).detach().cpu()
            relevant_y = get_masked(
              batch["user", "rates", "movie"].edge_label.detach().cpu(),
              batch, split_name)

            y_pred_tensors.append(relevant_pred)
            y_true_tensors.append(relevant_y)

        if batch_num >= num_batches:
            break

        model.train()

        pred = torch.cat(y_pred_tensors, dim=0).numpy()
        true = torch.cat(y_true_tensors, dim=0).numpy()

    return roc_auc_score(true, pred)

from torch import cuda
import gc
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss
from sklearn.metrics import roc_auc_score
import tqdm
import copy
import time

import torch
from torch_geometric.loader import ClusterData, ClusterLoader
from torch_geometric.nn import GCNConv
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score
import numpy as np


from dataclasses import dataclass

def get_time():
    """Returns the current Unix (epoch) timestamp, in seconds."""
    return round(time.time())

@dataclass(frozen=True)
class EpochResult:
    # "index" of the epoch
    # (this is also discernable from the position in ModelResult.epoch_results)
    epoch_num: int

    # Unix timestamps (seconds) when the epoch started/finished training, but not
    # counting evaluation
    train_start_time: int
    train_end_time: int

    # mean train loss taken across all batches
    mean_train_loss: float

    # accuracy on the training/validation set at the end of this epoch
    train_acc: float
    valid_acc: float

@dataclass(frozen=True)
class ModelResult:
    # Unix timestamp for when the model started training
    start_time: int
    # Unix timestamp for when the model completely finished (including evaluation
    # on the test set)
    end_time: int

    # list of EpochResults -- see above
    epoch_results: list

    # model state for reloading
    state_dict: dict

    # final accuracy on the full test set (after all epochs)
    test_acc: float

    def get_total_train_time_sec(self):
        """
        Helper function for calculating the total amount of time spent training, not
        counting evaluation. In other words, this only counts the forward pass, the
        loss calculation, and backprop for each batch.
        """
        return sum([
          er.train_end_time - er.train_start_time
          for er in self.epoch_results])

    def get_total_train_time_min(self):
        """get_total_train_time_sec, converted to minutes. See above."""
        return self.get_total_train_time_sec() // 60

class GCN(torch.nn.Module):
    def __init__(self, num_nodes, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.embedding = torch.nn.Embedding(num_nodes, in_channels)  # Embedding layer
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def encode(self, x, edge_index, edge_weight=None):
        # x is now the output from the embedding layer
        x = self.conv1(x, edge_index, edge_weight).relu()
        x = self.conv2(x, edge_index, edge_weight)
        return x

    def decode(self, z, edge_label_index):
        # Calculate the probabilities for the edges in edge_label_index
        src, dst = edge_label_index
        return (z[src] * z[dst]).sum(dim=-1)

    def forward(self, batch_data):
        # Use n_id from batch_data to gather node embeddings
        x = self.embedding(batch_data.n_id)
        edge_index, edge_weight = batch_data.edge_index, batch_data.edge_attr
        z = self.encode(x, edge_index, edge_weight)  # Encode
        return self.decode(z, batch_data.edge_label_index)  # Decode


from sklearn.metrics import roc_auc_score
import torch

def evaluate_model(model, loader, split_name, frac=1.0):
    """
    Evaluates the model using the provided loader and calculates the ROC AUC score.

    Parameters:
    - model: The trained model to evaluate.
    - loader: DataLoader for the dataset split to evaluate on.
    - device: The device (CPU or GPU) to use for evaluation.
    - split_name: String indicating the dataset split ('valid' or 'test').
    - frac: Fraction of the dataset to use for evaluation.
    """
    model.eval()

    y_true_tensors = []
    y_pred_tensors = []

    num_batches = round(frac * len(loader))

    print()
    for i, batch in enumerate(loader):
        if i >= num_batches:
            break
        print(f'\r{split_name} evaluation batch {i+1} / {num_batches}', end='')

        batch = batch.to(device)

        with torch.no_grad():
            batch_pred = model(batch)
#             print()
#             print(batch)
#             print(batch.edge_label_index)
#             print(batch.input_id)
#             relevant_y = torch.ones(len(batch_pred))
            relevant_y = batch.edge_label.float().detach().cpu()

            y_pred_tensors.append(batch_pred.detach().cpu())
            y_true_tensors.append(relevant_y.detach().cpu())


#         if (i > 50):
#             break

    pred = torch.cat(y_pred_tensors, dim=0).numpy()
    true = torch.cat(y_true_tensors, dim=0).numpy()

    roc_auc = roc_auc_score(true, pred)

    print()  # Ensure the next print statement is on a new line
    model.train()  # Set the model back to training mode if further training will occur

    return roc_auc


my_model_file = "my_model.pt"
my_results = torch.load(my_model_file, map_location=torch.device('cpu'))
model = GCN(data.num_nodes, in_channels=128, hidden_channels=64, out_channels=1).to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
# model = TheModelClass(*args, **kwargs)
# model.load_state_dict(my_results.state_dict)

model.load_state_dict(my_results.state_dict)

model.eval()


import os.path as osp

import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.explain import Explainer, GNNExplainer, ModelConfig
from torch_geometric.nn import GCNConv

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
    algorithm=GNNExplainer(epochs=1000),
    node_mask_type='attributes',
    edge_mask_type='object',
    model_config=model_config,
)


from torch_geometric.data import Data

# Assuming train_data is a Data object containing all your training data
# edge_label_index is a tensor containing the index of the edge you want to explain

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
    algorithm=GNNExplainer(epochs=1000),
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

