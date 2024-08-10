import sys
import os
import os.path as osp
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_sparse import SparseTensor
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import remove_self_loops
from torch_geometric.utils import to_undirected
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.explain import Explainer, GNNExplainer, ModelConfig
from torch_geometric.nn import GCNConv

from sklearn.metrics import precision_recall_curve, roc_curve, auc, roc_auc_score, average_precision_score, accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score

def initialize_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("Using CUDA: NVIDIA GPU")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using MPS: Apple Silicon GPU")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device


def encode_node_type(node_type, node_types, type_to_index):
    # One-hot encode the node types
    encoding = [0] * len(node_types)
    encoding[type_to_index[node_type]] = 1
    return encoding


def load_kg_data(edge_list, node_list):

    # Load the triples
    triples_df = pd.read_csv(edge_list, sep=',', header=0, names=['h', 'r', 't','w'])
    triples_df = triples_df[['h','r','t']]
    
    # Load the node to node type mapping
    node_types_df = pd.read_csv(node_list, sep=',', header=0, names=['node_name', 'node_type'])
    
    # Get unique node types and create a mapping
    node_types = node_types_df['node_type'].unique()
    type_to_index = {type_: idx for idx, type_ in enumerate(node_types)}
    
    # Apply encoding to the DataFrame
    node_types_df['type_vector'] = node_types_df['node_type'].apply(lambda nt: encode_node_type(nt, node_types, type_to_index))
    # Convert to PyTorch tensor
    node_features = torch.tensor(node_types_df['type_vector'].tolist()).float()
    
    # Map node names to indices
    unique_nodes = pd.concat([triples_df['h'], triples_df['t']]).unique()
    node_to_index = {node: idx for idx, node in enumerate(unique_nodes)}
    # Inverting the node_to_index mapping to get index_to_node
    index_to_node = {idx: node for node, idx in node_to_index.items()}

    # Convert edges to tensor format
    edge_index = torch.tensor([(node_to_index[h], node_to_index[t]) for h, t in zip(triples_df['h'], triples_df['t'])], dtype=torch.long).t().contiguous()
    # Create a PyTorch Geometric Data object
    data = Data(x=node_features, edge_index=edge_index)
    
    disease_list = ['ARR','CCD','CHD','CM','IHD','OTH','VD','VOO']
    node_types_df.loc[node_types_df['node_name'].isin(disease_list), 'node_type'] = "Disease"
    node_types_df[node_types_df['node_name'].isin(disease_list)]
    # TODO need to fix above bug, diseae list not showing up as the right node type
    disease_list = ['MeSH_Disease:D002313','MeSH_Disease:D002311']
    disease_indices = [node_to_index[n] for n in disease_list]

    # Create a PyTorch Geometric Data object with sparse node features
    data = Data(edge_index=edge_index)
    data.num_nodes = node_features.shape[0]  # set the number of nodes explicitly
    data.x = node_features  # sparse node features tensor
    
    # Make the graph undirected
    edge_index_undirected = to_undirected(data.edge_index, num_nodes=data.num_nodes)
    
    # Update the Data object
    data.edge_index = edge_index_undirected
    data.edge_index, _ = remove_self_loops(data.edge_index)
    
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

    # Convert the edge_index tensor to a list of edges
    edges = list(zip(edge_index[0].numpy(), edge_index[1].numpy()))
    edge_names = [(index_to_node[e1],index_to_node[e2]) for e1,e2 in edges]
    
    # Create a directed graph
    G = nx.DiGraph()
    G.add_edges_from(edge_names)

    # Get possible disease-gene predictions
    kg_edges = [(int(e1),int(e2)) for e1, e2 in data.edge_index.T]
    
    all_drug_disease_pairs = [
        ('DrugBank_Compound:DB00335','MeSH_Disease:D002313'),
        ('DrugBank_Compound:DB00571','MeSH_Disease:D002311'),
        ('DrugBank_Compound:DB00280','MeSH_Disease:D002311')
    ]
    all_drug_disease_pairs = [(node_to_index[n1],node_to_index[n2]) for n1,n2 in all_drug_disease_pairs]
    
    #all_disease_gene_pairs = prepare_gene_disease_predictions(disease_indices, kg_edges, index_to_node)
   
    # NEW TODO
#    target_proteins = ['UniProt:P35348', 'UniProt:Q12809', 'UniProt:O75469', 'UniProt:P03372', 'UniProt:P04278', 'UniProt:Q92731', 'UniProt:P28222', 'UniProt:P08908', 'UniProt:P08913', 'UniProt:P20309', 'UniProt:P08172', 'UniProt:P11229', 'UniProt:P48637', 'UniProt:Q03154', 'UniProt:P18089', 'UniProt:P18825']
#    target_protein_ids = [node_to_index[n] for n in target_proteins if n in node_to_index]
#    print(len(target_protein_ids))
#    all_disease_gene_pairs = get_prot_disease_rels(target_protein_ids,disease_indices,kg_edges,index_to_node)

    #return data, node_to_index, index_to_node, G, all_disease_gene_pairs
    return data, node_to_index, index_to_node, G, all_drug_disease_pairs


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
    # Convert probabilities to binary predictions
    y_pred = (out.cpu().numpy() > 0.5).astype(int)
    y_true = data.edge_label.cpu().numpy()

    # Calculate metrics
    roc_auc = roc_auc_score(y_true, out.cpu().numpy())
    auprc = average_precision_score(y_true, out.cpu().numpy())
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred)

    return {
        "roc_auc": roc_auc,
        "auprc": auprc,
        "accuracy": accuracy,
        "f1_score": f1,
        "precision": precision,
        "recall": recall
    }


@torch.no_grad()
def identify_optimal_threshold(data):
    model.eval()
    out = model(data.x, data.edge_index, data.edge_label_index).sigmoid()
    y_scores = out.cpu().numpy()
    y_true = data.edge_label.cpu().numpy()

    # Calculate precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    
    # Calculate F1 score for each threshold
    F1_scores = 2 * (precision * recall) / (precision + recall)
    optimal_idx = np.nanargmax(F1_scores)  # handle potential NaNs in F1_scores
    optimal_threshold = thresholds[optimal_idx]
    optimal_f1 = F1_scores[optimal_idx]

    plt.clf()
    plt.figure()
    plt.plot(thresholds, precision[:-1], 'b--', label='Precision')
    plt.plot(thresholds, recall[:-1], 'g-', label='Recall')
    plt.plot(thresholds, F1_scores[:-1], 'r-', label='F1 Score')
    plt.xlabel('Threshold')
    plt.legend(loc='upper left')
    plt.ylim([0,1])
    plt.show()

    print(f"Optimal prediction threshold {optimal_threshold} which achieved f1 {optimal_f1}")
    return optimal_threshold, optimal_f1


#def prepare_gene_disease_predictions(disease_nodes, kg_edges, index_to_node):
#    ''' This function identifies viable protein-disease predictions edges
#        which do not already exist in the graph
#    '''
#    
#    # Get the node index for all genes using a set comprehension for fast lookup
#    genes = {i for i, n in index_to_node.items() if 'Entrez' in n}
#    
#    # Create a set of all disease-gene pairs (using a set directly to optimize removal operations later)
#    all_disease_gene_pairs = {(gene, disease) for disease in disease_nodes for gene in genes}
#    
#    # Convert kg_edges to a set for fast operation and include both (n1, n2) and (n2, n1)
#    kg_edge_set = set(kg_edges) | {(n2, n1) for n1, n2 in kg_edges}
#    
#    print(f"{len(all_disease_gene_pairs)} possible disease-gene pairs")
#    
#    # Use set difference to remove existing edges from all pairs
#    all_disease_gene_pairs.difference_update(kg_edge_set)
#    
#    print(f"{len(all_disease_gene_pairs)} disease-gene pairs remaining after filtering kg")
#    
#    return all_disease_gene_pairs
#
#def get_prot_disease_rels(target_protein_ids,disease_nodes,kg_edges,index_to_node):
#
#    # Create a set of all disease-gene pairs (using a set directly to optimize removal operations later)
#    all_disease_gene_pairs = {(protein, disease) for disease in disease_nodes for protein in target_protein_ids}
#    
#    # Convert kg_edges to a set for fast operation and include both (n1, n2) and (n2, n1)
#    kg_edge_set = set(kg_edges) | {(n2, n1) for n1, n2 in kg_edges}
#    
#    print(f"{len(all_disease_gene_pairs)} possible disease-protein pairs")
#    
#    # Use set difference to remove existing edges from all pairs
#    all_disease_gene_pairs.difference_update(kg_edge_set)
#    
#    print(f"{len(all_disease_gene_pairs)} disease-protein pairs remaining after filtering kg")
#    
#    return all_disease_gene_pairs


def extract_relevant_node_features(node_features, node_pairs):
    # Get unique nodes
    unique_nodes = sorted(list({node for pair in node_pairs for node in pair}))

    # Create a mapping from old indices (global) to new indices (local to unique_node_features)
    node_index_mapping = {node: idx for idx, node in enumerate(unique_nodes)}

    # Convert unique_nodes list to a PyTorch tensor
    node_indices = torch.tensor(unique_nodes, dtype=torch.long)

    # Index node_features to get only the features of unique nodes
    unique_node_features = node_features[node_indices]

    return unique_node_features, node_index_mapping


def prepare_data_for_prediction(node_pairs, node_index_mapping):
    # Remap node indices in node_pairs to the new local indices
    remapped_pairs = [(node_index_mapping[n1], node_index_mapping[n2]) for n1, n2 in node_pairs]
    edge_index = torch.tensor(remapped_pairs, dtype=torch.long).t()
    
    return edge_index

@torch.no_grad()
def predict_links(model, node_pairs, node_features):
    
    with torch.no_grad():
        device = next(model.parameters()).device
        model.to(device)
        model.eval()
        
        # Prepare the data
        relevant_node_features, node_index_mapping = extract_relevant_node_features(node_features, node_pairs)
        edge_index = prepare_data_for_prediction(node_pairs, node_index_mapping).to(device)
    #     edge_labels = torch.ones(edge_index.size(1),2, dtype=torch.long).to(device) 
        edge_label_index = edge_index.clone().to(device)
        relevant_node_features = relevant_node_features.to(device)

        # Predict using the model
        outputs = model(relevant_node_features, edge_index, edge_label_index)

        # Convert tensor to numpy for further processing if needed
        probabilities = outputs.sigmoid().cpu().numpy()

        return probabilities, outputs


def output_predictions(edges, probabilities, output_file = 'predictions.tsv'):

    with open(output_file,'w') as outfile:
        outfile.write("edge\tprobability\n")
        out = [str(e)+"\t"+str(float(p)) for e,p in zip(edges,probabilities)]
        outfile.write("\n".join(out))


def get_ranked_output(edges, probs):
    # Combine edges and probabilities into a list of tuples
    edge_probability_pairs = list(zip(edges, probs))

    # Sort the list of tuples by probability in descending order
    edge_probability_pairs.sort(key=lambda x: x[1], reverse=True)

    return edge_probability_pairs


class ExplainableGraphModel:
    def __init__(self, model, train_data, val_data, index_to_node, G, model_config):
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.index_to_node = index_to_node
        self.G = G
        self.model_config = model_config
        self.explainer = Explainer(
            model=self.model,
            explanation_type='model',
            algorithm=GNNExplainer(epochs=200),
            node_mask_type='attributes',
            edge_mask_type='object',
            model_config=self.model_config,
        )

    def visualize_explainable_prediction(self, edge_to_test, n=10, print_weights=True, output_file=None):
        # Explain model output for the single edge:
        edge_label_index = torch.tensor(edge_to_test, dtype=torch.long).to(self.train_data.edge_index.device)
        
        # Explain the model's predictions
        explanation = self.explainer(
            x=self.train_data.x,
            edge_index=self.train_data.edge_index,
            edge_label_index=edge_label_index,
        )

        # Extract edge importance scores and ensure data is on CPU
        edge_mask = explanation.edge_mask.detach().cpu().numpy()
        edge_indices = self.train_data.edge_index.cpu().numpy()

        # Select top n edges by importance for visualization
        indices = np.argsort(-edge_mask)[:n]
        top_n_edges = edge_indices[:, indices]
        top_n_importances = edge_mask[indices]

        # Include edge_to_test if it's not already among the top n edges
        edge_to_test_converted = (self.index_to_node[int(edge_to_test[0])], self.index_to_node[int(edge_to_test[1])])
        if edge_to_test_converted not in [(self.index_to_node[int(e[0])], self.index_to_node[int(e[1])]) for e in top_n_edges.T]:
            top_n_edges = np.hstack([top_n_edges, np.array(edge_to_test).reshape(2, 1)])
            top_n_importances = np.append(top_n_importances, 1)  # assuming importance as 1 for visibility

        # Create subgraph including edge_to_test
        H = self.G.edge_subgraph([(self.index_to_node[int(e[0])], self.index_to_node[int(e[1])]) for e in top_n_edges.T if int(e[0]) in self.index_to_node and int(e[1]) in self.index_to_node])

        
        # Draw the subgraph
        plt.clf()
        plt.figure(figsize=(10,8))
#         pos = nx.spring_layout(H, seed=42)  # Node positions
        pos = nx.circular_layout(H)  # Node positions
        nx.draw_networkx_nodes(H, pos, node_size=200)
        edges = [(self.index_to_node[int(e[0])], self.index_to_node[int(e[1])]) for e in top_n_edges.T if int(e[0]) in self.index_to_node and int(e[1]) in self.index_to_node]
        edge_colors = ['red' if e == edge_to_test_converted else 'black' for e in edges]
        edge_widths = [5 * w / max(top_n_importances) for w in top_n_importances]
        nx.draw_networkx_edges(H, pos, edgelist=edges, width=edge_widths, edge_color=edge_colors)
        nx.draw_networkx_labels(H, pos, font_size=10, font_family="sans-serif")

        plt.title(f"Top {n} Important Edges and Specific Test Edge Highlighted")
        plt.show()

        # print weights
        importance_weights = []
        for e, imp in zip(top_n_edges.T,top_n_importances):
            e_name_pair = (index_to_node[e[0]],index_to_node[e[1]])
            importance_weights += [(e_name_pair, imp)]
        if print_weights:
            for e_name_pair, imp in importance_weights:
                print(e_name_pair,imp)
        # save output pdf and csv
        if output_file:
            plt.savefig(output_file)

            importances_outfile = output_file[:-4]+".tsv"
            with open(importances_outfile,"w") as imp_outfile:
                imp_outfile.write("\n".join(["\t".join([str(e),str(imp)]) for e,imp in importance_weights]))


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python kg_pred.py <kg_edge_list> <kg_node_list> <output_folder>")
        sys.exit(1)

    kg_edge_list = sys.argv[1]
    kg_node_list = sys.argv[2]
    output_folder = sys.argv[3]

    device = initialize_device()

    # Parse input data
    data, node_to_index, index_to_node, G, edges_to_predict = load_kg_data(kg_edge_list, kg_node_list)

    # Split the dataset
    transform = T.Compose([
        T.NormalizeFeatures(),
        T.ToDevice(device),
        T.RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=True),
    ])
    
    train_data, val_data, test_data = transform(data)
   
    # Train prediction model
    model_config = ModelConfig(
        mode='binary_classification',
        task_level='edge',
        return_type='raw',
    )

    model = GCN(14, 128, 64).to(device)
    #model = GCN(12, 128, 64).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)

    n_epochs = 1000
    for epoch in range(1, 1 + n_epochs):
        loss = train()
        if epoch % 100 == 0:
            eval_metrics = test(val_data)
            test_metrics = test(test_data)
            print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}')
            print(f'Val Accuracy: {eval_metrics["accuracy"]:.4f}, Test Accuracy: {test_metrics["accuracy"]:.4f}')
            print(f'Val Precision: {eval_metrics["precision"]:.4f}, Test Precision: {test_metrics["precision"]:.4f}')
            print(f'Val Recall: {eval_metrics["recall"]:.4f}, Test Recall: {test_metrics["recall"]:.4f}')
            print(f'Val F1 score: {eval_metrics["f1_score"]:.4f}, Test F1 score: {test_metrics["f1_score"]:.4f}')
            print(f'Val ROC AUC: {eval_metrics["roc_auc"]:.4f}, Test ROC AUC: {test_metrics["roc_auc"]:.4f}')
            print(f'Val AUPRC: {eval_metrics["auprc"]:.4f}, Test AUPRC: {test_metrics["auprc"]:.4f}')
    print("Model training successful.")
    # TODO save the model

    # Make predictions
    optimal_threshold, _ = identify_optimal_threshold(val_data)
    probabilities, outputs = predict_links(model, edges_to_predict, data.x)
    ranked_outputs = get_ranked_output(edges_to_predict,probabilities)
    pred_output = os.path.join(output_folder,"predictions.tsv")
    output_predictions(edges_to_predict,probabilities, output_file = pred_output)

    # Explain predictions
    graph_model = ExplainableGraphModel(model, train_data, val_data, index_to_node, G, model_config)
    
    n = 5 # Examine the top n predictions
    count = 0
    explanation_output_folder = os.path.join(output_folder,"explanations")
    if not os.path.exists(explanation_output_folder):
        os.makedirs(explanation_output_folder)
    for edge_to_explain, _ in ranked_outputs:
        e1, e2 = (index_to_node[edge_to_explain[0]],index_to_node[edge_to_explain[1]])
        edge_name = f"{e1}_{e2}"
        xai_figure_output = os.path.join(explanation_output_folder,f"{edge_name}_edge_importance.pdf")
        print(f"Edge: {edge_to_explain} {edge_name}")
        importances = graph_model.visualize_explainable_prediction(edge_to_explain, output_file=xai_figure_output)
        print(f"Saved to file {xai_figure_output}")

        count += 1
        if count == n:
            break

        
