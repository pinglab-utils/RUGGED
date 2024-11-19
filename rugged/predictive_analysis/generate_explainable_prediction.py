import sys
import os
import os.path as osp
import logging
import pandas as pd
import numpy as np
import networkx as nx
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import remove_self_loops, to_undirected
import torch_geometric.transforms as T
from torch_geometric.explain import Explainer, GNNExplainer, ModelConfig
from torch_geometric.nn import GCNConv
from sklearn.metrics import (
    precision_recall_curve, roc_auc_score, average_precision_score,
    accuracy_score, precision_score, recall_score, f1_score
)
import matplotlib.pyplot as plt
import itertools

class KnowledgeGraphModel:
    def __init__(self, edge_list, node_list, edges_to_predict):
        self.edge_list = edge_list
        self.node_list = node_list
        self.edges_to_predict = edges_to_predict
        self.data = None
        self.node_to_index = None
        self.index_to_node = None
        self.graph = None
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.device = self.initialize_device()

    @staticmethod
    def initialize_device():
        if torch.cuda.is_available():
            device = torch.device('cuda')
            logging.info("Using CUDA: NVIDIA GPU")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
            logging.info("Using MPS: Apple Silicon GPU")
        else:
            device = torch.device('cpu')
            logging.info("Using CPU")
        return device

    def encode_node_type(self, node_type, node_types, type_to_index):
        # One-hot encode the node types
        encoding = [0] * len(node_types)
        encoding[type_to_index[node_type]] = 1
        return encoding

    def load_kg_data(self):
        logging.info("Loading knowledge graph data.")
        
        # Load the triples (edges) and nodes
        triples_df = pd.read_csv(self.edge_list, sep=',', header=0, names=['h', 'r', 't', 'w'])
        triples_df = triples_df[['h', 'r', 't']]

        node_types_df = pd.read_csv(self.node_list, sep=',', header=0, names=['node_name', 'node_type'])

        # Get unique node types and create a mapping
        node_types = node_types_df['node_type'].unique()
        type_to_index = {type_: idx for idx, type_ in enumerate(node_types)}

        # Apply one-hot encoding to the node types
        node_types_df['type_vector'] = node_types_df['node_type'].apply(
            lambda nt: self.encode_node_type(nt, node_types, type_to_index)
        )

        # Convert node features to PyTorch tensor
        node_features = torch.tensor(node_types_df['type_vector'].tolist()).float()

        # Map node names to indices
        unique_nodes = pd.concat([triples_df['h'], triples_df['t']]).unique()
        node_to_index = {node: idx for idx, node in enumerate(unique_nodes)}
        index_to_node = {idx: node for node, idx in node_to_index.items()}
        self.index_to_node=index_to_node

        # Convert edges to tensor format
        edge_index = torch.tensor(
            [(node_to_index[h], node_to_index[t]) for h, t in zip(triples_df['h'], triples_df['t'])],
            dtype=torch.long
        ).t().contiguous()

        # Create a PyTorch Geometric Data object
        self.data = Data(x=node_features, edge_index=edge_index)
        self.data.num_nodes = node_features.shape[0]  # Set the number of nodes explicitly
        self.data.x = node_features  # Node features tensor
        
        # Make the graph undirected and remove self-loops
        self.data.edge_index = to_undirected(self.data.edge_index, num_nodes=self.data.num_nodes)
        self.data.edge_index, _ = remove_self_loops(self.data.edge_index)

        logging.info(f"Loaded {self.data.num_nodes} nodes and {self.data.edge_index.size(1)} edges.")

        # Create a NetworkX graph for visualizations
        edges = list(zip(self.data.edge_index[0].numpy(), self.data.edge_index[1].numpy()))
        edge_names = [(index_to_node[e1], index_to_node[e2]) for e1, e2 in edges]
        self.graph = nx.DiGraph()
        self.graph.add_edges_from(edge_names)
        
        # Filter out edges to predict that already exist in the knowledge graph
        existing_edges_set = set(edge_names)  # Convert existing edges to a set for fast lookup
        filtered_edges_to_predict = [
            (n1, n2) for n1, n2 in self.edges_to_predict
            if (n1, n2) not in existing_edges_set and (n2, n1) not in existing_edges_set  # Check both directions for undirected edges
        ]

        # Update self.edges_to_predict with the filtered list
        self.edges_to_predict = filtered_edges_to_predict
        # Convert the filtered edges to index-based format
        self.edges_to_predict_idx = [(node_to_index[n1], node_to_index[n2]) for n1, n2 in filtered_edges_to_predict]

        return self.data, node_to_index, index_to_node, self.graph, self.edges_to_predict_idx

    def prepare_data(self):
        logging.info("Preparing data for training.")
        transform = T.Compose([
            T.NormalizeFeatures(),
            T.ToDevice(self.device),
            T.RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=True),
        ])
        self.train_data, self.val_data, self.test_data = transform(self.data)

    def train_model(self, in_channels, hidden_channels, out_channels, n_epochs=1000, lr=0.01):
        logging.info("Training model.")
        model = GCN(in_channels, hidden_channels, out_channels).to(self.device)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)

        for epoch in range(1, 1 + n_epochs):
            loss = self.train_step(model, optimizer)
            if epoch % 100 == 0:
                eval_metrics = self.test(model, self.val_data)
                test_metrics = self.test(model, self.test_data)
                logging.info(f'Epoch: {epoch:02d}, Loss: {loss:.4f}')
                logging.info(f'Val Metrics: {eval_metrics}')
                logging.info(f'Test Metrics: {test_metrics}')
        logging.info("Model training complete.")
        
        #TODO return eval and save
        return model

    def train_step(self, model, optimizer):
        model.train()
        optimizer.zero_grad()
        out = model(self.train_data.x, self.train_data.edge_index, self.train_data.edge_label_index)
        loss = F.binary_cross_entropy_with_logits(out, self.train_data.edge_label)
        loss.backward()
        optimizer.step()
        return float(loss)

    @torch.no_grad()
    def test(self, model, data):
        model.eval()
        out = model(data.x, data.edge_index, data.edge_label_index).sigmoid()
        y_pred = (out.cpu().numpy() > 0.5).astype(int)
        y_true = data.edge_label.cpu().numpy()

        # Calculate metrics
        metrics = {
            "roc_auc": roc_auc_score(y_true, out.cpu().numpy()),
            "auprc": average_precision_score(y_true, out.cpu().numpy()),
            "accuracy": accuracy_score(y_true, y_pred),
            "f1_score": f1_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred)
        }
        return metrics
    
    @torch.no_grad()
    def predict_links(self, model, node_pairs, node_features):
        """Perform link prediction using the trained model."""
        logging.info("Predicting links.")
    
        with torch.no_grad():
            device = next(model.parameters()).device
            model.to(device)
            model.eval()

            # Prepare the data
            relevant_node_features, node_index_mapping = self.extract_relevant_node_features(node_features, node_pairs)
            edge_index = self.prepare_data_for_prediction(node_pairs, node_index_mapping).to(device)
        #     edge_labels = torch.ones(edge_index.size(1),2, dtype=torch.long).to(device) 
            edge_label_index = edge_index.clone().to(device)
            relevant_node_features = relevant_node_features.to(device)

            # Predict using the model
            outputs = model(relevant_node_features, edge_index, edge_label_index)

            # Convert tensor to numpy for further processing if needed
            probabilities = outputs.sigmoid().cpu().numpy()

            return probabilities, outputs

    @staticmethod
    def extract_relevant_node_features(node_features, node_pairs):
        # Extract unique node features needed for the given node pairs
        unique_nodes = sorted(list({node for pair in node_pairs for node in pair}))
        node_index_mapping = {node: idx for idx, node in enumerate(unique_nodes)}
        node_indices = torch.tensor(unique_nodes, dtype=torch.long)
        unique_node_features = node_features[node_indices]
        return unique_node_features, node_index_mapping

    @staticmethod
    def prepare_data_for_prediction(node_pairs, node_index_mapping):
        # Map the node indices in node_pairs to the new local indices
        remapped_pairs = [(node_index_mapping[n1], node_index_mapping[n2]) for n1, n2 in node_pairs]
        edge_index = torch.tensor(remapped_pairs, dtype=torch.long).t()
        return edge_index
    

    def save_predictions(self, edges, probabilities, node_index_mapping, output_file='predictions.tsv'):
        logging.info(f"Saving predictions to {output_file}.")
        with open(output_file, 'w') as outfile:
            outfile.write("edge\tprobability\n")
            for edge, prob in zip(edges, probabilities):
                edge_names = (node_index_mapping[edge[0]], node_index_mapping[edge[1]])
                outfile.write(f"{edge}\t{edge_names}\t{float(prob)}\n")


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
        z = self.encode(x, edge_index)
        return self.decode(z, edge_label_index).view(-1)

     
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
            algorithm=GNNExplainer(epochs=1000),
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
            e_name_pair = (self.index_to_node[e[0]],self.index_to_node[e[1]])
            importance_weights += [(e_name_pair, imp)]
        if print_weights:
            for e_name_pair, imp in importance_weights:
                print(e_name_pair,imp)
                logging.info(str(e_name_pair)+"\t"+str(imp))
        # save output pdf and csv
        if output_file:
            plt.savefig(output_file)

            importances_outfile = output_file[:-4]+".tsv"
            with open(importances_outfile,"w") as imp_outfile:
                imp_outfile.write("\n".join(["\t".join([str(e),str(imp)]) for e,imp in importance_weights]))

                
if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Insufficient arguments. Usage: python kg_pred.py <kg_edge_list> <kg_node_list> <output_folder>")
        sys.exit(1)

    kg_edge_list = sys.argv[1]
    kg_node_list = sys.argv[2]
    output_folder = sys.argv[3]
            
    # TODO For simplicity, define the disease and drug indices manually (this can be made dynamic based on input data)
    drug_candidates = ["DrugBank_Compound:DB00264", 
                       "DrugBank_Compound:DB00280", 
                       "DrugBank_Compound:DB00281", 
                       "DrugBank_Compound:DB00335", 
                       "DrugBank_Compound:DB00379", 
                       "DrugBank_Compound:DB00908", 
                       "DrugBank_Compound:DB01115", 
                       "DrugBank_Compound:DB01118", 
                       "DrugBank_Compound:DB01136", 
                       "DrugBank_Compound:DB01182", 
                       "DrugBank_Compound:DB01193"
                    ]
    disease_candidates = ['MeSH_Disease:D002311','MeSH_Disease:D019571']
    edges_to_predict = list(itertools.product(drug_candidates, disease_candidates))
    
    # Initialize logging
    logging.basicConfig(filename=os.path.join(output_folder,'output.log'), level=logging.INFO, 
                        format='%(asctime)s %(message)s')

    print("Loading Knowledge Graph")
    # Initialize device and data loading
    kg_model = KnowledgeGraphModel(kg_edge_list, kg_node_list, edges_to_predict)

    # Load and preprocess data
    kg_model.load_kg_data()

    # Prepare the data splits
    kg_model.prepare_data()

    print("Training GCN model")
    # Train and test the model
    # TODO make in_channels as a dynamic variable
    trained_model = kg_model.train_model(in_channels=6, hidden_channels=128, out_channels=64)

    # Save predictions
    print("Predicting edges")
    probabilities, outputs = kg_model.predict_links(trained_model, kg_model.edges_to_predict_idx, kg_model.data.x)
    ranked_outputs = get_ranked_output(kg_model.edges_to_predict_idx, probabilities)
    pred_output = os.path.join(output_folder, "predictions.tsv")
    print(probabilities)
    print(outputs)
    kg_model.save_predictions(kg_model.edges_to_predict_idx, probabilities, kg_model.index_to_node, output_file=pred_output)
    
    # Explain predictions
    model_config = ModelConfig(
        mode='binary_classification',
        task_level='edge',
        return_type='raw',
    )
    graph_model = ExplainableGraphModel(trained_model, kg_model.train_data, kg_model.val_data, kg_model.index_to_node, kg_model.graph, model_config)
    
    n = 5 # Examine the top n predictions
    logging.info(f"Explaining top {n} predictions")
    count = 0
    explanation_output_folder = os.path.join(output_folder,"explanations")
    if not os.path.exists(explanation_output_folder):
        os.makedirs(explanation_output_folder)
    for edge_to_explain, _ in ranked_outputs:
        e1, e2 = (kg_model.index_to_node[edge_to_explain[0]],kg_model.index_to_node[edge_to_explain[1]])
        edge_name = f"{e1}_{e2}"
        xai_figure_output = os.path.join(explanation_output_folder,f"{edge_name}_edge_importance.pdf")
        print(f"Edge: {edge_to_explain} {edge_name}")
        logging.info(f"Edge: {edge_to_explain} {edge_name}")
        importances = graph_model.visualize_explainable_prediction(edge_to_explain, output_file=xai_figure_output)
        print(f"Saved to file {xai_figure_output}")
        logging.info(f"Saved to file {xai_figure_output}")

        count += 1
        if count == n:
            break

