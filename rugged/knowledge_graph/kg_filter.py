import os
import sys
import pandas as pd
import networkx as nx

def parse_nodes(node_str):
    """Parses a comma-separated list of nodes."""
    nodes = node_str.split(',')
    return [node.strip() for node in nodes]  # Strip whitespace


def load_knowledge_graph(edge_file, sep="\t", header=None):
    kg_df = pd.read_csv(edge_file, sep=sep, header=header)
    kg_df.columns = ['h', 'r', 't'] # Assumed to be hrt format
    # Ensure 'weight' column exists for compatibility; set to 1 if absent
    if 'weight' not in kg_df.columns:
        kg_df['weight'] = 1.0
    
    return kg_df


def check_nodes_in_graph(G, nodes):
    """Checks if nodes exist in the knowledge graph."""
    missing_nodes = [node for node in nodes if node not in G]
    if missing_nodes:
        print(f"Warning: The following nodes are not in the knowledge graph: {missing_nodes}")
    return all(node in G for node in nodes)


def filter_knowledge_graph(kg_df, start_nodes, k=2):
    ''' This function extracts the edges between nodes which are reachable k-hop from any nodes in start_nodes'''
    
    # Create a directed graph from the DataFrame including edge attributes
    G = nx.from_pandas_edgelist(kg_df, 'h', 't', edge_attr=True, create_using=nx.Graph())
    
    # Check the starting nodes are in the graph
    check_nodes_in_graph(G, start_nodes)
    
    # Find nodes that are within k hops from any of the start nodes
    nodes_within_k_hops = set()
    for start_node in start_nodes:
        for edge in nx.bfs_edges(G, source=start_node, depth_limit=k):
            nodes_within_k_hops.update(edge)

    # Filter the original DataFrame to include only edges between nodes in nodes_within_k_hops
    filtered_edges = kg_df[(kg_df['h'].isin(nodes_within_k_hops)) & (kg_df['t'].isin(nodes_within_k_hops))]
    
    return filtered_edges


def node_type_mapping(df):
    # Extract all unique nodes from the DataFrame
    nodes = set(df['h']).union(set(df['t']))

    # Create a DataFrame from nodes with their respective types extracted
    node_df = pd.DataFrame({
        'node_name': list(nodes),
        'node_type': [node.split(':')[0] if ':' in node else "Disease" for node in nodes]
    })

    return node_df


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python kg_filter.py <nodes> <knowledge_graph> <output_folder>")
        print("Example: python kg_filter.py MeSH_Disease:D019571,MeSH_Disease:D002311 ./whole_kg.txt ./data")
        sys.exit(1)
        
    k=2
    starting_node_text = sys.argv[1]
    kg_file = sys.argv[2]
    output_folder = sys.argv[3]
        
    # Parse starting nodes
    starting_nodes = parse_nodes(starting_node_text)
    print(f"{len(starting_nodes)} starting nodes: {starting_nodes}")
    
    # Parse knowledge graph
    kg_df = load_knowledge_graph(kg_file)
    
    # Filter knowledge graph
    filtered_kg_df = filter_knowledge_graph(kg_df, starting_nodes, k=k)

    # Prepare node mapping df
    nodes_df = node_type_mapping(filtered_kg_df)
    
    # Output files
    filtered_out_edges_file = os.path.join(output_folder,f"./whole_kg_filtered_k{k}_edges.csv")
    filtered_out_nodes_file = os.path.join(output_folder,f"./whole_kg_filtered_k{k}_nodes.csv")
    filtered_kg_df.to_csv(filtered_out_edges_file, index=False)
    nodes_df.to_csv(filtered_out_nodes_file, index=False)
    
    print(f"{filtered_kg_df.shape[0]} lines written to {filtered_out_edges_file}")
    print(f"{nodes_df.shape[0]} lines written to {filtered_out_nodes_file}")
    print("Finished.")

