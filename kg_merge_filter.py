import sys
import pandas as pd
import networkx as nx

def normalize_protein_ids(node_id):
    if ":" not in node_id and "R-" not in node_id:
        # R-HSA-XXX is Reactome. All other node types have namespace identifier.
        protein_id = "UniProt:"+node_id.replace("_HUMAN", "")
        return protein_id
    else: 
        return node_id
    
    
def load_know2bio(know2bio_edge_file):
    k2bio_df = pd.read_csv(know2bio_edge_file, sep="\t", header=None)
    k2bio_df.columns = ['h', 'r', 't']
    k2bio_df['h'] = k2bio_df['h']
    k2bio_df['t'] = k2bio_df['t']
    k2bio_df['weight'] = 1.0  # Assign edge weight of 1.0 for all edges
    return k2bio_df


def load_caseolap_kg(caseolap_predictions_edge_file):
    caseolap_kg = pd.read_csv(caseolap_predictions_edge_file)
    
    # Get CVD to disease mapping
    disease_to_mesh_df = caseolap_kg[caseolap_kg['relation'] == 'MeSH_CVD']
    
    # Normalize protein IDs in the 'tail' column
    caseolap_kg['tail'] = caseolap_kg['tail'].apply(normalize_protein_ids)
    
    # Extract the modified DataFrame
    mask = (caseolap_kg['relation'] == 'predicted_association') | (caseolap_kg['relation'] == 'CaseOLAP_score')
    disease_protein_predictions_df = caseolap_kg[mask]
    
    return disease_to_mesh_df, disease_protein_predictions_df


def merge_k2bio_and_caseolap_predictions(k2bio_df, caseolap_predictions_df, caseolap_disease_map_df):
    
    # Drop duplicate rows and rename dataframe columns
    caseolap_predictions_df.columns = ['h','r','t','weight']
    caseolap_disease_map_df.columns = ['h','r','t','weight']
    k2bio_df = k2bio_df.drop_duplicates(keep='first', inplace=False)
    caseolap_predictions_df = caseolap_predictions_df.drop_duplicates(keep='first', inplace=False)
    caseolap_disease_map_df = caseolap_disease_map_df.drop_duplicates(keep='first', inplace=False)
    
    k2bio_nodes = set(k2bio_df['h']).union(set(k2bio_df['t']))
    print(f"Total nodes in k2bio_df: {len(k2bio_nodes)}")
    print(f"Total edges in k2bio_df: {len(k2bio_df)}")

    # Check if all proteins and diseases show up
    caseolap_nodes = set(caseolap_predictions_df['h']).union(set(caseolap_predictions_df['t']))
    disease_nodes = set(caseolap_disease_map_df['h'])
    missing_proteins = set()
    replacement_map = {}
    for item in caseolap_nodes:
        if item not in k2bio_nodes and item not in disease_nodes:
            # Attempt to resolve by appending "_HUMAN"
            new_item = item + "_HUMAN"
            if new_item not in k2bio_nodes:
                print(f"Warning! Missing protein: {item}")
                missing_proteins.add(item)
            else:
                # If the new_item resolves the issue, record it for replacement
                replacement_map[item] = new_item

    # Replace items in caseolap_predictions_df using the map
    caseolap_predictions_df['h'] = caseolap_predictions_df['h'].apply(lambda x: replacement_map.get(x, x))
    caseolap_predictions_df['t'] = caseolap_predictions_df['t'].apply(lambda x: replacement_map.get(x, x))

    # Filter out rows with missing proteins from caseolap_predictions_df
    filtered_caseolap_predictions_df = caseolap_predictions_df[~caseolap_predictions_df['t'].isin(missing_proteins)]
    print(f"Total edges in filtered_caseolap_predictions_df: {len(filtered_caseolap_predictions_df)}")
    print(f"Total edges in caseolap_predictions_df: {len(caseolap_predictions_df)}")
    print(f"Total edges in caseolap_disease_map_df: {len(caseolap_disease_map_df)}")

    # Concatenate the DataFrames
    merged_df = pd.concat([k2bio_df, filtered_caseolap_predictions_df, caseolap_disease_map_df], ignore_index=True)
    merged_nodes = set(merged_df['h']).union(set(merged_df['t']))
    print(f"Total nodes in merged_df: {len(merged_nodes)}")
    print(f"Total edges in merged_df: {len(merged_df)}")

    return merged_df


def filter_knowledge_graph(kg_df, start_nodes, k=2):
    ''' This function extracts the edges between nodes which are reachable k-hop from any nodes in start_nodes'''
    # Ensure 'weight' column exists for compatibility; set to 1 if absent
    if 'weight' not in kg_df.columns:
        kg_df['weight'] = 1.0
    
    # Create a directed graph from the DataFrame including edge attributes
    G = nx.from_pandas_edgelist(kg_df, 'h', 't', edge_attr=True, create_using=nx.DiGraph())
    
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
        print("Usage: python kg_merge_filter.py <caseolap_kg> <know2bio_kg> <output_folder>")
        sys.exit(1)

    caseolap_kg_file = sys.argv[1]
    know2bio_edges = sys.argv[2]
    output_folder = sys.argv[3]
    
    combined_out_edges_file = os.path.join(output_folder,"textmining_k2bio_knowledge_graph_whole_edges.csv")
    combined_out_nodes_file = os.path.join(output_folder,"./textmining_k2bio_knowledge_graph_whole_nodes.csv")
    filtered_out_edges_file = os.path.join(output_folder,"./textmining_k2bio_knowledge_graph_filtered_k2_edges.csv")
    filtered_out_nodes_file = os.path.join(output_folder,"./textmining_k2bio_knowledge_graph_filtered_k2_nodes.csv")

    # Load Know2BIO knowledge graph
    know2bio_df = load_know2bio(know2bio_edges)
    # Load CaseOLAP LIFT text mining result knowledge graph
    disease_to_mesh_df, disease_protein_predictions_df = load_caseolap_kg(caseolap_kg_file)
    # Combine knowledge graphs into one kg
    merged_kg_df = merge_k2bio_and_caseolap_predictions(know2bio_df, disease_protein_predictions_df, disease_to_mesh_df)
    # Filter subgraph relevant to study diseases
    disease_nodes = set(disease_to_mesh_df['h'])
    sub_df = filter_knowledge_graph(merged_kg_df,disease_nodes)
    print(f"Diseases: {disease_nodes}; k={k}; Filtered subgraph: {sub_df.shape}")

    # Prepare node mapping df
    merged_kg_nodes_df = node_type_mapping(merged_kg_df)
    sub_df_nodes_df = node_type_mapping(sub_df)

    # Output files
    merged_kg_df.to_csv(combined_out_edges_file, index=False)
    merged_kg_nodes_df.to_csv(combined_out_nodes_file, index=False)
    sub_df.to_csv(filtered_out_edges_file, index=False)
    sub_df_nodes_df.to_csv(filtered_out_nodes_file, index=False)

    print(f"{merged_kg_df.shape[0]} lines written to {combined_out_edges_file}")
    print(f"{merged_kg_nodes_df.shape[0]} lines written to {combined_out_nodes_file}")
    print(f"{sub_df.shape[0]} lines written to {filtered_out_edges_file}")
    print(f"{sub_df_nodes_df.shape[0]} lines written to {filtered_out_nodes_file}")
    print("Finished.")

