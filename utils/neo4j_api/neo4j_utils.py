import re
import json
import ijson

import sys
from utils.neo4j_api.neo4j_driver import Driver

from config import NODE_FEATURES_PATH, NODE_TYPES_PATH, EDGE_TYPES_PATH, QUERY_EXAMPLES

def get_node_features(node_id):
    """Retrieve node features for a specific node from a flat JSON file."""
    with open(NODE_FEATURES_PATH, 'r') as file:
        # Iterate over the top-level keys and values in the JSON
        for key, value in ijson.kvitems(file, ''):
            if key == node_id:
                return value  
    return None  # Return None if the node features are not found

def get_node_and_edge_types(node_path=NODE_TYPES_PATH, edge_path=EDGE_TYPES_PATH):
    node_types_path = node_path
    edge_types_path = edge_path

    node_types = ""
    edge_types = ""

    with open(node_types_path) as f:
        for node in f:
            node_types += f"\"{node.strip()}\";"
        node_types = node_types[:-1]

    with open(edge_types_path) as f:
        for edge in f:
            edge_types += f"\"{edge.strip()}\","
        edge_types = edge_types[:-1]

    return node_types, edge_types


def read_query_examples(query_examples_path=QUERY_EXAMPLES):
    with open(query_examples_path, 'r') as f:
        query_examples = f.read()
        return query_examples


def extract_code(response: str):
    code_blocks = re.findall(r'```(.*?)```', response, re.DOTALL)
    # Combine code to be one block
    code = '\n'.join(code_blocks)
    return code

# Test to see what types of nodes there are
def find_node_names(max_nodes_to_return=5, returned_nodes=['MeSH_Compound:C568512', 'molecular_function:0140775',
                                                           'MeSH_Tree_Disease:C17.800.893.592.450.200']):
    node_names = dict()
    # Check all node types
    for i, returned_node in enumerate(returned_nodes):
        if i == max_nodes_to_return:
            break
        with open(NODE_FEATURES_PATH) as f:
            # Set up iterator for a single node
            nodes_objects = ijson.items(f, returned_node)
            node_object = next(nodes_objects, None)
            # If there is a names attribute, then use those
            if node_object and 'names' in node_object.keys():
                node_names[returned_node] = node_object['names']
            else:
                node_names[returned_node] = ["No known names"]
            continue
    print("returned_nodes:", returned_nodes)
    print("max_nodes_to_return:", max_nodes_to_return)
    return node_names


def extract_results(query: str, driver, is_json=False):
    nodes = driver.query_database(query)

    # Set up the regex
    node_names = ["MeSH_Compound", "Entrez", "UniProt", "Reactome_Reaction", "MeSH_Tree_Disease", "MeSH_Disease",
                  "Reactome_Pathway", "MeSH_Anatomy", "cellular_component", "molecular_function", "MeSH_Tree_Anatomy",
                  "ATC", "DrugBank_Compound", "KEGG_Pathway", "biological_process"]
    node_finder_pattern = r'(\b(?:' + '|'.join(map(re.escape, node_names)) + r')\S*)'

    # Check if json returned an invalid or valid object
    if is_json:
        try:
            string_json_nodes = json.dumps(nodes)
        except:
            return {}
        matches = re.compile(node_finder_pattern).findall(string_json_nodes)
    else:
        matches = re.compile(node_finder_pattern).findall(str(nodes))

    # Clean up results
    replace_chars = ['{', '}', '\'', '\"', '[', ']', ',', '(', ')']
    print('MATCHES BEFORE CLEAN UP:', matches)
    for i, match in enumerate(matches):
        print(match)
        for char in replace_chars:
            match = match.replace(char, "")
        matches[i] = match
    # Take set
    print('MATCHES:', matches)

    node_features = find_node_names(returned_nodes=list(set(matches)))
    return matches, node_features
