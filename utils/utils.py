import re
import json
from pathlib import Path

def extract_code(response: str):
    code_blocks = re.findall(r'```(.*?)```', response, re.DOTALL)
    # Combine code to be one block
    code = '\n'.join(code_blocks)
    return code


def get_project_root() -> Path:
    return Path(__file__).parent.parent


def format_node_name(node_id, node_names):
    if node_id not in node_names:
        return f"{node_id} No known names"
    if node_names[0] == "No known names":
        return f"{node_id} ({node_names[0]})"
    return f"{node_names[0]} ({node_id})"

## TODO is this in neo4j util class?
#def extract_results(query: str, driver ,is_json=False):
#    nodes = driver.query_database(query)
#
#    # Set up the regex
#    node_names = ["MeSH_Compound", "Entrez", "UniProt", "Reactome_Reaction", "MeSH_Tree_Disease", "MeSH_Disease",
#                  "Reactome_Pathway", "MeSH_Anatomy", "cellular_component", "molecular_function", "MeSH_Tree_Anatomy",
#                  "ATC", "DrugBank_Compound", "KEGG_Pathway", "biological_process"]
#    node_finder_pattern = r'(\b(?:' + '|'.join(map(re.escape, node_names)) + r')\S*)'
#
#    # Check if json returned an invalid or valid object
#    if is_json:
#        try:
#            string_json_nodes = json.dumps(nodes)
#        except:
#            return {}
#        matches = re.compile(node_finder_pattern).findall(string_json_nodes)
#    else:
#        matches = re.compile(node_finder_pattern).findall(str(nodes))
#
#    # Clean up results
#    replace_chars = ['{', '}', '\'', '\"', '[', ']', ',', '(', ')']
#    print('MATCHES BEFORE CLEAN UP:', matches)
#    for i, match in enumerate(matches):
#        print(match)
#        for char in replace_chars:
#            match = match.replace(char, "")
#        matches[i] = match
#    # Take set
#    print('MATCHES:', matches)
#
#    return find_node_names(returned_nodes=list(set(matches)))

# Redundant function, seems to be in neo4j util class now
# def find_node_names(max_nodes_to_return=100, returned_nodes=['MeSH_Compound:C568512', 'molecular_function:0140775', 'MeSH_Tree_Disease:C17.800.893.592.450.200'], debug=False):
#     node_data = dict()
    
#     # Load the entire JSON data
#     with open(NODE_FEATURES_PATH, 'r') as f:
#         data = json.load(f)
        
#         valid_count = 0  # Track the number of valid nodes found
        
#         for returned_node in returned_nodes:
#             if valid_count == max_nodes_to_return:
#                 break
            
#             if debug:
#                 print(f"Processing node: {returned_node}")
            
#             # Check if returned_node exists in the JSON data
#             if returned_node in data:
#                 node_object = data[returned_node]
                
#                 # Collect names and description if they exist
#                 entry = {}
#                 if 'names' in node_object:
#                     entry['names'] = node_object['names']
#                 if 'description' in node_object:
#                     entry['description'] = node_object['description']
                
#                 # Only add the entry if at least one field is present
#                 if entry:
#                     node_data[returned_node] = entry
#                     valid_count += 1
#                     if debug:
#                         print(f"Data found for {returned_node}: {entry}")
#                 elif debug:
#                     print(f"No 'names' or 'description' for node: {returned_node}")
#             elif debug:
#                 print(f"Path '{returned_node}' not found in JSON.")
    
#     if debug:
#         print("returned_nodes:", returned_nodes)
#         print("max_nodes_to_return:", max_nodes_to_return)
    
#     return node_data
