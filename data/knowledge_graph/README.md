# Knowledge Graph

This project requires three primary input files to represent Knowledge Graph (KG) data: 
* edges.tsv: Tab-separated values (TSV) representing edges of the KG.
* nodes.tsv: Tab-separated values (TSV) detailing nodes and their node types within the KG.
* node_properties.json: JSON file representing node features of each node.

Each file should follow the specified format and file type to ensure compatibility with the program.

## 1. Edges File (edges.tsv)

The edges file contains triples representing the relationships in the knowledge graph. Each row in the file represents an edge in the following format:


`head    relation    tail    weight (optional)`


* head: The ID of the source node (e.g., "Node1").
* relation: The type of relationship between nodes (e.g., "interacts_with").
* tail: The ID of the target node (e.g., "Node2").
* weight (optional): A numerical value representing the weight or strength of the relationship (e.g., "0.75").
Do not include the header within the file.

## 2. Nodes File (nodes.tsv)

The nodes file maps each node to its corresponding type. Each row in the file should follow this format:


`node_id    node_type`


node_id: The unique identifier for the node (e.g., "Node1").
node_type: The type or category of the node (e.g., "UniProt", "Entrez").
Typically, follow the namespace identifier (node type) followed by the node name (i.e., Uniprot:Protein1)

## 3. Node Properties File (node_properties.json)

The node properties file is a JSON dictionary where each key represents a node ID, and the value is another dictionary containing the properties of that node. Properties can include metadata, attributes, or any additional information.
Format:


```
{
  "node_id": {
    "property_1": "value_1",
    "property_2": "value_2",
    ...
  },
  ...
}
```



