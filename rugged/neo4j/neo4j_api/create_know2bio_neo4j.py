import sys
sys.path.append("../../../")

import pandas as pd
from neo4j import GraphDatabase

from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

def clear_graph(session, debug=True):
    """
    This function deletes all nodes and edges in the graph.
    """
    query = """
    MATCH (n)
    DETACH DELETE n
    """
    session.run(query)
    
    if debug:
        count_nodes_and_relationships(session, debug=True)

def count_nodes_and_relationships(session, debug=True):
    """
    This function counts the number of nodes and edges in the graph.
    """
    # Count nodes
    node_query = "MATCH (n) RETURN COUNT(n) AS nodeCount"
    nodes = session.run(node_query).single()["nodeCount"]

    # Count relationships
    relationship_query = "MATCH ()-->() RETURN COUNT(*) AS relationshipCount"
    rels = session.run(relationship_query).single()["relationshipCount"]
    
    if debug:
        print(f"Number of nodes: {nodes}")
        print(f"Number of relationships: {rels}")

    return nodes, rels


def create_nodes_batch(session, nodes, batch_size=1000):
    """
    This function creates nodes in batches to improve speed.
    """
    # Group nodes by their type to avoid dynamic labels in the Cypher query
    nodes_by_type = {}
    for node in nodes:
        node_type = node.split(":")[0]
        node_name = node_type + ":" + ":".join(node.split(":")[1:])
        if node_type not in nodes_by_type:
            nodes_by_type[node_type] = []
        nodes_by_type[node_type].append({"name": node_name})

    # Iterate over each node type and create nodes in batches
    for node_type, nodes_list in nodes_by_type.items():
        for i in range(0, len(nodes_list), batch_size):
            batch = nodes_list[i:i + batch_size]
            query = f"""
            UNWIND $batch AS row
            CREATE (n:`{node_type}` {{name: row.name}})
            """
            session.run(query, batch=batch)
            print(f"Nodes of type '{node_type}' batch created: {i + len(batch)} out of {len(nodes_list)}", end="\r", flush=True)
        print() # New line for each node type

def create_relationships_batch(session, relationships, batch_size=1000):
    """
    This function creates relationships in batches, grouping by relationship type to avoid dynamic Cypher.
    """
    # Group relationships by type
    relationships_by_type = {}
    for h, r, t in relationships:
        if r not in relationships_by_type:
            relationships_by_type[r] = []
        relationships_by_type[r].append({"e1_name": h, "e2_name": t})

    # Iterate over each relationship type and create relationships in batches
    for rel_type, rel_list in relationships_by_type.items():
        for i in range(0, len(rel_list), batch_size):
            batch = rel_list[i:i + batch_size]
            # Note that the relationship type is directly inserted into the query
            query = f"""
            UNWIND $batch AS row
            MATCH (e1 {{name: row.e1_name}})
            MATCH (e2 {{name: row.e2_name}})
            MERGE (e1)-[:`{rel_type}`]->(e2)
            """
            session.run(query, batch=batch)
            print(f"Relationships of type '{rel_type}' batch created: {i + len(batch)} out of {len(rel_list)}", end="\r", flush=True)
        print() # New line for each node type

        
def load_kg(kg_file):
    df = pd.read_csv(kg_file, sep="\t")
    df.columns = ['h', 'r', 't']

    nodes = set(df['h']).union(set(df['t']))
    uniq_edges = set(df['r'])
    relationships = list(zip(df['h'], df['r'], df['t']))

    print("%d nodes and %d edges (%d unique edges)" % (len(nodes), len(relationships), len(uniq_edges)))
    return df, nodes, relationships

class Neo4jConnector:
    def __init__(self, uri, user, password):
        self._driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self._driver.close()        

# Connect to Neo4j Server
print("Connecting to Neo4j Server")
connector = Neo4jConnector(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

# Load Knowledge Graph Data
print("Loading Knowledge Graph Data")
input_file = '../../../data/knowledge_graph/whole_kg.txt'
_, nodes, relationships = load_kg(input_file)

with connector._driver.session() as session:
    # Clear graph if needed
    clear_graph(session, debug=False)
    
    # Create nodes in batches
    print("Creating nodes in batches")
    create_nodes_batch(session, nodes)
    print(f"\n{len(nodes)} nodes created.")
    
    # Create relationships in batches
    print("Creating relationships in batches")
    create_relationships_batch(session, relationships)
    print(f"\n{len(relationships)} relationships created.")
    
    # Print summary
    node_count, relationship_count = count_nodes_and_relationships(session)
    print(f"Final count - Nodes: {node_count}, Relationships: {relationship_count}")

# Close connection
connector.close()
