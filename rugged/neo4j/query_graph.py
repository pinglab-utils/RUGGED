import pandas as pd
from neo4j import GraphDatabase, RoutingControl
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

neo4j_uri = NEO4J_URI
neo4j_user = NEO4J_USER
neo4j_password = NEO4J_PASSWORD

def get_schema():
    with GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password)) as driver:
        return driver.execute_query("CALL db.schema.nodeTypeProperties", database_="neo4j", routing_=RoutingControl.READ)
    
def get_node_labels():
    with GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password)) as driver:
        return driver.execute_query("CALL db.labels()", database_="neo4j", routing_=RoutingControl.READ)
    
def get_relationship_types():
    with GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password)) as driver:
        return driver.execute_query("CALL db.relationshipTypes()", database_="neo4j", routing_=RoutingControl.READ)

if __name__ == "__main__":
    print("GRAPH SCHEMA")
    print(get_schema())
    print("NODE TYPES")
    print(get_node_labels())
    print("RELATIONSHIP TYPES")
    print(get_relationship_types())