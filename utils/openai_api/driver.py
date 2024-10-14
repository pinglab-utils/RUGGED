import sys
sys.path.append('../')
from neo4j import GraphDatabase, RoutingControl
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
import json

URI = NEO4J_URI
AUTH = (NEO4J_USER, NEO4J_PASSWORD)

node_records = dict()

def get_names(driver, node):
    records = driver.execute_query(
        "MATCH (a:{node}) RETURN a.name".format(node=node),
        node=node, database_="neo4j", routing_=RoutingControl.READ,
    )
    return records

def process_records(records, node):
    for i in range(len(records[0])):
        node_records[records[0][i][0]] = node

def main():
    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        node = ['DrugBank_Compound', 'ATC', 'Entrez', 'KEGG_Pathway', 'MeSH_Anatomy', 'MeSH_Compound',
                'MeSH_Disease', 'MeSH_Tree_Anatomy', 'MeSH_Tree_Disease', 'Reactome_Pathway', 'Reactome_Reaction',
                'UniProt', 'biological_process', 'cellular_component', 'molecular_function']
        
        for n in node:
            print('On node', n)
            records = get_names(driver, n)
            process_records(records, n)
    
    with open('node_records.json', 'w') as outf:
        json.dump(node_records, outf, indent=4, sort_keys=False)

if __name__ == "__main__":
    main()
    
