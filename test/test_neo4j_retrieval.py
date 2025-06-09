import os, sys, argparse

# Add the root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from rugged.knowledge_graph.query import QuerySystem


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run a Cypher query for test retrieval.")
    parser.add_argument('--query', type=str, help='Cypher query string')
    args = parser.parse_args()
    
    # Default directories
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
    log_file = os.path.join(base_dir, 'log', 'test_log.txt')
        
    cypher_query = args.query or ''' MATCH (dr:DrugBank_Compound)-[r]-(d:MeSH_Disease)
        WHERE d.name IN ['MeSH_Disease:D019571', 'MeSH_Disease:D002311']
        RETURN dr.name AS Drug, COLLECT(DISTINCT d.name) AS Diseases
        '''
    try:
        # Initialize system
        qs = QuerySystem(log_file, query_only=True)
        print("Testing query")
        print(cypher_query)

        # Retrieve results
        results = qs.retrieve_graph(cypher_query)
        query_results = results['query_results']
        node_features = results['node_features']
        print(query_results)
        print(node_features)
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == '__main__':
    main()

