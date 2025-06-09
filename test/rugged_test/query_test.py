import os, sys
# Add the root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from rugged.knowledge_graph.query import QuerySystem


def main():
    # Default directories
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    log_file = os.path.join(base_dir, 'log', 'test_log.txt')
    
    question = 'What drugs are currently being prescribed to treat Arrhythmogenic Cardiomyopathy?'
    try:
        print('User Input: ' + question)
        qs = QuerySystem(log_file)
        print("Query System initialized successfully")
        
        print("Crafting query...")
        success, cypher_query = qs.create_query(question)
        print(cypher_query)
        
        print("Accessing information...")
        query_results, node_features = qs.retrieve_graph(cypher_query)
        print(query_results)
        print(node_features)
        
        print("Generating response...")
        response = qs.evaluate_results(question, cypher_query, query_results, node_features)
        print(response)

    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == '__main__':
    main()

