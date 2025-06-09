import sys
import re

from config import NODE_FEATURES_PATH
from config import BIAS_MITIGATION_PROMPT
from config import QUERY_REASONING_PROMPT
from utils.neo4j_api.neo4j_driver import Driver
from utils.neo4j_api.neo4j_utils import get_node_and_edge_types, read_query_examples, extract_results
from utils.utils import extract_code
from rugged.agents import agent_loader

from config import REASONING_AGENT, CYPHER_QUERY_AGENT, QUERY_VERIFICATION_AGENT

NODE_TYPES, EDGE_TYPES = get_node_and_edge_types()
QUERY_EXAMPLES = read_query_examples()

class QuerySystem:
    def __init__(self, log_file, query_only=False):
        self.name = 'QuerySystem'
        # Log file for writing program progress
        self.log_file = log_file
        
        # skip agent initialization if only querying the knowledge graph
        if not query_only:

            # LLM for reasoning the response
            self.reasoning_agent = agent_loader.load_reasoning_agent(REASONING_AGENT)

            # LLM for generating the cypher query
            self.cypher_query_agent = agent_loader.load_cypher_query_agent(CYPHER_QUERY_AGENT)

            # LLM for verifying the cypher query
            self.query_verification_agent = agent_loader.load_query_verification_agent(QUERY_VERIFICATION_AGENT)


    def create_query(self, user_query, max_tries = 5):
        ''' Generates a Cypher query based on user input '''
                
        # Track number of verification cycles
        count = 0
        query_history = []
        success = False
        # Cypher Query Agent generates a Cypher query based on user question
        cypher_query = self.cypher_query_agent.generate_query(user_query)
        print("Cypher Query Agent: ")
        print(cypher_query)
        query_history.append(cypher_query)
        while (count <= max_tries) and not success:

            # Retrieve results from Neo4j graph using cypher_query
            results = self.retrieve_graph(cypher_query)
            query_results = results['query_results']
            node_features = results['node_features']
            
            # Query Verification Agent checks the generated query
            cypher_code = extract_code(cypher_query)
            success, response = self.query_verification_agent.verify_query(user_query, cypher_code, query_results, node_features)
            print("Query Verification Agent: ")
            print(response)

            if not success:
                count += 1
                # Cypher Query Agent revises Cypher query based on Verification Agent response
                cypher_query = self.cypher_query_agent.revise_query(cypher_query, response)
                query_history.append(cypher_query)

        return success, extract_code(cypher_query)


    def retrieve_graph(self, cypher_code, max_entities=50):
        ''' Retrieves graph edges and node features related to the cypher query '''
        # Extract cypher code from the Cypher query agent response
        #cypher_code = extract_code(cypher_query)
        
        # Try to retrieve using query
        neo4j_driver = Driver()
        query_code, query_results, query_msg_raw = neo4j_driver.check_query(cypher_code)

        nodes, node_features = extract_results(query_results, neo4j_driver)

        success = (query_code == 0)
        return {'success':success,
                'query_results':query_results,
                'nodes':nodes,
                'node_features': node_features}

                
    def evaluate_response(self, user_query, cypher_query, query_results, node_features):
        prompt = self.reasoning_agent.prepare_query_prompt(user_query, cypher_query, query_results, node_features)
        response = self.reasoning_agent.reason(prompt)
        return response

    
