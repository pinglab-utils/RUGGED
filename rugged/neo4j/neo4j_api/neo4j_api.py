import os
import sys
sys.path.append("../")
from neo4j import GraphDatabase, RoutingControl
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
from utils.logger import get_log_file, write_to_log 

def log_neo4j_resuts(text):
    log_folder = os.path.join('../neo4j_log')
    log_file = get_log_file(log_folder)
    write_to_log(log_file, text)

class Neo4j_API:

    def __init__(self):
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        self.verify_connection()

    def get_node_type_properties(self):
        """
        This function returns the node schema of the graph.
        The results are returned as nodeType to a list of protertyName (node features)
        """
        query = "call db.schema.nodeTypeProperties"

        # Make query to Neo4j
        result = self.search(query)

        # Parse results
        node_types = {}
        for rec in result.records:
            assert len(rec['nodeLabels']) == 1, "Node types should be unique"

            node_type = rec['nodeLabels'][0]
            node_properties = rec['propertyName']
            node_types[node_type] = node_properties
        
        # Write to log
        text = """
        QUERY: {}
        RESPONSE: {}
        """.format(query, node_types)
        log_neo4j_resuts(text)

        return node_types
    
    def get_node_types(self):
        return self.get_node_type_properties().keys()
    
    def get_node_types_as_csv(self):
        """
        Return a dict of the node types.
        From get_node_type_properties() method.
        """
        return self.get_node_type_properties()

    def get_rel_types(self):
        """
        This function returns the relationship types in the graph as a list.
        """

        query = '''MATCH ()-[r]->()
                RETURN DISTINCT type(r) AS relationshipType;
                '''

        # Make query to Neo4j
        result = self.search(query)

        # Parse results
        relationships =[]
        for rec in result.records:
            relationships.append(rec['relationshipType'])
        
        # Write to log
        text = """
        QUERY: {}
        RESPONSE: {}
        """.format(query, relationships)
        log_neo4j_resuts(text)

        return relationships

    def get_uniq_relation_pairs(self):
        """
        This function returns the unique relationships and the node types it connects.
        """
        query = '''MATCH (n1)-[r]->(n2)
                RETURN DISTINCT type(r) AS relationshipType, labels(n1) AS nodeType1, labels(n2) AS nodeType2;
                '''

        # Make query to Neo4j
        result = self.search(query)

        # Parse results
        relationships = []
        for rec in result.records:
            assert len(rec['nodeType1']) == 1 and len(rec['nodeType2']) == 1, "Node types should be unique"

            res = (rec['nodeType1'][0], rec['relationshipType'], rec['nodeType2'][0])
            relationships.append(res)
        
        # Write to log
        text = """
        QUERY: {}
        RESPONSE: {}
        """.format(query, relationships)
        log_neo4j_resuts(text)

        return relationships

    def verify_connection(self):
        """
        This function verifies the connection to Neo4j.
        """
        try:
            self.driver.verify_connectivity()
        except Exception as e:
            print("Unable to verify connection.")
            print("Error: {}".format(e))
        return True

    def search(self, query):
        """
        This function handles execution of a Cypher query to Neo4j and returns output.
        """
        try:
            response = self.driver.execute_query(query, database_="neo4j", routing_=RoutingControl.READ)
            return response
        except Exception as e:
            print("Query was unsuccessful.")
            print("Error: {}".format(e))
            return False

    def query(self):
        """
        This function loops an interactive query builder.
        """
        query = input("Enter a query: ")

        while True:
            if query == 'q':
                break
            else:
                result = self.search(query)

            # Enable user to enter another query if previous failed
            if not result:
                query = input("Enter a query: ")
            else:
                print(result)
                break

    def example_node(self, n=10):
        '''
        This function returns examples of nodes in the graph based on user selected node type.
        '''

        # Get user to specify node type for example
        print("Which node type do you want to see examples of?")
        node_to_prop = self.get_node_type_properties()
        idx_to_node = {idx: node for idx, node in enumerate(node_to_prop.keys())}
        for idx, node_type in idx_to_node.items():
            print(f"{idx}: {node_type}")

        user_input = input("Enter a number: ")
        while True:
            valid_input = int(user_input) in idx_to_node.keys()
            if valid_input:
                break
            else:
                print("Invalid input")
                user_input = input("Enter a number: ")

        # Make query
        query = '''MATCH (n:%s)
        RETURN n LIMIT %s
        ''' % (idx_to_node[int(user_input)], n)
        result = self.search(query)

        # Parse results
        node_names = []
        if result:
            for rec in result.records:
                node_names.append(rec['n']['name'])

        # Write to log
        text = """
        QUERY: {}
        RESPONSE: {}
        """.format(query, node_names)
        log_neo4j_resuts(text)
        
        return node_names

    def example_relationship(self, n=10):
        '''
        This function returns examples of relationships in the graph based on user selected relationship type.
        '''

        # Get user to specify relationship type for example
        print("Which relationship type do you want to see examples of?")
        relationships = self.get_rel_types()
        idx_to_rel = {idx: rel for idx, rel in enumerate(relationships)}
        for idx, rel in idx_to_rel.items():
            print(f"{idx}: {rel}")

        user_input = input("Enter a number: ")
        while True:
            valid_input = int(user_input) in idx_to_rel.keys()
            if valid_input:
                break
            else:
                print("Invalid input")
                user_input = input("Enter a number: ")

        # Make query
        query = '''MATCH (n1)-[r:`%s`]->(n2)
        RETURN n1, r, n2 LIMIT %s
        ''' % (idx_to_rel[int(user_input)], n)
        result = self.search(query)

        # Parse results
        relationships = []
        if result:
            for rec in result.records:
                n1 = rec['n1']['name']
                r = rec['r'].type
                n2 = rec['n2']['name']
                relationships.append((n1, r, n2))
        
        # Write to log
        text = """
        QUERY: {}
        RESPONSE: {}
        """.format(query, relationships)
        log_neo4j_resuts(text)

        return relationships


def get_user_prompt(options):
    """
    Return a user prompt based on the functionalities.
    """
    prompt = "Please select an option:\n"
    for key, value in options.items():
        prompt += f"{key}: {value[0]}\n"
    return prompt


def interactive():
    interactive_options = {'q': ('Query', 'Query the Neo4j Database with your own Cypher command'),
                           'n': ('Node Properties', 'Get the node schema of the graph'),
                           'r': ('Relationships', 'Get the relationship types in the graph'),
                           'u': ('Unique relationships', 'Get the unique relationships and the node types it connects'),
                           'en': ('Example node', 'Get examples of nodes in the graph based on node type'),
                           'er': ('Example relationship', 'Get examples of relationships in the graph based on '
                                                          'relationship type')
                           }

    neo4j_api = Neo4j_API()

    while True:
        # Get user input
        user_input = input(get_user_prompt(interactive_options))

        if user_input == 'q':
            print("Query.")
            neo4j_api.query()
        elif user_input == 'n':
            print("Node Properties.")
            result = neo4j_api.get_node_type_properties()
            for p, v in result.items():
                print(p + ",{}".format(v.split()))
            print("Total number of node types: ", len(result), "\n")
        elif user_input == 'r':
            print("Relationships.")
            result = neo4j_api.get_rel_types()
            for r in result:
                if r[-1] == '>':
                    print("\"`{}`\"".format(r))
                else:
                    print("\"{}\"".format(r))
            print("Total number of relationships: ", len(result), "\n")
        elif user_input == 'u':
            print("Unique relationships.")
            result = neo4j_api.get_uniq_relation_pairs()
            for n1, r, n2 in result:
                print(n1, r, n2)
            print("Total number of unique relationships: ", len(result), "\n")
        elif user_input == 'en':
            print("Example node.")
            result = neo4j_api.example_node()
            for r in result:
                print(r)
        elif user_input == 'er':
            print("Example relationship.")
            result = neo4j_api.example_relationship()
            for r in result:
                print(r)

if __name__ == "__main__":
    interactive()