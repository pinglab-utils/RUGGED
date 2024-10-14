# test_neo4j.py tests the functionality for neo4j_api

import os, sys

# Add the root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import unittest
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
from neo4j import GraphDatabase


class TestNeo4j(unittest.TestCase):

    def test_neo4j_config(self):
        self.assertIsNotNone(NEO4J_URI, "NEO4J_URI should not be None")
        self.assertIsNotNone(NEO4J_USER, "NEO4J_USER should not be None")
        self.assertIsNotNone(NEO4J_PASSWORD, "NEO4J_PASSWORD should not be None")

        print("NEO4J_URI:", NEO4J_URI)
        print("NEO4J_USER:", NEO4J_USER)
        print("NEO4J_PASSWORD:", NEO4J_PASSWORD)

    def test_neo4j_connection(self):
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        with driver.session() as session:
            result = session.run("RETURN 1 AS x")
            record = result.single()
            self.assertEqual(record["x"], 1)

            driver.close()

        print("Neo4j connection successful\n")
        # NOTE: First time setting up Neo4j, you may have to change credentials
        # This command in the Neo4j Browser worked for me
        # ALTER USER neo4j_api SET PASSWORD 'password';

    def test_neo4j_database(self):
        # This test will count the number of nodes and edges in the graph
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        with driver.session() as session:
            # Count nodes
            node_count_query = "MATCH (n) RETURN count(n) AS node_count"
            node_result = session.run(node_count_query)
            node_count = node_result.single()["node_count"]

            # Count edges (relationships)
            edge_count_query = "MATCH ()-[r]->() RETURN count(r) AS edge_count"
            edge_result = session.run(edge_count_query)
            edge_count = edge_result.single()["edge_count"]

            print(f"\nNumber of nodes: {node_count}")
            print(f"Number of edges: {edge_count}")

            # Asserting that there is at least one node and one edge for the test to pass
            self.assertGreaterEqual(node_count, 1, "There are no nodes. Is the graph empty?")
            self.assertGreaterEqual(edge_count, 1, "There are no edges. Is the graph empty?")

            driver.close()

        print("Neo4j database node and edge counts verified\n")



if __name__ == '__main__':
    unittest.main()
    loader = unittest.TestLoader()

    # Define test order
    test_order = [
        'test_neo4j_config',
        'test_neo4j_connection',
        'test_neo4j_database'
    ]

    # Run each test individually
    for test_name in test_order:
        suite = loader.loadTestsFromName(f"{TestNeo4j.__name__}.{test_name}")
        unittest.TextTestRunner(verbosity=2).run(suite)
