# test_neo4j.py tests the functionality for neo4j_api

import sys
sys.path.append('../')

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


if __name__ == '__main__':
    unittest.main()
    loader = unittest.TestLoader()

    # Define test order
    test_order = [
        'test_neo4j_config',
        'test_neo4j_connection'
    ]

    # Run each test individually
    for test_name in test_order:
        suite = loader.loadTestsFromName(f"{TestNeo4j.__name__}.{test_name}")
        unittest.TextTestRunner(verbosity=2).run(suite)