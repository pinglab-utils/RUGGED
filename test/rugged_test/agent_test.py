import os
import sys
import unittest

# Add the root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from rugged.agents import agent_loader
from config import REASONING_AGENT, TEXT_EVALUATOR_AGENT, CYPHER_QUERY_AGENT, QUERY_VERIFICATION_AGENT

class TestAgents(unittest.TestCase):

    def test_config(self):
        """Test that all agent configurations are not None."""
        self.assertIsNotNone(REASONING_AGENT, "REASONING_AGENT should not be None")
        self.assertIsNotNone(TEXT_EVALUATOR_AGENT, "TEXT_EVALUATOR_AGENT should not be None")
        self.assertIsNotNone(CYPHER_QUERY_AGENT, "CYPHER_QUERY_AGENT should not be None")
        self.assertIsNotNone(QUERY_VERIFICATION_AGENT, "QUERY_VERIFICATION_AGENT should not be None")
        # Print agent configurations
        print("REASONING_AGENT:", REASONING_AGENT)
        print("TEXT_EVALUATOR_AGENT:", TEXT_EVALUATOR_AGENT)
        print("CYPHER_QUERY_AGENT:", CYPHER_QUERY_AGENT)
        print("QUERY_VERIFICATION_AGENT:", QUERY_VERIFICATION_AGENT)

    def test_reasoning_agent(self):
        """Test the instantiation and basic functionality of the reasoning agent."""
        reasoning_agent = agent_loader.load_reasoning_agent(REASONING_AGENT)
        response = reasoning_agent.run("This is a test")
        self.assertTrue(response, "Response should not be empty")
        print("Truncated reasoning agent response: ", response[:20])

    def test_text_evaluator_agent(self):
        """Test the instantiation and basic functionality of the text evaluator agent."""
        text_eval_agent = agent_loader.load_text_evaluator_agent(TEXT_EVALUATOR_AGENT)
        response = text_eval_agent.run("Evaluate this text")
        self.assertTrue(response, "Response should not be empty")
        print("Truncated text evaluator response: ", response[:20])

    def test_cypher_query_agent(self):
        """Test the instantiation and basic functionality of the cypher query agent."""
        cypher_query_agent = agent_loader.load_cypher_query_agent(CYPHER_QUERY_AGENT)
        response = cypher_query_agent.run("MATCH (n) RETURN n LIMIT 1")
        self.assertTrue(response, "Response should not be empty")
        print("Cypher query agent response: ", response[:20])

    def test_query_verification_agent(self):
        """Test the instantiation and basic functionality of the query verification agent."""
        query_verification_agent = agent_loader.load_query_verification_agent(QUERY_VERIFICATION_AGENT)
        response = query_verification_agent.run("Verify this query")
        self.assertTrue(response, "Response should not be empty")
        print("Query verification agent response: ", response[:20])


if __name__ == '__main__':
    unittest.main()
