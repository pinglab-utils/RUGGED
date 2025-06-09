import os
import sys
import unittest

# Add the root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from rugged.agents import agent_loader

class TestLLMs(unittest.TestCase):

    def test_ollama(self):
        """Test the instantiation and basic functionality of the Ollama LLM."""
        llm = agent_loader.load_llm("ollama")
        llm.invoke("This is a test")
        response = llm.get_response()
        self.assertTrue(response, "Response should not be empty")
        print("Truncated response: ", response[:20])
        
    def test_kobold(self):
        """Test the instantiation and basic functionality of the KoboldAI LLM."""
        llm = agent_loader.load_llm("kobold")
        llm.invoke("This is a test")
        response = llm.get_response()
        self.assertTrue(response, "Response should not be empty")
        print("Truncated response: ", response[:20])
        
    def test_huggingface(self):
        """Test the instantiation and basic functionality of the HuggingFace LLM."""
        llm = agent_loader.load_llm("huggingface")
        llm.invoke("This is a test")
        response = llm.get_response()
        self.assertTrue(response, "Response should not be empty")
        print("Truncated response: ", response[:20])
        
    def test_openai(self):
        """Test the instantiation and basic functionality of the OpenAI LLM."""
        llm = agent_loader.load_llm("OpenAI") # TODO fix this in agent_loader and config to specify which model
        llm.invoke("This is a test")
        response = llm.get_response()
        self.assertTrue(response, "Response should not be empty")
        print("Truncated response: ", response[:20])


if __name__ == '__main__':
    unittest.main()
