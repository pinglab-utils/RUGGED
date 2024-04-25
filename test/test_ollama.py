# test_ollama.py tests the functionality for the Ollama API
import sys
sys.path.append('../')

import unittest
from config import OLLAMA_URI
from langchain_community.llms import Ollama

class TestOllama(unittest.TestCase):

    def test_ollama_config(self):
        self.assertIsNotNone(OLLAMA_URI, "OLLAMA_URI should not be None")

        print("OLLAMA_URI:", OLLAMA_URI)

    def test_ollama_api_call(self):
        user_input = "Test user input"

        llm = Ollama(model="llama2", base_url=OLLAMA_URI)

        response = llm.invoke(user_input)
        self.assertIsNotNone(response, "Ollama API response should not be None")
        print(user_input)
        print(response)
        print("Ollama API call successful\n")


if __name__ == '__main__':
    unittest.main()
    loader = unittest.TestLoader()
    # Define test order
    test_order = [
        'test_ollama_config',
        'test_ollama_api_call'
    ]

    # Run each test individually
    for test_name in test_order:
        suite = loader.loadTestsFromName(f"{TestOllama.__name__}.{test_name}")
        unittest.TextTestRunner(verbosity=2).run(suite)
