# test_koboldai.py tests the functionality for the KoboldAI API
import sys
sys.path.append('../')

import unittest
from config import KOBOLDAI_URI
from rugged.llms.koboldai_llm import KoboldAILLM

class TestKoboldAI(unittest.TestCase):

    def test_koboldai_config(self):
        self.assertIsNotNone(KOBOLDAI_URI, "KOBOLDAI_URI should not be None")

        print("KOBOLDAI_URI:", KOBOLDAI_URI)

        
    def test_koboldai_api_call(self):
        """Test the invoke method of KoboldAILLM."""        
        user_input = "Test user input"
        llm = KoboldAILLM()
        llm.invoke(user_input)
        response = llm.get_response()
 
        self.assertIsNotNone(response, "KoboldAI API response should not be None")
        print(user_input)
        print(response)
        print("KoboldAI API call successful\n")


if __name__ == '__main__':
    unittest.main()
    loader = unittest.TestLoader()
    # Define test order
    test_order = [
        'test_koboldai_config',
        'test_koboldai_api_call'
    ]

    # Run each test individually
    for test_name in test_order:
        suite = loader.loadTestsFromName(f"{TestKoboldAI.__name__}.{test_name}")
        unittest.TextTestRunner(verbosity=2).run(suite)
