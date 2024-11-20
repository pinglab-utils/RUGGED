# test_koboldai.py tests the functionality for the KoboldAI API
import sys
sys.path.append('../')

import unittest
from config import HF_MODEL
from rugged.llms.huggingface_llm import HuggingFaceLLM

class TestHuggingfaceLLM(unittest.TestCase):

    def test_huggingface_config(self):
        self.assertIsNotNone(HF_MODEL, "HF_MODEL should not be None")

        print("HF_MODEL:", HF_MODEL)

        
    def test_huggingface_api_call(self):
        """Test the invoke method of HuggingFaceLLM."""        
        user_input = "Test user input"
        llm = HuggingFaceLLM(device=-1)
        llm.invoke(user_input)
        response = llm.get_response()
 
        self.assertIsNotNone(response, "HuggingFace LLM response should not be None")
        print(user_input)
        print(response)
        print("HuggingFace LLM call successful\n")


if __name__ == '__main__':
    unittest.main()
    loader = unittest.TestLoader()
    # Define test order
    test_order = [
        'test_huggingface_config',
        'test_huggingface_api_call'
    ]

    # Run each test individually
    for test_name in test_order:
        suite = loader.loadTestsFromName(f"{TestHuggingfaceLLM.__name__}.{test_name}")
        unittest.TextTestRunner(verbosity=2).run(suite)
