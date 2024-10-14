# test_openai.py tests the functionality for the OpenAI API
import sys
sys.path.append('../')

import unittest
from config import OPENAI_KEY
from utils.openai_api.openai_client import OpenAI_API

class TestOpenAI(unittest.TestCase):

    def test_openai_config(self):
        self.assertIsNotNone(OPENAI_KEY, "OPENAI_KEY should not be None")

        print("OPENAI_KEY:", OPENAI_KEY)

    def test_openai_api_call(self):
        user_input = "Test user input"
        openai_api = OpenAI_API()
        #response = openai_api.single_chat(summarize=True)
        response = openai_api.call_openai_api(user_input)
        self.assertIsNotNone(response, "OpenAI API response should not be None")

        print(user_input)
        print(response)
        print("OpenAI API call successful\n")


if __name__ == '__main__':
    unittest.main()
    loader = unittest.TestLoader()
    import pdb;pdb.set_trace()
    # Define test order
    test_order = [
        'test_openai_config',
        'test_openai_api_call'
    ]

    # Run each test individually
    for test_name in test_order:
        suite = loader.loadTestsFromName(f"{TestOpenAI.__name__}.{test_name}")
        unittest.TextTestRunner(verbosity=2).run(suite)
