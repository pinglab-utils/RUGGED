# test_openai.py tests the functionality for the OpenAI API
import sys
sys.path.append('../')
import time

import unittest
from config import OPENAI_KEY
from openai import OpenAI
from rugged.llms.openai_llm import OpenAILLM

class TestOpenAI(unittest.TestCase):

    def test_openai_config(self):
        
        print("Testing OpenAI API Key...")
        self.assertIsNotNone(OPENAI_KEY, "OPENAI_KEY should not be None")

        print("OPENAI_KEY:", OPENAI_KEY)
        
    def test_openai_api_call(self):
        print("Testing OpenAI API Call...")
        time.sleep(2)
        client = OpenAI(api_key=OPENAI_KEY)

        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": "Write a haiku about recursion in programming."
                }
            ]
        )

        print(completion.choices[0].message)
        

    def test_valid_model_name(self):
        """Test that valid model names do not raise an error."""
        print("Testing OpenAI Model instatiation...")
        try:
            llm = OpenAILLM(model_name="gpt-3.5-turbo")
            llm.invoke("This is a test")
        except ValueError as e:
            self.fail(f"Initialization raised ValueError unexpectedly: {e}")
            
    def test_invalid_model_name(self):
        """Test that invalid model names raise an error during invoke."""
        print("Testing OpenAI Model instatiation (invalid model)...")
        llm = OpenAILLM(model_name="invalid-model")
        with self.assertRaises(Exception) as context:  # Catch all exceptions to ensure it's handled
            llm.invoke("This is a test")
        self.assertIn("The model `invalid-model` does not exist", str(context.exception))

        
    def test_openai_llm_invoke(self):
        """Test the invoke method of OpenAILLM."""
        user_prompt = "Explain the importance of AI in healthcare."
        openai_llm = OpenAILLM(model_name="gpt-3.5-turbo")
        openai_llm.invoke(user_prompt)
        response = openai_llm.get_response()
        self.assertIsNotNone(response, "OpenAILLM invoke response should not be None")
        self.assertTrue(isinstance(response, str), "OpenAILLM invoke response should be a string")
        print(f"Prompt: {user_prompt}")
        print(f"Response: {response}\n")
        print("OpenAILLM invoke test successful\n")

if __name__ == '__main__':
    unittest.main()
    loader = unittest.TestLoader()
    
    # Define test order
    test_order = [
        'test_openai_config',
        'test_openai_api_call',
        'test_valid_model_name',
        'test_invalid_model_name'
        'test_openai_llm_invoke',
    ]

    # Run each test individually
    for test_name in test_order:
        suite = loader.loadTestsFromName(f"{TestOpenAI.__name__}.{test_name}")
        unittest.TextTestRunner(verbosity=2).run(suite)
