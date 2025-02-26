from .base_llm import BaseLLM
from langchain.chains import ConversationChain
from langchain_ollama.llms import OllamaLLM
# from config import KOBOLDAI_URI

class OllamaLLM(BaseLLM):
    def __init__(self, model_name: str):
        # Initialize Kobold model 
        model = OllamaLLM(model=model_name)
        self.llm = ConversationChain(llm=model)
        self.index = 0
    
    def invoke(self, prompt: str) -> str:
        self.index += 2
        return self.llm.invoke(prompt)
    
    def get_response(self):
        message = self.llm.memory.chat_memory.messages[self.index-1].content
        return message