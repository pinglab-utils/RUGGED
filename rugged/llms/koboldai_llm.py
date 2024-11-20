from .base_llm import BaseLLM
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain_community.llms import KoboldApiLLM
from config import KOBOLDAI_URI

class KoboldAILLM(BaseLLM):
    def __init__(self):
        # Initialize Kobold model 
        kb_llm = KoboldApiLLM(endpoint=KOBOLDAI_URI, max_length=80)
        self.llm = ConversationChain(llm=kb_llm)
        self.index = 0
    
    def invoke(self, prompt: str) -> str:
        self.index += 2
        return self.llm.invoke(prompt)
    
    def get_response(self):
        message = self.llm.memory.chat_memory.messages[self.index-1].content
        return message