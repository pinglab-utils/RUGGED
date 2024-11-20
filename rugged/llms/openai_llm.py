from .base_llm import BaseLLM
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from config import OPENAI_KEY

class OpenAILLM(BaseLLM):
    def __init__(self, model_name: str):
        self.llm = ConversationChain(llm=ChatOpenAI(model_name=model_name, openai_api_key=OPENAI_KEY))
        self.index = 0
    
    def invoke(self, prompt: str) -> str:
        self.index += 2
        return self.llm.invoke(prompt)
    
    def get_response(self):
        message = self.llm.memory.chat_memory.messages[self.index-1].content
        return message