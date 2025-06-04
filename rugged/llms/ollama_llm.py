from .base_llm import BaseLLM
from langchain.chains import ConversationChain
from langchain_ollama import ChatOllama as ollama_llm
#from langchain_ollama.llms import OllamaLLM as ollama_llm
#from config import OLLAMA_URI, OLLAMA_MODEL
from config import OLLAMA_URI

class OllamaLLM(BaseLLM):
    def __init__(self):
        # Initialize Ollama model 
        OLLAMA_MODEL="Llama3" #TODO temp
        model = ollama_llm(model=OLLAMA_MODEL, base_url=OLLAMA_URI)
        self.llm = ConversationChain(llm=model)
        self.index = 0
        #TODO make a function to check if the specified model is downloaded, then download it if not.
    
    def invoke(self, prompt: str) -> str:
        self.index += 2
        return self.llm.invoke(prompt)
    
    def get_response(self):
        message = self.llm.memory.chat_memory.messages[self.index-1].content
        return message
