from abc import ABC, abstractmethod

class BaseLLM(ABC):
    @abstractmethod
    def invoke(self, prompt: str) -> str:
        """ Invoke the LLM with a prompt. """
        pass
    
    @abstractmethod
    def get_response(self, prompt: str) -> str:
        """ Return the most recent response. """
        pass