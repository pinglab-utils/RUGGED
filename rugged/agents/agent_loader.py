from ..llms.openai_llm import OpenAILLM
from ..huggingface_llm.py import HuggingFaceLLM
from ..koboldai_llm import KoboldAILLM
from ..ollama_llm.py import OllamaLLM

def load_agent(model_name):
    ''' Load the appropriate LLM agent based on the model name. '''

    match model_name:
        case 'ollama':
            return OllamaLLM()
        case 'kobold':
            return KoboldAILLM()
        case 'huggingface':
            return HuggingFaceLLM()
        case _:
            # Load OpenAI model
            return OpenAILLM(model_name=model_name)
    