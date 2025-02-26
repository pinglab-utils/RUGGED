from ..llms.openai_llm import OpenAILLM
from ..llms.huggingface_llm import HuggingFaceLLM
from ..llms.koboldai_llm import KoboldAILLM
from ..llms.ollama_llm import OllamaLLM

from .text_evaluator_agent import TextEvaluatorAgent
from .reasoning_agent import ReasoningAgent
from .query_verification_agent import QueryVerificationAgent
from .cypher_query_agent import CypherQueryAgent

def load_llm(model_name):
    '''Load the appropriate LLM based on the model name.'''
    if model_name == 'ollama':
        return OllamaLLM()
    elif model_name == 'kobold':
        return KoboldAILLM()
    elif model_name == 'huggingface':
        return HuggingFaceLLM()
    else:
        return OpenAILLM(model_name=model_name)

def load_text_evaluator_agent(model_name):
    '''Instantiate a Text Evaluator Agent'''
    llm = load_llm(model_name)
    return TextEvaluatorAgent(llm)

def load_reasoning_agent(model_name):
    '''Instantiate a Reasoning Agent'''
    llm = load_llm(model_name)
    return ReasoningAgent(llm)

def load_query_verification_agent(model_name):
    '''Instantiate a Query Verification Agent'''
    llm = load_llm(model_name)
    return QueryVerificationAgent(llm)

def load_cypher_query_agent(model_name):
    '''Instantiate a Cypher Query Agent'''
    llm = load_llm(model_name)
    return CypherQueryAgent(llm)
