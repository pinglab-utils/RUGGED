from ..llms.base_llm import BaseLLM
from config import PREDICTION_EXPLORER_PROMPT, BIAS_MITIGATION_PROMPT
from config import PREDICTION_EXPLORER_EXAMPLE, LITERATURE_RETRIEVAL_EXAMPLE
from config import QUERY_REASONING_PROMPT

class ReasoningAgent:
    def __init__(self, llm: BaseLLM):
        # LLM used for the agent
        self.llm = llm
        
    def prepare_predict_prompt(self, user_query, nodes, prediction_results):
        ''' This function prepares the predict prompt for the reasoning agent using a template query '''
        node1, node2 = nodes
        prompt = BIAS_MITIGATION_PROMPT + '\n' + PREDICTION_EXPLORER_PROMPT.replace("[USER_QUERY]", user_query)
        prompt = prompt.replace("[NODE1]",node1)
        prompt = prompt.replace("[NODE2]",node2)
        prompt = prompt.replace("[EXAMPLE]", PREDICTION_EXPLORER_EXAMPLE)
        prompt = prompt.replace("[PREDICTION_RESULTS]", prediction_results)
        return prompt
   
    
    def prepare_search_prompt(self, convo_summary, documents):
        prompt = LITERATURE_RETRIEVAL_PROMPT.replace("[USER_QUESTION]",self.user_query)
        prompt = prompt.replace("[SUMMARY_OF_CONVERSAION]", convo_summary)
        prompt = prompt.replace("[DOCUMENTS]", documents)
        prompt = prompt.replace("[EXAMPLE]", LITERATURE_RETRIEVAL_EXAMPLE)
        return prompt


    def prepare_query_prompt(self, user_query, cypher_query, query_results, node_features):
        prompt = BIAS_MITIGATION_PROMPT + '\n' + QUERY_REASONING_PROMPT.replace("[USER_QUERY]", user_query)
        prompt = prompt.replace("[QUERY_RESULTS]",query_results)
        prompt = prompt.replace("[NODE_FEATURES]", str(node_features))
        prompt = prompt.replace("[CYPHER_QUERY]", cypher_query)
        return prompt
    

    # Call the LLM to reason based on the prompt
    def reason(self, prompt):
        
        # Increment by one for invoke prompt
        self.llm.invoke(prompt)
        
        # Increment by one for response
        message = self.llm.get_response()
        
        return message
    
