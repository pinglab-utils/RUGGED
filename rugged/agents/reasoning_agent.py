from ..llms.base_llm import BaseLLM
from config import PREDICTION_EXPLORER_PROMPT, BIAS_MITIGATION_PROMPT
from config import PREDICTION_EXPLORER_EXAMPLE as EXAMPLE

class ReasoningAgent:
    def __init__(self, llm: BaseLLM):
        # LLM used for the agent
        self.llm = llm
        # Index for conversation chain TODO not sure we need to track this at the agent level?
        self.r_index = 0
        
    def prepare_predict_prompt(self, user_query, nodes, prediction_results):
        ''' This function prepares the predict prompt for the reasoning agent using a template query '''
        node1, node2 = nodes
        prompt = BIAS_MITIGATION_PROMPT + '\n' + PREDICTION_EXPLORER_PROMPT.replace("[USER_QUERY]", user_query)
        prompt = prompt.replace("[NODE1]",node1)
        prompt = prompt.replace("[NODE2]",node2)
        prompt = prompt.replace("[EXAMPLE]", EXAMPLE)
        prompt = prompt.replace("[PREDICTION_RESULTS]", prediction_results)
        return prompt
    
    # Call the LLM to reason based on the prompt
    def reason(self, prompt):
        
        # Increment by one for invoke prompt
        self.r_index += 1
        self.llm.invoke(prompt)
        
        # Increment by one for response
        self.r_index += 1
        message = self.llm.get_response()
        
        return message
    

                                    



    