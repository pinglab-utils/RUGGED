from ..llms.base_llm import BaseLLM

class QueryVerificationAgent:
    def __init__(self, llm: BaseLLM):
        self.llm = llm
        
        
    def init_query_evaluator(self):
        # Also provide the query evaluator with the relevant context
        self.query_evaluator.invoke(
            "What are the node types in the neo4j graph? List the name of the nodes exactly as it appears in quotes and seperate with commas.")
        self.query_evaluator.memory.chat_memory.messages[1].content = NODE_TYPES
        self.query_evaluator.invoke(
            "What are the relationships? List the name of the relationship exactly as it appears in quotes and seperate with commas.")
        self.query_evaluator.memory.chat_memory.messages[3].content = EDGE_TYPES

        # Provide the query evaluator with examples
        self.query_evaluator.invoke("What are some examples of cypher queries for this graph?")
        self.query_evaluator.memory.chat_memory.messages[5].content = QUERY_EXAMPLES

        return 7
