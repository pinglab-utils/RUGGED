from ..llms.base_llm import BaseLLM

from utils.neo4j_api.neo4j_utils import get_node_and_edge_types, read_query_examples
NODE_TYPES, EDGE_TYPES = get_node_and_edge_types()
QUERY_EXAMPLES = read_query_examples()
from config import VERIFICATION_AGENT_PROMPT

class QueryVerificationAgent:
	def __init__(self, llm: BaseLLM):
		self.llm = llm
		
		self.init_query_evaluator()
		
	def init_query_evaluator(self):
		# The intial context is composed the nodes and edges
		self.llm.invoke(
			"What are the node types in the neo4j graph? List the name of the nodes exactly as it appears in quotes and seperate with commas.")
		self.llm.llm.memory.chat_memory.messages[1].content = NODE_TYPES
		self.llm.invoke(
			"What are the relationships? List the name of the relationship exactly as it appears in quotes and seperate with commas.")
		self.llm.llm.memory.chat_memory.messages[3].content = EDGE_TYPES

		# Acknowledge some of the rules that may be needed for formulating queries
		self.llm.invoke("What are some general rules for formulating queries?")
		self.llm.llm.memory.chat_memory.messages[
			5].content = "All nodes have the \'name\' property. Node names are typically identifiers such as 'UniProt:P04004'. You must include a namespace identifier such as UniProt: before node names.\n" + QUERY_EXAMPLES

	def verify_query(self, user_query, cypher_code, query_results, node_features):
		prompt = VERIFICATION_AGENT_PROMPT.replace("[USER_QUERY]", user_query)
		prompt = prompt.replace("[CYPHER_QUERY]", cypher_code)
		prompt = prompt.replace("[QUERY_RESULTS]", query_results)
		prompt = prompt.replace("[NODE_FEATURES]", str(node_features))

		# Send prompt to Query Verification LLM Agent
		self.llm.invoke(prompt)
		response = self.llm.get_response() 
		is_pass = response.strip().lower().startswith("verification: pass")
		return is_pass, response
