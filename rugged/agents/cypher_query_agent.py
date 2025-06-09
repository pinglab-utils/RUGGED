import os

from ..llms.base_llm import BaseLLM
from utils.neo4j_api.neo4j_utils import get_node_and_edge_types, read_query_examples
NODE_TYPES, EDGE_TYPES = get_node_and_edge_types()
QUERY_EXAMPLES = read_query_examples()

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
#from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from utils.ner_utils.named_entity_recognition import NamedEntityRecognition

from config import KG_FAISS_INDEX, CYPHER_QUERY_FIRST_PROMPT_INSTRUCTIONS, CYPHER_QUERY_REVISION_PROMPT_INSTRUCTIONS

class CypherQueryAgent:
	def __init__(self, llm: BaseLLM):

		#memory = ConversationBufferMemory(return_messages=True)
		#self.llm = ConversationChain(llm=llm, memory=memory)
		self.llm = llm

		self.emb_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
		self.ner = NamedEntityRecognition()

		self.init_query_builder()
		self.user_query = ""
	
	def init_query_builder(self):
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
	

	def generate_query(self, user_query):

		# Perform NER to get node features for user query
		self.user_query = user_query
		cleaned_ents = self.ner.get_context(user_query)

		# Define context
		context = dict()
		for ent in cleaned_ents:
			context[ent] = self.match_entity_to_kg(ent)
	
		# Create instruction prompt for Cypher Query Agent
		example_node = "MATCH (p1:Entrez {name: 'Entrez:1756'})"
		initial_query_prompt = CYPHER_QUERY_FIRST_PROMPT_INSTRUCTIONS.format(
			QUESTION=self.user_query,
			CONTEXT=context,
			EXAMPLE_NODE=example_node
		)

		# Call the LLM to generate Cypher query
		self.llm.invoke(initial_query_prompt)
		message = self.llm.get_response()

		return message


	def revise_query(self, cypher_query, response):
		# Create instruction prompt for revising Cypher query
		revision_query_prompt = CYPHER_QUERY_REVISION_PROMPT_INSTRUCTIONS.format(
		  QUESTION=self.user_query,
		  ORIGINAL_QUERY=cypher_query,
		  VERIFIER_RESPONSE=response
		)

		# Call the LLM to generate revised Cypher query
		self.llm.invoke(revision_query_prompt)
		revised_query = self.llm.get_response()

		return revised_query


	def match_entity_to_kg(self, named_entity: str):
		'''
		Return node names for a given named entity.
		'''
		vectorstore = FAISS.load_local(KG_FAISS_INDEX, self.emb_model,
									   allow_dangerous_deserialization=True)
		query = "Which nodes are related to the term: {}".format(named_entity)
		docs = vectorstore.similarity_search(query)

		return [doc.metadata['node_name'] for doc in docs][:3]


