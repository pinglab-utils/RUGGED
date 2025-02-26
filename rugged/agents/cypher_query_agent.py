from ..llms.base_llm import BaseLLM

from utils.neo4j_api.neo4j_utils import get_node_and_edge_types, read_query_examples
NODE_TYPES, EDGE_TYPES = get_node_and_edge_types()
QUERY_EXAMPLES = read_query_examples()

class CypherQueryAgent:
    def __init__(self, llm: BaseLLM):
        self.llm = llm

    def init_query_builder(self, debug=False):
        # The intial context is composed the nodes and edges
        self.query_builder.invoke(
            "What are the node types in the neo4j graph? List the name of the nodes exactly as it appears in quotes and seperate with commas.")
        self.query_builder.memory.chat_memory.messages[1].content = NODE_TYPES
        self.query_builder.invoke(
            "What are the relationships? List the name of the relationship exactly as it appears in quotes and seperate with commas.")
        self.query_builder.memory.chat_memory.messages[3].content = EDGE_TYPES

        # Acknowledge some of the rules that may be needed for formulating queries
        self.query_builder.invoke("What are some general rules for formulating queries?")
        self.query_builder.memory.chat_memory.messages[
            5].content = "All nodes have the \'name\' property. Node names are typically identifiers such as 'UniProt:P04004'. You must include a namespace identifier such as UniProt: before node names.\n" + QUERY_EXAMPLES

        if debug:
            print('PRINTING QUERY BUILDER MEMORY')
            print(self.query_builder.memory)
            print('Query Builder initialization completed.')

        return 7
    
    def generate_initial_query_context(self):
        # Generate inital query context
        # Also return the inital named entities
        context = self.ner.get_context(self.inital_question)
        print('GENERATING NAMED ENTITIES')
        print(context)
        print()
        example_node = "MATCH (p1:Entrez {name: 'Entrez:1756'})"
        llm_context = """
        Write a query in cypher to answer the question \"{}\". 
        Here is a dictionary where the key corresponds to a possible named entities in the question and the values correspond to node names in the graph: {}
        Here is additional context for the query containing the node names. 
        For example, this would mean if the context included the tuple ('dystrophin', 'Entrez:1756'), then {} would find the node. 
        Avoid using the CONTAINS keyword.
        """.format(self.inital_question, context, example_node)
        return llm_context, context