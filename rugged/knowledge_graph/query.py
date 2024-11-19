import sys
import re
import random
import json
import ijson
from openai import OpenAI
from langchain.chains import ConversationChain
from langchain_openai import ChatOpenAI

from config import OPENAI_KEY
from config import NODE_FEATURES_PATH
from config import BIAS_MITIGATION_PROMPT
from config import QUERY_PROMPT
from utils.neo4j_api.neo4j_driver import Driver
from utils.openai_api.named_entity_recognition import NamedEntityRecognition
from utils.openai_api.rag_system import RAG
from utils.neo4j_api.neo4j_utils import get_node_and_edge_types, read_query_examples
from utils.utils import extract_code
from utils.utils import extract_results, find_node_names


NODE_TYPES, EDGE_TYPES = get_node_and_edge_types()
QUERY_EXAMPLES = read_query_examples()

class QuerySystem:
    def __init__(self):

        # Save an NER object to generate a context for the initial query
        self.ner = NamedEntityRecognition()

        # Initialize query builder, query evaluator and reasoner
        self.query_builder = ConversationChain(llm=ChatOpenAI(model_name="gpt-4o", openai_api_key=OPENAI_KEY))
        self.query_evaluator = ConversationChain(llm=ChatOpenAI(model_name="gpt-4o", openai_api_key=OPENAI_KEY))
        self.reasoner = ConversationChain(llm=ChatOpenAI(model_name="gpt-4o", openai_api_key=OPENAI_KEY))

        # Add contexts to each LLM agent
        self.qb_index = self.init_query_builder(debug=False)
        self.qe_index = self.init_query_evaluator()
        self.r_index = 1

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

    
                
    def prepare_prompt(self, user_query, cypher_query, query_results, node_features):
        
        prompt = BIAS_MITIGATION_PROMPT + '\n' + QUERY_PROMPT.replace("[USER_QUERY]", user_query)
        prompt = prompt.replace("[QUERY_RESULTS]",query_results)
        prompt = prompt.replace("[NODE_FEATURES]", str(node_features))
        prompt = prompt.replace("[CYPHER_QUERY]", cypher_query)
        return prompt
    
    
#     def run(self):      
          #TODO simplify the function below!
#         neo4j_driver = Driver()
#         query_code, query_msg, query_msg_raw = neo4j_driver.check_query(cypher_query)
#         sampled_names = extract_results(query_msg)
#         reasoner_query = self.prepare_prompt(user_query, cypher_query, query_msg, sampled_names)
    
#         self.reasoner.invoke(reasoner_query)
#         message = self.reasoner.memory.chat_memory.messages[self.r_index].content
#         self.r_index += 2

#         print('REASONER MESSAGE')
#         print(message)
    
    def run(self, initial_question, max_num_results_returned=30):
        
        # Save the inital question for downstream purposes
        self.inital_question = initial_question
        
        # Define max_tries
        # Will be updated per iteration
        max_tries = 5

        # Save a driver instance for downstream query checking
        neo4j_driver = Driver()
        self.driver = neo4j_driver

        for i in range(9999):
            if i == 0:
                # Upon initally starting the conversation, ask the query builder to design a query
                initial_question, ner_context = self.generate_initial_query_context()

                # DEBUG 
                print("*****\nNAMED ENTITIES\n{}\n*****".format(ner_context))
                # DEBUG

                # Build an inital query
                self.query_builder.invoke(initial_question)
                generated_query = extract_code(self.query_builder.memory.chat_memory.messages[self.qb_index].content)
                self.qb_index += 2
                # Have the query evaluator modify the query
                self.query_evaluator.invoke(
                    'Make sure the following query is correct syntactically and makes sense to answer the question {}. Modify if needed and otherwise return the original query. Limit to 30 results.'.format(
                        generated_query))
                modified_query = extract_code(self.query_evaluator.memory.chat_memory.messages[self.qe_index].content)
                self.qe_index += 2
                # NOTE: THIS MAY BE BETTER ELSEWHERE
                # checker_results, checker_message = extract_results(modified_query)

                # DEBUG
                # This gave me an error for some random reason
                # print('QUERY BUILDER HISTORY AT INDEX {}'.format(self.qb_index))
                # print(self.query_builder.memory.chat_memory.messages[self.qb_index].content)
                # print('QUERY EVALUATOR HISTORY AT INDEX {}'.format(self.qe_index))
                # print(self.query_evaluator.memory.chat_memory.messages[self.qe_index].content)
                # DEBUG

            else:
                # In the else, either you are reprompting or you are still trying to 
                # get a query that can compile
                if query_code == -1:
                    self.query_builder.invoke(query_msg)
                else:
                    return
#                     updated_input = input('User Input: ')
#                     self.query_builder.invoke(updated_input)

                # In either case, you end up building a query and checking it
                generated_query = extract_code(self.query_builder.memory.chat_memory.messages[self.qb_index].content)
                self.qb_index += 2
                self.query_evaluator.invoke(
                    'Make sure the following query is correct syntactically and makes sense to answer the question {}. Modify if needed and otherwise return the original query.'.format(
                        generated_query))
                modified_query = extract_code(self.query_evaluator.memory.chat_memory.messages[self.qe_index].content)
                # NOTE: Place this elsewhere
                # checker_results, checker_message = extract_results(modified_query)
                self.qe_index += 2

                # DEBUG
                print('i: {}, self.qb_index: {}, self.qe_index: {}'.format(i, self.qb_index, self.qe_index))
                print('QUERY BUILDER HISTORY AT INDEX {}'.format(self.qb_index))
                print(self.query_builder.memory)
                print('QUERY EVALUATOR HISTORY AT INDEX {}'.format(self.qe_index))
                print(self.query_evaluator.memory)
                # DEBUG

            # After the query is checked, see if anything returns
            if modified_query != "":
                query_code, query_msg, query_msg_raw = neo4j_driver.check_query(modified_query)
            else:
                query_code, query_msg, query_msg_raw = neo4j_driver.check_query(generated_query)

            if i > max_tries:
                # If the max tries are exceeded, increment by i and then retype reasoner query
                max_tries += i
                reasoner_query = """
                Formulate a response based on your knowledge to this question: {}.
                I tried to query a biomedical knowledge graph with this query: {}.
                I could not get a query to work so I have no results.
                Try to reason through the question anyway.""".format(self.inital_question, modified_query)

                # DEBUG
                # print('MODIFIED QUERY')
                # print(modified_query)
                # print('CHECKER RESULTS')
                # print(checker_results)
                # DEBUG

                self.reasoner.invoke(reasoner_query)
                message = self.reasoner.memory.chat_memory.messages[self.r_index].content
                self.r_index += 2

                # DEBUG
                print('QUERY BUILDER HISTORY')
                print(self.query_builder.memory)
                print('QUERY EVALUATOR HISTORY')
                print(self.query_evaluator.memory)
                print('ORIGINAL GENERATED QUERY')
                print(generated_query)
                print('MODIFIED QUERY')
                print(modified_query)
                # DEBUG

                print('REASONER MESSAGE')
                print(message)

            elif query_code == -1 and i < max_tries:
                # If the query code is -1, then there was an error
                # Restart the loop and go back to the top
                continue
            else:
                # Consider the current question being asked
                if i == 0:
                    question = self.inital_question
                else:
                    # TODO sometimes ends up here, when updated_input is null
                    import pdb;pdb.set_trace()
                    if updated_input:
                        question = updated_input

                # If the query code is not -1, then you should be able to use the reasoner
                # Once the reasoner gives a response, reprompt
                if type(query_msg_raw) == list and len(query_msg_raw) > max_num_results_returned:
                    sampled_query_msg_raw = random.sample(query_msg_raw, max_num_results_returned)
                    sampled_names = extract_results(sampled_query_msg_raw, self.driver)
                    reasoner_query = """
                    Formulate a response based on your knowledge to this question: {}.
                    I queried a biomedical knowledge graph with this query: {}.
                    These are {} of the results (there may be more): {}
                    These are the names of 5 results: {}
                    Also cite the results of the query in your response.""".format(question, modified_query,
                                                                                   max_num_results_returned,
                                                                                   sampled_query_msg_raw, sampled_names)
                else:
                    sampled_names = extract_results(query_msg, self.driver)
                    reasoner_query = """
                    Formulate a response based on your knowledge to this question: {}.
                    I queried a biomedical knowledge graph with this query: {}.
                    These are the results: {}
                    These are the names of some of the results if there are any: {}
                    Also cite the results of the query in your response.""".format(question, modified_query, query_msg,
                                                                                   sampled_names)

                # DEBUG
                print('MODIFIED QUERY')
                print(modified_query)
                print('CHECKER RESULTS')
                print(sampled_names)
                # DEBUG

                self.reasoner.invoke(reasoner_query)
                message = self.reasoner.memory.chat_memory.messages[self.r_index].content
                self.r_index += 2

                # DEBUG
                print('QUERY BUILDER HISTORY')
                print(self.query_builder.memory)
                print('QUERY EVALUATOR HISTORY')
                print(self.query_evaluator.memory)
                print('ORIGINAL GENERATED QUERY')
                print(generated_query)
                print('MODIFIED QUERY')
                print(modified_query)
                # DEBUG

                print('REASONER MESSAGE')
                print(message)

                