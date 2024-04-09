# NOTE: This file uses LangChain capabilites in conversing with the user
from openai import OpenAI
import sys
import os
from langchain.chains import ConversationChain
from langchain_openai import ChatOpenAI

sys.path.append('../')
from config import OPENAI_KEY
client = OpenAI(api_key=OPENAI_KEY)

NODES = ...
EDGES = ...

def get_log_file(directory):
    """
    Find the directory for the logs and return the relevant 
    file that needs to be written to.
    """
    try:
        # Create the output directory if it doesn't exist
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Find the next available log file
        log_file = None
        i = 0
        while True:
            log_file = os.path.join(directory, f"log_{i}.txt")
            if not os.path.exists(log_file):
                break
            i += 1

        return log_file
    except Exception as e:
        print(f"An error occurred: {str(e)}")

def write_to_log(log_file, text):
    """Using a log file, write text to it."""
    try:
        with open(log_file, 'a') as file:
            file.write(text + '\n')
    except Exception as e:
        print(f"An error occured: {str(e)}")

# TODO: Store in txt
NODE_TYPES = """
MeSH_Compound,['name']
Entrez,['name']
UniProt,['name']
Reactome_Reaction,['name']
MeSH_Tree_Disease,['name']
MeSH_Disease,['name']
Reactome_Pathway,['name']
MeSH_Anatomy,['name']
cellular_component,['name']
molecular_function,['name']
MeSH_Tree_Anatomy,['name']
ATC,['name']
DrugBank_Compound,['name']
KEGG_Pathway,['name']
biological_process,['name']
"""
# TODO: Store in txt
RELATIONSHIP_TYPES = """
"isa"
"`-increases->`"
"`-decreases->`"
"`-interacts_with->`"
"`-output->`"
"`-underexpressed_in->`"
"involved_in"
"`-overexpressed_in->`"
"located_in"
"`-may_participate_in->`"
"enables"
"-associated_with-"
"is_active_in"
"part_of"
"-diseases_share_variants-"
"decreases"
"`-transcription_factor_targets->`"
"-is-"
"`-pathway_is_parent_of->`"
"-ppi-"
"`-catalystActivity->`"
"increases"
"`-participates_in->`"
"`-involved_in->`"
"`-encodes->`"
"-diseases_share_genes-"
"`-input->`"
"`-negatively_regulates->`"
"`-entityFunctionalStatus->`"
"`-regulates->`"
"`-treats->`"
"`-associated_with->`"
"`-regulatedBy->`"
"colocalizes_with"
"`-positively_regulates->`"
"`-drug_participates_in->`"
"`-inhibits_downstream_inflammation_cascades->`"
"`-inhibitory_allosteric_modulator->`"
"acts_upstream_of"
"`-affects->`"
"NOT|involved_in"
"`-unknown->`"
"`-disease_involves->`"
"`-negative_modulator->`"
"-not_associated_with-"
"`-drug_participates_in_pathway->`"
"`-potentiator->`"
"`-suppressor->`"
"`-activator->`"
"-drug_uses_protein_as_enzymes-"
"NOT|located_in"
"`-drug_targets_protein->`"
"`-stimulator->`"
"`-inhibitor->`"
"acts_upstream_of_or_within"
"`-nucleotide_exchange_blocker->`"
"`-agonist->`"
"`-antagonist->`"
"`-partial_agonist->`"
"`-cofactor->`"
"-drug_uses_protein_as_transporters-"
"`-stabilization->`"
"`-binder->`"
"`-inducer->`"
"`-ligand->`"
"`-modulator->`"
"-drug_uses_protein_as_carriers-"
"`-component_of->`"
"`-chelator->`"
"`-regulator->`"
"`-chaperone->`"
"`-inactivator->`"
"`-neutralizer->`"
"`-cleavage->`"
"NOT|part_of"
"NOT|enables"
"`-multitarget->`"
"`-oxidizer->`"
"contributes_to"
"NOT|contributes_to"
"NOT|acts_upstream_of_or_within_negative_effect"
"acts_upstream_of_or_within_positive_effect"
"`-antisense_oligonucleotide->`"
"`-downregulator->`"
"NOT|colocalizes_with"
"`-inverse_agonist->`"
"`-partial_antagonist->`"
"`-translocation_inhibitor->`"
"`-blocker->`"
"acts_upstream_of_positive_effect"
"`-weak_inhibitor->`"
"`-antibody->`"
"`-inhibition_of_synthesis->`"
"`-binding->`"
"`-product_of->`"
"`-substrate->`"
"`-other/unknown->`"
"`-degradation->`"
"`-allosteric_modulator->`"
"`-positive_allosteric_modulator->`"
"NOT|acts_upstream_of_or_within"
"`-other->`"
"NOT|is_active_in"
"acts_upstream_of_or_within_negative_effect"
"acts_upstream_of_negative_effect"
"""

class OpenAI_API():

    def __init__(self):
        # TODO: Find a way to add context without having to send "useless" queries
        self.conversation = ConversationChain(llm=ChatOpenAI(model_name="gpt-4", openai_api_key=OPENAI_KEY))
        self.log_folder = os.path.join('../chat_log')
    
    def add_context(self, debug=True):
        log_file = get_log_file(self.log_folder)
        # Document the node types and the node properties
        self.conversation.invoke("What are the node types in the neo4j graph? List a node label, then a comma, then a list of node properties.") 
        # Change the contents
        self.conversation.memory.chat_memory.messages[1].content = NODE_TYPES
        # Document in logs
        write_to_log(log_file, "What are the node types in the neo4j graph? List a node label, then a comma, then a list of node properties.\n" + self.conversation.memory.chat_memory.messages[1].content)
        
        # Document the relationships 
        self.conversation.invoke("What are the relationships? List the name of the relationship exactly as it appears in quotes and seperate with new lines.") 
        # Change the contents
        self.conversation.memory.chat_memory.messages[3].content = RELATIONSHIP_TYPES
        # Document in logs
        write_to_log(log_file, "What are the relationships? List the name ofand seperate with new lines.\n" + self.conversation.memory.chat_memory.messages[3].content)
        
        if debug:
          print("SUCCESSFULLY ADDED CONTEXT...")
          print(self.conversation.memory)  

    
    def single_chat(self, summarize=False, init_query=''):
        # For a single chat, just write everything that occurs in a conversation in 1 log file
        log_file = get_log_file(self.log_folder)

        
        # Index at 5 since there were already previous queries to add context
        self.add_context(debug=False)
        conv_index = 5

        # Begin conversation
        for i in range(9999):
            if init_query != '' and i == 0:
                user_input = init_query
            else:
                # Collect user input
                user_input = input("User Input: ")

            self.conversation.invoke(user_input) 
            llm_response = self.conversation.memory.chat_memory.messages[conv_index].content
            conv_index += 2

            # Show response to the screen
            print(llm_response)

            # Save responses
            write_to_log(log_file, "User query: " + user_input)
            write_to_log(log_file, llm_response)
            write_to_log(log_file, "---------------------------------------------------------------------------")

if __name__ == "__main__":
    x = OpenAI_API()
    x.single_chat()

'''
Need to find a way to use the NER and the queries

1. User asks a query
2. Perform NER so that we can see if we have any matches
    a. Could do a linear search to see if an entity appears anywhere in the graph
    b. 
3. Use the NER to help with the context of query generation
'''



