'''
1. Ask for user input 
2. Perform NER
    a. with NER results, see what hits the dictionary 
    b. if no exact match, re ask user for more context
3. Send the context to the llm and ask to generate a query
    - Here are the relevant terms in the graph: ___, ____, ___, ...
    - Now, make me a query to <original ask>
'''

'''
NOTE
Since the mesh ids may not be updated, I focused specifically on UniProt
'''

from openai_client_langchain import OpenAI_API
from named_entity_recognition import SpacyNER
from protein_utils import Protein_Utility

class Interface():
    def __init__(self):
        self.llm_client = OpenAI_API()
        # TODO: idk if these two belong here
        # Maybe keep these in another file/move them so their utility
        # is better delegated
        self.ner = SpacyNER()
        self.protein_utility = Protein_Utility()

    def process_ner_gene_products(self, ner_results: list):
        '''
        TODO: Might have to move this over to the protein utils class

        For the time being, only focus on processing proteins.
        Return a list of proteins that had findable aliases
        '''
        findable_protein_names = dict()
        not_findable_protein_names = set()

        for ner_result in ner_results:
            if ner_result[1] == 'GENE_OR_GENE_PRODUCT':
                # See if the protein name has a findable corresponding UniProt ID
                proteins = self.protein_utility.search_protein(ner_result[0])

                # Check if the protein name exists in the KG
                found_proteins = self.protein_utility.check_in_kg(proteins)

                # Document whether or not the protein is in the KG
                if found_proteins == []:
                    not_findable_protein_names.add(ner_result[0])
                else:
                    findable_protein_names[ner_result[0]] = found_proteins

        return findable_protein_names, not_findable_protein_names
    
    def process_ner_results(self, ner_results: list):
        '''
        For all NER results, check if there are any name matches in the nodes.
        syntrophin gamma 1
        '''
        for result in ner_results:
            ...

    def chat(self):
        '''
        Try and get as much info out of the user as possible.
        Once there is enough information, then use the llm 
        '''
        user_input = input("User: ")
        disease_ents, scientific_ents = self.ner.get_entities(text=user_input)
        print("PRE CLEAN UP")
        print("DISEASE ENTS:", disease_ents)
        print("SCIENTIFIC ENTS:", scientific_ents)
        self.ner.clean_ner_results(scientific_ents=scientific_ents, disease_ents=disease_ents)
        print('POST CLEAN UP')
        print("ALL ENTS:", disease_ents + scientific_ents)
        return
        # Process gene products
        findable_proteins, not_findable_proteins = self.process_ner_gene_products(scientific_ents)

        '''
        Everything below this comment maybe should belong elsewhere
        '''

        # Save a list of proteins to remove from the set of not_finable proteins
        # Remove after the loop below to avoid set size change runtime errors
        proteins_to_remove_from_nonfindable_proteins = list()
    
        # TODO: if a protein is not findable, see if there are any more specifics you can get out of it
        # TODO: how many times to reprompt?
        for non_findable_protein in not_findable_proteins:
            reprompt = input('No match found for protein \'{}\'. List any possible synomyms or aliases: '.format(non_findable_protein))
            # Villin: 'actin-binding protein'
            # TODO: Maybe make the "getting entities" process into a different function since some code repeats
            # TODO: have a way of only having to get sci ents since getting disease ents may waste time
            _, reprompt_sci_ents = self.ner.get_entities(text=reprompt)
            
            # Find possible matches in the KG for the other protein synonym
            found_reprompted_proteins = self.process_ner_gene_products(reprompt_sci_ents)

            # If there is a match, then you can remove the protein from the 'not found' set after iteration
            if found_reprompted_proteins:
                proteins_to_remove_from_nonfindable_proteins.append(non_findable_protein)
                # TODO: may be extra processing for if there is more than just 1 synonym
                # NOTE: There is a lot of additional type casting for the RHS, but that is just because
                # of the weird return nature of found_reprompted_proteins which is from 
                # self.process_ner_gene_products(.)
                findable_proteins[non_findable_protein] = list(found_reprompted_proteins[0].values())[0]

        for found_protein in proteins_to_remove_from_nonfindable_proteins:
            not_findable_proteins.remove(found_protein)
                
        # TODO: Send this data to the LLM and see if it can come up with a legit query
        print('FINDABLE', findable_proteins)
        print('NOT FINDABLE', not_findable_proteins)

        llm_context = """
        Someone is asking to write a query in cypher to \"{}\". 
        Here is additional context for the query containing the node names: {}.
        """.format(user_input, findable_proteins)

        self.llm_client.single_chat(init_query=llm_context)

def main():
    interface = Interface()
    interface.chat()

if __name__ == "__main__":
    main()
