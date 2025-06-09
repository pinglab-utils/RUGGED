'''
From the protein list, you need to search for the protein name and
the corresponding uniprot id. 

Once you get the name, you can return the corresponding uniprot id.


Maybe before all of this, make a file with the correct information.
Can check if there is HUMAN, RAT, MOUSE, PIG and then take everything before that as a possible protein.
Will likely be a linear search.
'''

import re
import json

PROTEIN_ALIAS_FILE_PATH = 'data/protein_aliases.txt'
GRAPH_DATA_FILE_PATH = 'data/node_records.json'

class Protein_Utility():
    def get_uniprot_ids(self, line: list) -> list:
        """
        Return the UniProt IDs for a given line.
        """
        for i, item in enumerate(line):
            if 'HUMAN' in item or 'RAT' in item or 'MOUSE' in item or 'PIG' in item:
                return i
    
    def search_protein(self, protein_name: str) -> list:
        """
        Return proteins that appear somewhere in the list.
        """
        relevant_proteins = list()
        with open(PROTEIN_ALIAS_FILE_PATH, 'r') as file:
            for i, line in enumerate(file):
                # Process each line by splitting
                processed_line = line.strip().split('|')
                # For all of the entries, check if the desired protein name exists somewhere in the synonyms
                for entry in processed_line:
                    if re.search(protein_name.lower(), entry.lower()) is not None:
                        relevant_proteins += processed_line[:self.get_uniprot_ids(line=processed_line)]
        
        # Either return a set of relevant 
        return set(relevant_proteins) if relevant_proteins != [] else None
    
    def check_in_kg(self, proteins: list) -> list:
        # Error checking
        # User can pass in an empty list or None to signify that there is no UniProt ID record
        if not proteins:
            return []

        # Save result of proteins
        proteins_in_kg = list()

        # Save graph data in function
        graph_data = json.loads(open(GRAPH_DATA_FILE_PATH, "r").read())
        
        # Save all node types
        node_types = ['ATC', 'MeSH_Disease', 'biological_process', 'molecular_function', 'MeSH_Compound', 
                      'DrugBank_Compound', 'MeSH_Anatomy', 'KEGG_Pathway', 'MeSH_Tree_Disease', 'Reactome_Reaction', 
                      'MeSH_Tree_Anatomy', 'Reactome_Pathway', 'cellular_component', 'Entrez', 'UniProt']
        
        # Loop through all keywords
        for protein in proteins:
            # Check all node types for each of the
            for node in node_types:
                # The keyword will have to have a specific prefix
                check_word = "{}:{}".format(node, protein)
                # Make note that the keyword was found 
                if check_word in graph_data:
                    proteins_in_kg.append(check_word)
        
        return proteins_in_kg

                    
if __name__ == "__main__":
    p = Protein_Utility()

    protein_list = ['Q86XR7', 'P25815', 'Q5TZ20', 'nugget','Q9H6I2']

    proteins = p.check_in_kg(protein_list)

    print(set(protein_list) - set(proteins))

                        






