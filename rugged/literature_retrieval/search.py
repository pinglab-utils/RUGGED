import os
import pandas as pd
import re
import ast  
import json
import ijson
from fuzzywuzzy import fuzz
from tqdm import tqdm

from config import LITERATURE_RETRIEVAL_PROMPT
from config import LITERATURE_RETRIEVAL_EXAMPLE as EXAMPLE
from config import REASONING_AGENT, TEXT_EVALUATOR_AGENT

from rugged.agents import agent_loader
from utils.logger import write_to_log

class LiteratureSearch:

    def __init__(self, user_query, conversation_summary, log_file, corpus='./data/text_corpus/pubmed.json',
                 full_text_file = './data/pmid2fulltext_sections.json'):
        self.name = "Literature Search"
        
        self.user_query = user_query
        self.conversation_summary = conversation_summary
        
        self.corpus = corpus
       
        # LLM for evaluating the response
        self.text_evaluator_agent = agent_loader.load_text_evaluator_agent(TEXT_EVALUATOR_AGENT)

        # LLM for reasoning the response
        self.reasoning_agent = agent_loader.load_reasoning_agent(REASONING_AGENT)
       

    def retrieve_documents(self, kw1, kw2, MAX_PUBS=10, TIMEOUT=60mins):
        
        # Perform key-word based search TODO have it randomly search the corpus until max number found, change this to a while loop
        pmid_list = self.search_keywords_in_corpus(self.corpus, kw1, kw2)
        print(pmid_list)
        
        # Determine which are what kind of publication.
        original_research_contributions, clinical_case_reports = self.get_pmids(pmid_list, self.corpus)
        
        # Get full text of publications
        #TODO

        # Perform literature verification
        filtered_orc = self.original_research_contributions(original_research_contributions)
        filtered_ccr = self.original_research_contributions(clinical_case_reports)
        
        if len(filtered_orc) > MAX_PUBS:

            # Use FAISS to identify most relevant pubs
        # TODO FAISS crashes for some corpra, fix this!
        ## Perform FAISS search only on original research contributions
        #faiss_docs = self.faiss_search_in_corpus()
        #filtered_faiss = self.original_research_contributions(faiss_docs)
        
        ## Combine orc docs
        #orc_docs = filtered_faiss.update(filtered_orc)
        
        # Combine as text block
        #docs_to_return = f"Original Research Articles: {orc_docs}\n\nClinical Case Reports: {filtered_ccr}"
        docs_to_return = f"Original Research Articles: {filtered_orc}\n\nClinical Case Reports: {filtered_ccr}"
        
        return docs_to_return
        
        
    def get_pmids(self, pmids, pubmed_file):
        ''' Only return original research contribution and clinical case report documents '''
        orc_docs = []
        ccr_docs = []
        for json_object in self.read_large_json(pubmed_file):
            pmid = json_object.get('PMID', None)
            if pmid in pmids:
                if pmid['PublicationType'] == "Original Contribution":
                    orc_docs += [json_object]
                elif pmid['PublicationType'] == "ClinicalCaseReport":
                    ccr_docs += [json_object]
        return orc_docs, ccr_docs
    
    
    def read_large_json(self, file_path):
        with open(file_path, 'r') as file:
            for line in file:
                try:
                    json_obj = json.loads(line)
                    yield json_obj
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON: {e}")


    def fuzzy_keyword_in_text(self, keywords, text, threshold=80):
        if text is None:
            return False
        text = text.lower()
        for keyword, synonyms in keywords.items():
            all_terms = [keyword.lower()] + [synonym.lower() for synonym in synonyms]
            for term in all_terms:
                if fuzz.partial_ratio(term, text) >= threshold:
                    return True
        return False

    
    def search_keywords_in_corpus(self, file_path, keyword_set_1, keyword_set_2):
        # TODO change to ElasticSearch for better performance
        pmids = []
        line_count = 0

        for json_object in self.read_large_json(file_path):
            line_count += 1
            pmid = json_object.get('PMID', None)
            title = json_object.get('ArticleTitle', '')
            abstract = json_object.get('Abstract', '')

            text = (title or '') + ' ' + (abstract or '')
            
            # Skip if title and abstract are empty
            if not title and not abstract:
                continue
                
            # Fuzzy keyword search in case of mispelling or formatting difference
            if self.fuzzy_keyword_in_text(keyword_set_1, text) and self.fuzzy_keyword_in_text(keyword_set_2, text):
                pmids.append(pmid)
                print(pmid)
                print(json.dumps(json_object, indent=2))  # Print the matching json_object

            if line_count % 100 == 0:
                print(f"Processed {line_count} documents...", end='\r')

        return pmids


    def identify_keywords(self):
        # TODO replace w the one from Joseph
        pmid_list = ["34412508", "27939893", "34650309", "14663615", "23937302", "2463576"] 
        keyword_set_1 = {'metoprolol': ['Kapspargo', 'Lopressor', 'Lopressor Hct', 'Toprol']}
        keyword_set_2 = {
            'Arrhythmogenic Cardiomyopathy': ['Arrhythmogenic Right Ventricular Dysplasia', 
                                              'Arrhythmogenic Right Ventricular Cardiomyopathy-Dysplasia', 
                                              'Arrhythmogenic Right Ventricular Cardiomyopathy Dysplasia', 
                                              'ARVD-C', 'Arrhythmogenic Right Ventricular Cardiomyopathy', 
                                              'Ventricular Dysplasia, Right, Arrhythmogenic', 
                                              'Arrhythmogenic Right Ventricular Dysplasia-Cardiomyopathy', 
                                              'Right Ventricular Dysplasia, Arrhythmogenic']
        }
        return keyword_set_1, keyword_set_2
  

    def run(self):
        print("Performing Literature Retrieval...")
        
        print("Identifying keywords based on query...")
        kw1, kw2 = self.identify_keywords()
        print(kw1)
        print(kw2)
        #TODO Add user confirmation with timeout, to allow the user to correct keywords
              
        print("Retrieving documents...")
        self.documents = self.retrieve_documents(kw1, kw2)
        
        print("Preparing prompt...")
        prompt = reasoning_agent.prepare_search_prompt(self.conversation_summary, self.documents)
        print("Sending query to reasoning agent...")
        response = reasoning_agent.reason(prompt)
        
        return response<F2>
    
