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
from utils.openai_api.named_entity_recognition import NamedEntityRecognition

class LiteratureSearch:

    def __init__(self, log_file, corpus='./data/text_corpus/pubmed.json',
                 full_text_file = './data/textpmid2fulltext_sections.json', query_only=False):
        self.name = "LiteratureSearch"
        
        self.ner = NamedEntityRecognition()
        
        self.corpus = corpus
        self.full_text_file = full_text_file
       
        # Skip LLM instatiation if using literature search query only
        if not query_only:
            # LLM for evaluating the response
            self.text_evaluator_agent = agent_loader.load_text_evaluator_agent(TEXT_EVALUATOR_AGENT)

            # LLM for reasoning the response
            self.reasoning_agent = agent_loader.load_reasoning_agent(REASONING_AGENT)
       

    def identify_keywords(self):
        keywords = self.ner.get_context(self.user_query)
        return keywords


    def get_pmids(self, pmids, pubmed_file):
        ''' Only return original research contribution and clinical case report documents '''
        orc_docs = []
        ccr_docs = []
        all_pmids = []
        missing_publication_type=False
        for json_object in self.read_large_json(pubmed_file):
            pmid = json_object.get('PMID', None)
            if pmid in pmids:
                if 'PublicationType' not in json_object.keys():
                    # Catch error if missing publication type
                    missing_publication_type = True
                elif json_object['PublicationType'] == "Original Contribution":
                    orc_docs += [json_object]
                elif json_object['PublicationType'] == "ClinicalCaseReport":
                    ccr_docs += [json_object]
                all_pmids += [json_object]
        if missing_publication_type and len(orc_docs) == 0 and len(ccr_docs) == 0:
            # Report the field is missing for all pmids and return full list
            print("WARNING: PublicationType field is missing. Full PMID list is returned")
            return all_pmids, all_pmids
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
        if type(keywords) is str:
            return fuzz.partial_ratio(keywords, text) >= threshold
        
        # Merge synonyms list
        text = text.lower()
        for keyword, synonyms in keywords.items():
            all_terms = [keyword.lower()] + [synonym.lower() for synonym in synonyms]
            for term in all_terms:
                if fuzz.partial_ratio(term, text) >= threshold:
                    return True
        return False

    
    def search_keywords_in_corpus(self, keywords, file_path):
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
            for kw in keywords:
                if self.fuzzy_keyword_in_text(kw, text):
                    pmids.append(pmid)
                    break

            if line_count % 100 == 0:
                print(f"Processed {line_count} documents...", end='\r')

        return pmids


    def retrieve_full_text(self, pmids):
        # TODO check if full text file is available

        # Extract PMIDs from the json files
        pmid_set = set([p['PMID'] for p in pmids])

        # Extract full text where available
        matched_full_text = {}
        with open(self.full_text_file, 'r', encoding='utf-8') as f1:
            # ijson.kvitems parses top-level key-value pairs
            for pmid_key, entry in ijson.kvitems(f1, ''):
                if pmid_key in pmid_set:
                    matched_full_text[pmid_key] = entry
        
        # Append full text to json object
        ret_pmids = []
        for p in pmids:
            pmid = p['PMID']
            if pmid in matched_full_text:
                p['full_text'] = matched_full_text[pmid]
            else:
                p['full_text'] = "Not available"
            ret_pmids += [p]
        return ret_pmids


    def retrieve_documents(self, keywords, TIMEOUT=3600): 
        # Default timeout is 60 mins
        '''
        TODO: search batches of the corpus, e.g. 1000 at a time, until TIMEOUT is reached.
        '''
        
        # Perform key-word based search 
        pmid_list = self.search_keywords_in_corpus(keywords, self.corpus)
        print(pmid_list)
        
        # Determine which are what kind of publication.
        original_research_contributions, clinical_case_reports = self.get_pmids(pmid_list, self.corpus)
        
        # Get full text of publications
        orc_full_text = self.retrieve_full_text(original_research_contributions) 
        ccr_full_text = self.retrieve_full_text(clinical_case_reports)

        return orc_full_text, ccr_full_text


    def run(self, user_query, conversation_summary):
        self.user_query = user_query
        self.conversation_summary = conversation_summary
        print("Performing Literature Retrieval...")
        
        print("Identifying keywords based on query...")
        keywords = self.identify_keywords()
        print(keywords)
        #TODO Add user confirmation with timeout, to allow the user to correct keywords
              
        print("Retrieving documents...")
        original_research_contributions, clinical_case_reports = self.retrieve_documents(keywords)
        
        print("Verifying documents...")
        filtered_orc = self.text_evaluator_agent.validate_literature(self.user_query, original_research_contributions)
        filtered_ccr = self.text_evaluator_agent.validate_literature(self.user_query, clinical_case_reports)
        documents = list(set(filtered_orc + filtered_ccr))
        
        print("Preparing prompt...")
        prompt = self.reasoning_agent.prepare_search_prompt(self.conversation_summary, documents)
        print("Sending query to reasoning agent...")
        response = self.reasoning_agent.reason(prompt)
        
        return response
    
