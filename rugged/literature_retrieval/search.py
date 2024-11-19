import os
import pandas as pd
import re
import ast  
import json
import ijson
from fuzzywuzzy import fuzz
from tqdm import tqdm

from config import OPENAI_KEY
from config import LITERATURE_RETRIEVAL_PROMPT
from config import LITERATURE_RETRIEVAL_EXAMPLE as EXAMPLE

from utils.logger import write_to_log
from openai import OpenAI
from langchain.chains import ConversationChain
from langchain_openai import ChatOpenAI
from config import OPENAI_KEY
from config import NODE_FEATURES_PATH

from transformers import BartTokenizer, BartForConditionalGeneration

class LiteratureSearch:

    def __init__(self, user_query, convo_summary, log_file, corpus='./data/text_corpus/pubmed.json', full_text_file = './data/pmid2fulltext_sections.json'):
        self.name = "Literature Search"
        
        self.user_query = user_query
        self.convo_summary = convo_summary
        
        self.corpus = corpus
        
        # LLM for reasoning and evaluating the response
        self.search_evaluator = ConversationChain(llm=ChatOpenAI(model_name="gpt-4o", openai_api_key=OPENAI_KEY))
        self.reasoner = ConversationChain(llm=ChatOpenAI(model_name="gpt-4o", openai_api_key=OPENAI_KEY))
        self.e_index = 0
        self.r_index = 0 
        
        # Text summarizer
        model_name = 'facebook/bart-large-cnn'
        self.tokenizer = BartTokenizer.from_pretrained(model_name)
        self.summarizer_model = BartForConditionalGeneration.from_pretrained(model_name)

        
    def retrieve_documents(self, kw1, kw2):
        
        # Perform key-word based search
        pmid_list=self.search_keywords_in_corpus(self.corpus, kw1, kw2)
        print(pmid_list)
        
        # Determine which are what kind of publication.
        original_research_contributions, clinical_case_reports = self.get_pmids(pmid_list, self.corpus)
        
        # Perform literature checker agent
        filtered_orc = self.original_research_contributions(original_research_contributions)
        filtered_ccr = self.original_research_contributions(clinical_case_reports)
        
        # Perform FAISS search only on original research contributions
        faiss_docs = self.faiss_search_in_corpus()
        filtered_faiss = self.original_research_contributions(faiss_docs)
        
        # Combine orc docs
        orc_docs = filtered_faiss.update(filtered_orc)
        
        # Combine as text block
        docs_to_return = f"Original Research Articles: {orc_docs}\n\nClinical Case Reports: {filtered_ccr}"
        
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
    
    def verify_literature(self, docs):
        literature_verification_prompt = f"Are the following documents relevant to the user query? Do not explain how they are relevant, only provide a list of PMIDs which are NOT relevant (e.g., 'The following are not relevant to the user query: 123456, 098765'). User query: {self.user_query}\nDocuments: {str(docs)}"
        self.search_evaluator.invoke(literature_verification_prompt)
        self.e_index += 2
        response = self.query_evaluator.memory.chat_memory.messages[self.qe_index].content
        relevant_pmids = [r.strip() for r in response.split(":")[1].split(",")]
        relevant_docs = []
        for d in docs:
            if d['PMID'] in relevant_pmids:
                relevant_docs += [d]
        return relevant_docs
        
        
    def prepare_prompt(self):
        prompt = LITERATURE_RETRIEVAL_PROMPT.replace("[USER_QUESTION]",self.user_query)
        prompt = prompt.replace("[SUMMARY_OF_CONVERSAION]",self.convo_summary)
        prompt = prompt.replace("[DOCUMENTS]",self.documents)
        prompt = prompt.replace("[EXAMPLE]",EXAMPLE)
        self.prompt = prompt
        
        
    def identify_keywords(self):
        #TODO extract keywords with entity matching from user question and conversation history
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
                
            if self.fuzzy_keyword_in_text(keyword_set_1, text) and self.fuzzy_keyword_in_text(keyword_set_2, text):
                pmids.append(pmid)
                print(pmid)
                print(json.dumps(json_object, indent=2))  # Print the matching json_object

            if line_count % 100 == 0:
                print(f"Processed {line_count} documents...", end='\r')

        return pmids
    
    def summarize_section(text):
        inputs = self.tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
        summary_ids = self.summarizer_model.generate(inputs, max_length=200, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return self.clean_summary(summary)  # Clean each summary


    def clean_summary(text):
        # Remove any unwanted text patterns
        cleaned_text = text.replace("Summarize: ", "")
        cleaned_text = cleaned_text.replace("summarize: ", "")
        # Additional cleanup rules can be added here
        return cleaned_text


    def summarize_long_document(text, chunk_size=500):
        words = text.split()
        chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
        summaries = [self.summarize_section(chunk) for chunk in chunks]
        combined_summary = ' '.join(summaries)
        return combined_summary
    
    
    def prepare_prompt(self):
        prompt = BIAS_MITIGATION_PROMPT + '\n' + LITERATURE_RETRIEVAL_PROMPT.replace("[USER_QUERY]", self.user_query)
        prompt = prompt.replace("[SUMMARY_OF_CONVERSAION]", self.convo_summary)
        prompt = prompt.replace("[DOCUMENTS]", str(self.documents))
        return prompt

                                    
    def reason(self, prompt):
        
        # Increment by one for invoke prompt
        self.r_index += 1
        self.reasoner.invoke(prompt)
        
        # Increment by one for response
        self.r_index += 1
        message = self.reasoner.memory.chat_memory.messages[self.r_index-1].content
        
        return message
        
        
    def run(self):
        print("Performing Literature Retrieval...")
        
        print("Identifying keywords based on query...")
        kw1, kw2 = self.identify_keywords()
        print(kw1)
        print(kw2)
        #TODO user confirmation with timeout?
              
        print("Retrieving documents...")
        self.documents = self.retrieve_documents(kw1, kw2)
        
        print("Preparing prompt...")
        prompt = self.prepare_prompt()
        print("Sending query to reasoning agent...")
        response = self.reason(prompt)
        
        return response
    