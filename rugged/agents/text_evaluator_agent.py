from ..llms.base_llm import BaseLLM
from transformers import BartTokenizer, BartForConditionalGeneration

import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS

from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from langchain.embeddings import HuggingFaceEmbeddings

from config import BIAS_MITIGATION_PROMPT, LITERATURE_VALIDATION_PROMPT

class TextEvaluatorAgent:
    def __init__(self, llm: BaseLLM):
        self.llm = llm
    
        # Text summarizer
        summarizer_model_name = 'facebook/bart-large-cnn'
        self.summarizer_tokenizer = BartTokenizer.from_pretrained(summarizer_model_name)
        self.summarizer_model = BartForConditionalGeneration.from_pretrained(summarizer_model_name)

        # FAISS embedding model
        self.faiss_emb_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


    def summarize_documents(self, docs, LEN_CUTOFF=4000):
        # LEN_CUTOFF determines how to summarize the document
        summaries = []
        for i, doc in enumerate(docs):
            doc_text = str(doc)
            if len(doc_text) > LEN_CUTOFF:
                # Document is too long, hierarchically sumarize
                summary = self.summarize_long_document(doc_text)
            else:
                # Summarize the shorter document
                summary = self.summarize_doc(doc_text)
            summaries += [summary]
            print(f"Summarizing documents {i+1} of {len(docs)}", end='\r')
        print("\n")
        return summaries
    

    def summarize_doc(self, doc_text):
        ''' Summarize small documents with LLM agent '''
        prompt = (
            f"Summarize the below document in 200 words or fewer.\nDocument: {doc_text}"
        )
        response = self.llm.invoke(prompt)
        return response

    
    def summarize_long_document(self, text, chunk_size=500):
        ''' Hierarchically summarize with BART '''
        words = text.split()
        chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
        summaries = [self.summarize_section(chunk) for chunk in chunks]
        combined_summary = ' '.join(summaries)
        return combined_summary

    
    def summarize_section(self, text):
        ''' Summarize with BART '''
        inputs = self.summarizer_tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
        summary_ids = self.summarizer_model.generate(inputs, max_length=200, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
        summary = self.summarizer_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return self.clean_summary(summary)  # Clean each summary


    def clean_summary(self, text):
        # Remove any unwanted text patterns
        cleaned_text = text.replace("Summarize: ", "")
        cleaned_text = cleaned_text.replace("summarize: ", "")
        # Additional cleanup rules can be added here
        return cleaned_text


    def faiss_filter(self, user_query, docs, k):
        
        # Build FAISS index from documents
        vector_store = FAISS.from_documents(docs, embedding=faiss_emb_model)
        
        # Run similarity search
        top_docs = vector_store.similarity_search(user_query, k=k)

        return top_docs


    def parse_verifier_response(self, response):
        
        text = response['response']
        if "yes" in text.lower():
            return True
        return False
    

    def prepare_prompt(self, user_query, document):
        prompt = BIAS_MITIGATION_PROMPT + '\n' + LITERATURE_VALIDATION_PROMPT.replace("[USER_QUERY]", user_query)
        prompt = prompt.replace("[DOCUMENT]", str(document))
        return prompt
    

    def validate_literature(self, user_query, docs, MAX_PUBS=10):
        # Process documents to fit within context window
        summarized_docs = self.summarize_documents(docs)
        verified_docs = [] 

        for i, sd in enumerate(summarized_docs):
            print(f"Verifying document {i+1} of {len(docs)}", end='\r')
            
            # Evaluate relevance
            prompt = self.prepare_prompt(user_query, sd)
            response = self.llm.invoke(prompt)
            
            verified = self.parse_verifier_response(response)
            if verified:
                verified_docs += [sd]
        print("\n")

        # Use FAISS to identify MAX_PUBS documents to include
        if len(verified_docs) > MAX_PUBS:
            verified_docs = self.faiss_filter(user_query, verified_docs, MAX_PUBS)

        return verified_docs

    
    
