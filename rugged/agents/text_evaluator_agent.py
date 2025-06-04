from ..llms.base_llm import BaseLLM
from transformers import BartTokenizer, BartForConditionalGeneration

class TextEvaluatorAgent:
    def __init__(self, llm: BaseLLM):
        self.llm = llm
    
        # Text summarizer
        summarizer_model_name = 'facebook/bart-large-cnn'
        self.summarizer_tokenizer = BartTokenizer.from_pretrained(summarizer_model_name)
        self.summarizer_model = BartForConditionalGeneration.from_pretrained(summarizer_model_name)


    def summarize_documents(self, docs):
        summaries = []
        for doc in docs:
            doc_text = str(doc)
            if len(doc_text) > 500:
                summary = summarize_long_document(doc_text)
            else:
                summary = summarize_doc(doc_text)
            summaries += [summary]
        return summaries
    

    def summarize_doc(self, doc_text):
        ''' Summarize small documents with LLM agent '''
        prompt = (
            f"Summarize the below document in 200 words or fewer.\nDocument: {doc_text}"
        )
        response = self.llm.invoke(prompt)
        return response

    
    def summarize_long_document(text, chunk_size=500):
        ''' Hierarchically summarize with BART '''
        words = text.split()
        chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
        summaries = [self.summarize_section(chunk) for chunk in chunks]
        combined_summary = ' '.join(summaries)
        return combined_summary

    
    def summarize_section(text):
        ''' Summarize with BART '''
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

    
    def prepare_prompt(self, user_query, documents):
        prompt = BIAS_MITIGATION_PROMPT + '\n' + LITERATURE_VALIDATION_PROMPT.replace("[USER_QUERY]", user_query)
        prompt = prompt.replace("[DOCUMENTS]", str(documents))
        return prompt
    

    def validate_literature(self, user_query, docs):
        # Process documents to fit within context window
        summarized_docs = self.summarize_documents(docs)
        
        # Evaluate relevance
        prompt = self.prepare_prompt(user_query, docs)
        response = self.llm.invoke(prompt)

        # Filter irrelevant documents
        irrelevant_pmids = [r.strip() for r in response.split(":")[1].split(",")]
        return [d for d in docs if d["PMID"] not in irrelevant_pmids]

    
    
