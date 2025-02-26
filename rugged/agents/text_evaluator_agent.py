from ..llms.base_llm import BaseLLM

class TextEvaluatorAgent:
    def __init__(self, llm: BaseLLM):
        self.llm = llm
    
    def summarize_doc(self, doc_text):
        prompt = (
            f"Summarize the below document in 200 words or fewer.\nDocument: {doc_text}"
        )
        response = self.llm.invoke(prompt)
        return response
    
    def summarize_documents(self, docs):
        summaries = []
        for doc in docs:
            doc_text = str(doc)
            summary = summarize_doc(doc_text)
            summaries += [summary]
        return summaries
    
    
    def validate_literature(self, user_query, docs):
        summarized_docs = self.summarize_documents(docs)
        prompt = (
            f"Are the following documents relevant to the user query? "
            f"Do not explain how they are relevant, only provide a list of PMIDs "
            f"which are NOT relevant (e.g., 'The following are not relevant to the user query: 123456, 098765'). "
            f"User query: {user_query}\nDocuments: {str(summarized_docs)}"
        )
        response = self.llm.invoke(prompt)
        irrelevant_pmids = [r.strip() for r in response.split(":")[1].split(",")]
        return [d for d in docs if d["PMID"] not in irrelevant_pmids]
