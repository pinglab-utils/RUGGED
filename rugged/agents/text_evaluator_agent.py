class TextEvaluatorAgent:
    def __init__(self, llm: BaseLLM):
        self.llm = llm
        
    #TODO add summarization functionality here

    def validate_literature(self, user_query: str, docs: list[dict]) -> list[dict]:
        prompt = (
            f"Are the following documents relevant to the user query? "
            f"Do not explain how they are relevant, only provide a list of PMIDs "
            f"which are NOT relevant (e.g., 'The following are not relevant to the user query: 123456, 098765'). "
            f"User query: {user_query}\nDocuments: {str(docs)}"
        )
        response = self.llm.invoke(prompt)
        irrelevant_pmids = [r.strip() for r in response.split(":")[1].split(",")]
        return [d for d in docs if d["PMID"] not in irrelevant_pmids]
