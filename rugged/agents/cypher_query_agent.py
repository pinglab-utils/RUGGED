class CypherQueryAgent:
    def __init__(self, llm: BaseLLM):
        self.llm = llm