from ..llms.base_llm import BaseLLM

class ReasoningAgent:
    def __init__(self, llm: BaseLLM):
        self.llm = llm

    def reason(self, prompt: str) -> str:
        return self.llm.invoke(prompt)

                                    



    