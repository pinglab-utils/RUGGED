import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOpenAI


class RAG():
    def __init__(self):
        self.llm_model = 'gpt-3.5-turbo'
        self.emb_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    def query(self, named_entity: str):
        '''
        Return node names for a given named entity.
        '''
        vectorstore = FAISS.load_local("./data/knowledge_graph/kg_index", self.emb_model,
                                       allow_dangerous_deserialization=True)
        query = "Which nodes are related to the term: {}".format(named_entity)
        docs = vectorstore.similarity_search(query)

        return [doc.metadata['node_name'] for doc in docs][:3]


if __name__ == "__main__":
    rag = RAG()
    query = rag.query('Carbonic anhydrase')
    print(query)