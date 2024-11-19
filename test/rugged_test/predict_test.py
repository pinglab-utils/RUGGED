import os, sys
# Add the root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import re
import random
from openai import OpenAI
from langchain.chains import ConversationChain
from langchain_openai import ChatOpenAI

# Save open AI key
from config import OPENAI_KEY
from utils.neo4j_api.neo4j_driver import Driver
from utils.openai_api.named_entity_recognition import NamedEntityRecognition
from utils.openai_api.rag_system import RAG
from utils.neo4j_api.neo4j_utils import get_node_and_edge_types, read_query_examples
from utils.utils import extract_code
from rugged.knowledge_graph.query import Chat

NODE_TYPES, EDGE_TYPES = get_node_and_edge_types()
QUERY_EXAMPLES = read_query_examples()


if __name__ == '__main__':
    question = 'What drugs are currently being prescribed to treat Arrhythmogenic Cardiomyopathy?'
    print('User Input: ' + question)
    chat = Chat(question)
    chat.conversation()
