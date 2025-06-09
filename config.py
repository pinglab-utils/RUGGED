''' config.py handles the configuration of the application '''

import json
import os
from utils.utils import get_project_root

# Project base directory
project_root = get_project_root()

# Configuration file paths
config_paths = {
    "neo4j": os.path.join(project_root, 'config/neo4j_config.json'),
    "openai": os.path.join(project_root, 'config/openai_key.txt'),
    "ollama": os.path.join(project_root, 'config/ollama_config.json'),
    "koboldai": os.path.join(project_root, 'config/koboldai_config.json'),
    "huggingface": os.path.join(project_root, 'config/huggingface_config.json'),
    "agents": os.path.join(project_root, 'config/llm_agents.json'),
    "prompts": os.path.join(project_root, 'config/prompts.json')
}

# Function to load JSON configuration files
def load_json_config(file_path):
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
        # Handle path conversion to absolute paths
        for key, value in data.items():
            if 'PATH' in key and isinstance(value, str):
                data[key] = os.path.abspath(os.path.join(project_root, value))
        return data
    except FileNotFoundError:
        print(f"Configuration file not found: {file_path}")
    except json.JSONDecodeError:
        print(f"Error decoding JSON in file: {file_path}")
    return {}

# Load configurations
neo4j_config = load_json_config(config_paths["neo4j"])
ollama_config = load_json_config(config_paths["ollama"])
koboldai_config = load_json_config(config_paths["koboldai"])
huggingface_config = load_json_config(config_paths["huggingface"])
agents_config = load_json_config(config_paths["agents"])
prompts_config = load_json_config(config_paths["prompts"])

# Neo4j Configuration
NEO4J_URI = neo4j_config.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = neo4j_config.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = neo4j_config.get("NEO4J_PASSWORD", "password")

# Knowledge graph config files
NODE_RECORDS_PATH = neo4j_config.get("NODE_RECORDS_PATH",os.path.join(project_root, "data/knowledge_graph/node_records.json"))
NODE_FEATURES_PATH = neo4j_config.get("NODE_FEATURES_PATH",os.path.join(project_root, "data/knowledge_graph/node_features.json"))
NODE_TYPES_PATH = neo4j_config.get("NODE_TYPES_PATH",  os.path.join(project_root, "data/knowledge_graph/node_types.txt"))
EDGE_TYPES_PATH = neo4j_config.get("EDGE_TYPES_PATH", os.path.join(project_root, "data/knowledge_graph/edge_types.txt"))
QUERY_EXAMPLES = neo4j_config.get("QUERY_EXAMPLES", os.path.join(project_root, "data/query_examples.txt"))
KG_FAISS_INDEX = neo4j_config.get("KG_FAISS_INDEX", os.path.join(project_root, "data/knowledge_graph/kg_index"))

# LLM Agent config
REASONING_AGENT = agents_config.get("REASONING_AGENT","")
TEXT_EVALUATOR_AGENT = agents_config.get("TEXT_EVALUATOR_AGENT","")
CYPHER_QUERY_AGENT = agents_config.get("CYPHER_QUERY_AGENT","")
QUERY_VERIFICATION_AGENT  = agents_config.get("QUERY_VERIFICATION_AGENT","")

# Prompt tuning config
BIAS_MITIGATION_PROMPT = prompts_config.get("BIAS_MITIGATION_PROMPT","")
CYPHER_QUERY_FIRST_PROMPT_INSTRUCTIONS = prompts_config.get("CYPHER_QUERY_FIRST_PROMPT_INSTRUCTIONS","")
CYPHER_QUERY_REVISION_PROMPT_INSTRUCTIONS = prompts_config.get("CYPHER_QUERY_REVISION_PROMPT_INSTRUCTIONS","")
VERIFICATION_AGENT_PROMPT = prompts_config.get("VERIFICATION_AGENT_PROMPT","")
QUERY_REASONING_PROMPT = prompts_config.get("QUERY_REASONING_PROMPT","")
PREDICTION_EXPLORER_PROMPT = prompts_config.get("PREDICTION_EXPLORER_PROMPT","")
PREDICTION_EXPLORER_EXAMPLE = prompts_config.get("PREDICTION_EXPLORER_EXAMPLE","")
LITERATURE_VALIDATION_PROMPT = prompts_config.get("LITERATURE_VALIDATION_PROMPT","")
LITERATURE_RETRIEVAL_PROMPT = prompts_config.get("LITERATURE_RETRIEVAL_PROMPT","")
LITERATURE_RETRIEVAL_EXAMPLE = prompts_config.get("LITERATURE_RETRIEVAL_EXAMPLE","")

# Ollama and KoboldAI Configuration
OLLAMA_URI = ollama_config.get("OLLAMA_URI", None)
KOBOLDAI_URI = koboldai_config.get("KOBOLDAI_URI", None)

# Huggingface Configuration
HF_MODEL = huggingface_config.get("HF_MODEL", None)

# OpenAI Configuration
try:
    with open(config_paths["openai"], 'r') as openai_file:
        # Ensure the file is not empty
        if not openai_file.read(1):
            raise ValueError("OpenAI key file is empty")
        openai_file.seek(0)  # Rewind
        OPENAI_KEY = openai_file.read().strip()
except FileNotFoundError:
    raise FileNotFoundError("OpenAI key file not found")
except ValueError as e:
    raise ValueError(str(e))

# Mesh Configuration
MESH_ID_CONFIG = {
    "resolve_abbreviations": True,
    "linker_name": "mesh",
    "max_entities_per_mention": 1
}
