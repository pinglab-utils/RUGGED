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
    "koboldai": os.path.join(project_root, 'config/koboldai_config.json')
}

# Function to load JSON configuration files
def load_json_config(file_path):
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Configuration file not found: {file_path}")
    except json.JSONDecodeError:
        print(f"Error decoding JSON in file: {file_path}")
    return {}

# Load configurations
neo4j_config = load_json_config(config_paths["neo4j"])
ollama_config = load_json_config(config_paths["ollama"])
koboldai_config = load_json_config(config_paths["koboldai"])

# Neo4j Configuration
NEO4J_URI = neo4j_config.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = neo4j_config.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = neo4j_config.get("NEO4J_PASSWORD", "password")

# Ollama and KoboldAI Configuration
OLLAMA_URI = ollama_config.get("OLLAMA_URI", None)
KOBOLDAI_URI = koboldai_config.get("KOBOLDAI_URI", None)
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
