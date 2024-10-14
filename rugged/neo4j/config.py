# config.py handles the configuration of the application

import json
import os
from utils.utils import get_project_root

# Project base directory
project_root = get_project_root()

# Relative path of configuration files
neo4j_config_path = os.path.join(project_root, 'config/neo4j_config.json')
openai_key_path = os.path.join(project_root, 'config/openai_key.txt')

# Load the neo4j_api configuration file
with open(neo4j_config_path, 'r') as config_file:
    config_data = json.load(config_file)

# Define variables for configuration values
NEO4J_URI = config_data.get('NEO4J_URI', 'bolt://localhost:7687')
NEO4J_USER = config_data.get('NEO4J_USER', 'neo4j')
NEO4J_PASSWORD = config_data.get('NEO4J_PASSWORD', 'password')

# Load the OpenAI configuration file
with open(openai_key_path, 'r') as config_file:
    # Make sure the file is not empty
    assert config_file.read(1), "OpenAI key file is empty"
    config_file.seek(0)  # Rewind
    OPENAI_KEY = config_file.read().strip()

MESH_ID_CONFIG = {
    "resolve_abbreviations": True,
    "linker_name": "mesh",
    "max_entities_per_mention": 1
}