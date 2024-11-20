# Config
The `config` directory contains configuration files required to set up and customize the RUGGED system. These files define settings for large language models (LLMs), the Neo4j graph database, and prompt configurations. Below is an overview of each file and its purpose.

1. **prompts.json** (Required) Contains the prompts used to instruct the LLM agents. Modify these prompts directly to fine-tune their behavior.
2. **llm_agents.json** (Required) Specifies the LLM agents used by the software and the corresponding model names. If using OpenAI, specify the model name directly (e.g., "gpt-4o"). If using Ollama, configure the model on startup and use "ollama" as the model name in this file. If using Kobold, configure the model on startup and use "kobold" as the model name in this file.
3. **neo4j_config.json** (Required) Stores login credentials for the Neo4j graph database and specifies the source files for the knowledge graph.
4. **openai_key.txt** (Optional) The API key used to access the OpenAI API. 
5. **ollama_config.json** (Optional) Contains configuration for the Ollama LLM service, specifying the IP address.
6. **koboldai_config.json** (Optional) Contains configuration for the KoboldAI LLM service, specifying the IP address.
