# (Retrieval Under Graph-Guided Explainable disease Distinction (RUGGED)
This repository contains the code associated with the in-progress publication titled "Evidence-based Knowledge Synthesis and Hypothesis Validation: Navigating Biomedical Knowledge Bases via Explainable AI and Agentic Systems". This computational workflow integrates Large Language Model (LLM) inference with Retrieval Augmented Generation (RAG) drawing evidence from trustworthy and curated biomedical knowledge bases as well as peer reviewed biomedical text publications. This approach streamlines the identification of explainable and actionable predictions, synthesizing new knowledge from up-to-date information, to pinpoint promising directions for hypothesis-driven investigations.

Please visit the arxiv for more information: [arxiv](https://arxiv.org/abs/2407.12888)

![workflow](assets/workflow.png)
**Figure 1. Retrieval Under Graph-Guided Explainable disease Distinction (RUGGED) Workflow.** RUGGED is comprised of four main components: 1) downloading and processing data from ethically sourced and professionally managed resources (e.g., PubMed and curated biomedical knowledgebases), 2) extracting information from peer-reviewed reports and integrating them into a knowledge graph, 3) preparing the text and graph information within database services, 4) identifying explainable predictions regarding the connections among biomedical entities within the knowledge graph, and 5) a Retrieval Augmented Generation (RAG) workflow (detailed in Figure 2.) enabling knowledge synthesis via applications of LLM with validated complex molecular relationships with AI-supported disease predictions to achieve enhanced exploration of biomedical information. A human-in-the-loop review step can be conducted by the user to enhance accuracy of the output.

![workflow](assets/retrieval.png)
**Figure 2. Retrieval Architecture and Workflow.** The Retrieval Augmented Generation (RAG) framework employs multiple LLM Agents, each executing specific tasks to support access to relevant information based on the user query. This system provides documented evidence for the user-facing GPT-based Reasoning Agent, facilitating user-agent interaction and synthesis of knowledge. 1) Biomedical Text Retrieval: Peer-reviewed original contributions and review articles are filtered based on their relevance to understand disease associations. A vector database is constructed for author and editor validated text evidence weighted based on the corresponding614
section of the publication, respectively: 70% Abstract, 10% Results, 10% Metadata, 10% for all other subsections. A key-word search and similarity search against the text embedding of the user query, together identify relevant documents. Summaries of each document are generated using a BERT-based summarizer, with the GPT-based Text Evaluator Agent refining the search to validate query-document relevance. 2) Knowledge Graph Retrieval: A BERT-based named entity recognition and GPT-based relation extraction module connects the user query to relevant entities in the knowledge graph. A similarity search in a vector database identifies pertinent nodes and edges. Data is retrieved from the Neo4j database via Cypher queries generated by the GPT-based Cypher Query Agent and refined by the Query Verification Agent.

Docker: https://hub.docker.com/r/arpelletier/jove_llm

