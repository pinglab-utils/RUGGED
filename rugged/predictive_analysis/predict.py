import os
import pandas as pd
import re
import ast 
import json
import ijson
import re

from utils.logger import write_to_log
from openai import OpenAI
from langchain.chains import ConversationChain
from langchain_openai import ChatOpenAI
from rugged.agents import agent_loader

from utils.utils import format_node_name
from utils.neo4j_api.neo4j_utils import get_node_features, find_node_names # TODO remove find_node_names from general util class!!


from config import REASONING_AGENT

class PredictionExplorer:
    def __init__(self, input_directory, log_file):
        # Log file for writing program progress
        self.log_file = log_file

        # LLM for reasoning the response
        self.reasoning_agent = agent_loader.load_reasoning_agent(REASONING_AGENT)
        
        # Precomputed prediction files
        self.input_directory = input_directory
        self.load_prediction_files()
        
        
    def load_prediction_files(self):
        """Search for prediction files (PDF, TSV, and log) in the output directory and parse."""
        parsed_data = {}
        predictions = None
        log_file = None

        # Iterate through the files in the directory
        for root, _, files in os.walk(self.input_directory):
            for file in files:
                basename, ext = os.path.splitext(file)  # Split the file into base name and extension

                # Ensure the basename is in the dictionary
                if basename not in parsed_data:
                    parsed_data[basename] = {}

                # Check if the file is a PDF or TSV and store accordingly
                if ext == ".pdf":
                    pdf_path = os.path.join(root, file)
                    parsed_data[basename]['pdf'] = pdf_path  # Store PDF path

                elif ext == ".tsv":
                    tsv_path = os.path.join(root, file)
                    tsv_data = pd.read_csv(tsv_path, sep='\t')
                    parsed_data[basename]['tsv'] = tsv_data  # Store parsed TSV data

                    if "predictions" in basename:  # If the file is predictions.tsv
                        predictions = tsv_data  # Store it separately as predictions

                elif ext == ".log":
                    log_path = os.path.join(root, file)
                    with open(log_path, 'r') as log_file_handle:
                        prediction_log_file = [l.strip("\n") for l in log_file_handle.readlines()]  # Read the log file contents

        # Filter out basenames without both PDF and TSV
        explainable_prediction_files = {basename: data for basename, data in parsed_data.items() if 'pdf' in data and 'tsv' in data}

        # Match predictions with explainable prediction files
        matched_results = self.match_predictions(predictions, explainable_prediction_files)
        
        self.explainable_prediction_files = explainable_prediction_files
        self.predictions = predictions
        self.prediction_log_file = prediction_log_file
        self.matched_results = matched_results

        
    def match_predictions(self, predictions, explainable_prediction_files):
        """Match predictions with explainable prediction files and node features."""
        result = {}

        for basename, details in explainable_prediction_files.items():
            # Use a regular expression to capture the compound and disease parts
            match = re.match(r"(DrugBank_Compound:[^_]+)_MeSH_Disease:([^_]+)_", basename)

            if match:
                compound = match.group(1).strip()  # Strip any whitespace
                disease = f"MeSH_Disease:{match.group(2).strip()}"  # Reformat the disease part and strip whitespace
                parsed_tuple = (compound, disease)
                
                # Iterate over predictions
                for e, p in zip(predictions['edge'], predictions['probability']):
                    # Convert the edge from string to tuple using ast.literal_eval
                    try:
                        e_tuple = ast.literal_eval(e)  # Convert string representation of tuple to actual tuple
                        e_normalized = (e_tuple[0].strip(), e_tuple[1].strip())  # Normalize the tuple for comparison
                    except (SyntaxError, ValueError):
                        print(f"Skipping invalid edge format: {e}")
                        continue
                    
                    if e_normalized == parsed_tuple:
                        # Retrieve node features for the matched nodes
                        node1, node2 = parsed_tuple
                        node1_features = get_node_features(node1) 
                        node2_features = get_node_features(node2)

                        # Map the matched tuple to TSV, PDF, probability, and node features
                        result[parsed_tuple] = {
                            'tsv': details['tsv'],          # TSV data from explainable_prediction_files
                            'pdf': details['pdf'],          # PDF path from explainable_prediction_files
                            'probability': p,               # Probability value from predictions
                            'node1_features': node1_features, # Features of node1
                            'node2_features': node2_features  # Features of node2
                        }
        return result
    
    
    def prompt_user_for_confirmation(self):
        while True:
            choice = input("\nAre these the correct input files? (y/n): ").strip().lower()
            if choice in ['y', 'yes']:
                return True
            elif choice in ['n', 'no']:
                return False
            else:
                print("Invalid input. Please enter 'y' for yes, or 'n' for no.")
               
        
    def get_input_directory(self):
        while True:
            directory = input("\nPlease enter the new input directory path (or type 'q' to quit): ").strip()

            if directory.lower() == 'q':
                return None
            elif os.path.isdir(directory):
                return directory
            else:
                print("Invalid directory. Please enter a valid directory path or type 'q' to quit.")

                
    def confirm_files(self):
        while True:
            # Print details of the prediction files
            self.print_prediction_files()

            # Prompt the user to select yes or no
            confirmed_files = self.prompt_user_for_confirmation()

            if confirmed_files:
                break
            else:
                # Prompt user for a new input directory
                input_directory = self.get_input_directory()

                # Check if the user chose to quit
                if input_directory is None:
                    return False

                # Load new prediction files if a directory was provided
                self.input_directory = input_directory
                self.locate_prediction_files()

        return True

            
    def format_prediction_details(self, nodes, selected_prediction):
        """Format selected prediction details into a string for LLM input."""

        # Nodes
        node1, node2 = nodes
        
        # Probability
        prediction_details = f"Prediction between '{node1}' and '{node2}' with a probability of {selected_prediction['probability']:.2%}:\n"

        # Edge Importance Data (TSV)
        prediction_details += "\nEdge Importance Data:\n"
        prediction_details += selected_prediction['tsv'].to_string(index=False) + "\n\n"

        # Node 1 Features
        prediction_details += f"Node Features for {node1}:\n"
        for key, value in selected_prediction['node1_features'].items():
            prediction_details += f"  {key}: {value}\n"

        # Node 2 Features
        prediction_details += f"Node Features for {node2}:\n"
        for key, value in selected_prediction['node2_features'].items():
            prediction_details += f"  {key}: {value}\n"

        # format the node ids to names
        formatted_text = prediction_details
        # Set up the regex
        node_names = ["MeSH_Compound", "Entrez", "UniProt", "Reactome_Reaction", "MeSH_Tree_Disease", "MeSH_Disease",
                      "Reactome_Pathway", "MeSH_Anatomy", "cellular_component", "molecular_function", "MeSH_Tree_Anatomy",
                      "ATC", "DrugBank_Compound", "KEGG_Pathway", "biological_process"]
        node_finder_pattern = r'(\b(?:' + '|'.join(map(re.escape, node_names)) + r')\S*)'
        matches = re.compile(node_finder_pattern).findall(prediction_details)

        # Clean up results
        replace_chars = ['{', '}', '\'', '\"', '[', ']', ',', '(', ')']
        for i, match in enumerate(matches):
            for char in replace_chars:
                match = match.replace(char, "")
            if match[-1] == ":":
                match = match[:-1]
            matches[i] = match
        matches = list(set(matches))
        names_matches = find_node_names(max_nodes_to_return=100, returned_nodes=matches)
        for m in matches:
            formatted_m = self.get_node_name(m,names_matches)
            if formatted_m[0] != "No known names":
                formatted_text = formatted_text.replace(m,formatted_m[0])
        return formatted_text

        
    def print_prediction_files(self):
        
        # Display the results for verification
        print("Explainable Prediction Files:")
        for basename, details in self.explainable_prediction_files.items():
            print(f"Base name: {basename}")
            print(f"PDF Path: {details['pdf']}")
            print(f"TSV Data:\n{details['tsv']}\n")

        # Display predictions data
        print("\nPredictions Data (from predictions.tsv):")
        if self.predictions is not None:
            print(self.predictions.shape)
            print(self.predictions)

        # Display log file content
        print("\nLog File Content (from output.log):")
        if self.prediction_log_file is not None:
            print(f"{len(self.prediction_log_file)} lines")

        # Display matched results
        for key, value in self.matched_results.items():
            print(f"Matched tuple: {key}")
            print(f"Probability: {value['probability']}\n")
            print(f"TSV Data:\n{value['tsv']}")
            print(f"PDF Path: {value['pdf']}")

            # Display node features
            node1, node2 = key
            print(f"Node 1 ({node1}) Features: {value['node1_features']}")
            print(f"Node 2 ({node2}) Features: {value['node2_features']}\n")
            
    def prompt_user_for_prediction(self, matched_results):
        """Prompt the user to select a prediction to explore."""
        
        # Map to node names
        nodes = []
        for n1,n2 in matched_results.keys():
            nodes.append(n1)
            nodes.append(n2)
        nodes = list(set(nodes))
        node_to_node_features = find_node_names(returned_nodes=nodes)
        
        #TODO sort the prediction by probability first
        
        # Explain what was found
        print("\nWe have found the following explainable predictions based on the provided data:")
        for i, (key, value) in enumerate(matched_results.items(), 1):
            node1, node2 = key
            node1_name = self.get_node_name(node1, node_to_node_features)
            node2_name = self.get_node_name(node2, node_to_node_features)
            print(f"{i}) Prediction between '{node1_name}' and '{node2_name}' with a probability of {value['probability']:.2%}")
        
        # Ask the user which prediction to explore
        while True:
            try:
                choice = int(input("\nEnter the number of the prediction you'd like to explore (or 0 to exit): "))
                if choice == 0:
                    print("Exiting...")
                    return None
                elif 1 <= choice <= len(matched_results):
                    # Get user choice
                    selected_key = list(matched_results.keys())[choice - 1]
                    # Format with the node names
                    node1,node2 = selected_key
                    node1_name = self.get_node_name(node1, node_to_node_features)
                    node2_name = self.get_node_name(node2, node_to_node_features)
                    formatted_nodes = node1_name, node2_name
                    return selected_key, matched_results[selected_key]
                else:
                    print(f"Please enter a number between 1 and {len(matched_results)}.")
            except ValueError:
                print("Invalid input. Please enter a number.")
                
    def get_node_name(self, node_id, node_to_node_features):
        """ Retrieve the node's name from features or use the node_id if not found. """
        assert isinstance(node_to_node_features, dict), "node_to_node_features must be a dictionary"
        
        if node_id in node_to_node_features:
            temp = node_to_node_features[node_id]
            if isinstance(temp, list):
                return temp  
            elif 'names' in temp:
                return temp['names']  
            else:
                return [node_id] 
        else:
            return [node_id]


    def run(self, user_query):
        """Main function to execute the workflow."""
        # Prompt the user to select a prediction
        formatted_nodes, selected_prediction = self.prompt_user_for_prediction(self.matched_results)
        
        # Prepare the prompt payload
        print("Processing input files...")
        supporting_information = self.format_prediction_details(formatted_nodes, selected_prediction)
        prompt = self.reasoning_agent.prepare_predict_prompt(user_query, formatted_nodes, supporting_information)
        
        # Send to Reasoner LLM Agent
        print("Sending query to reasoning agent...")
        response = self.reasoning_agent.reason(prompt)
                    
        return response

        
    def deny(self):
        print("Please run prediction first!!")


