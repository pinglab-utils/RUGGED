import os
import sys
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from utils.utils import get_project_root
from utils.logger import get_log_file, write_to_log  

from rugged.knowledge_graph.query import Chat
from rugged.predictive_analysis.predict import PredictionExplorer
from rugged.literature_retrieval.search import LiteratureSearch


class RuggedSystem:    
    def __init__(self, log_file=None, input_directory='./output', convo_file='./output/conversation_log.txt',
                         corpus='./data/text_corpus/pubmed.json'):
        """
        Initialize the RuggedSystem class with default or provided configurations.
        """
        self.log_file = log_file
        self.input_directory = input_directory
        self.corpus = corpus
        self.convo_file = convo_file
        
        # Initialize modules
        print("Loading QuerySystem")
        qs = QuerySystem()
        print("Loading PredictionExplorer")
        pe = PredictionExplorer(self.input_directory, self.log_file)
        print("Loading LiteratureSearch")
        ls = LiteratureSearch(self.log_file, corpus=self.corpus)
    
    
    def query(self, input_text):
        print("Querying Knowledge Graph...")
        print(f"User query: {input_text}")
        write_to_log(self.log_file, 'User Input: ' + input_text)
        self.qs.query(input_text)
        response = self.qs.run()
        self.write_to_summary(input_text, response)
        return response

    
    def predict(self, input_text):
        print("Predicting...")
        print(f"Input directory: {self.input_directory}")
        write_to_log(self.log_file, 'User Input: ' + self.input_directory)

        if pe.confirm_files():  
            # Proceed with prediction
            response = self.pe.run(input_text)
        else:
            # Prompt user to run predictive analysis ahead of time
            response = self.pe.deny()
        
        self.write_to_summary(input_text, response)
        return response

    
    def search(self, input_text):
        print("Searching literature...")
        write_to_log(self.log_file, 'User Input: ' + input_text)
        summary = self.summary_conversation()
        # Send summary to Literature earch
        response = self.ls.run(input_text, summary)
        self.write_to_summary(input_text, response)
        return response

    
    def write_to_summary(self, input_text, response):
        """ Write the conversation between User and RUGGED to a file """
        try:
            with open(convo_file, 'a') as file:
                file.write(f"User: {input_text}\n")
                file.write(f"RUGGED: {response}\n")
        except Exception as e:
            print(f"An error occurred: {str(e)}")

        
    def summary_conversation(self):
        """ Summarizes the conversation between User and RUGGED so far """
        try:
            with open(self.convo_file, 'r') as file:
                convo_text = file.read()  # Read the entire file into one long string
        except FileNotFoundError:
            print(f"Conversation file '{self.convo_file}' not found.")
            return ""  
        except Exception as e:
            print(f"An error occurred while reading the conversation file: {str(e)}")
            return ""  
        
        convo_summary = self.ls.summarize_long_document(convo_text)
        return convo_summary


def show_help():
    help_text = """
    Available commands:
      - query <input>: Run the query function with the specified input.
      - predict <input>: Run the predict function with the specified input.
      - search <input>: Run the search function with the specified input.
      - help: Show this help message.
      - quit: Exit the program.
    """
    print(help_text)
    

def get_user_input(prompt="> "):
    """ Prompts the user for input, supports multiline input. """
    user_input = ""
    while True:
        new_input = input(prompt)
        user_input += new_input.strip() + "\n"  
        quote_count = user_input.count('"')
        
        if quote_count % 2 == 0:  # Exit loop if quotes are balanced
            break
        else:
            prompt = ">> "  # Change prompt for multiline continuation

    command, *content = user_input.split(" ", 1)  
    content = content[0] if content else ""  
    return command.strip("\n"), content.strip('"')  


def main(log_dir='./log/'):
    # Initialize the log file
    log_file = get_log_file(log_dir)  
    
    # Instantiate RUGGED
    rs = RuggedSystem(log_file=log_file)
    
    print("Welcome to the RUGGED CLI! Type 'help' for instructions.")
    show_help()

    while True:

        command, content = get_user_input()  # Get command and content from encapsulated function

        response = None

        if command.lower() == "query":
            response = rs.query(content, log_file=log_file)
            print(response)

        elif command.lower() == "predict":
            response = rs.predict(content, log_file=log_file)
            print(response)

        elif command.lower() == "search":
            response = rs.search(content, log_file=log_file)
            print(response)

        elif command.lower() == "help":
            show_help()

        elif command.lower() == "quit":
            print("Exiting program.")
            if log_file:
                write_to_log(log_file, "Program exited by user.")
            sys.exit()

        else:
            response = "Invalid command. Type 'help' for instructions."
            print(response)

        # Log the user input and the response
        if log_file:
            write_to_log(log_file, "User: " + command + " " + content)
            if response:
                write_to_log(log_file, response)

                
if __name__ == "__main__":
    main()
