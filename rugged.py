import os
# import openai
# from utils.utils import get_project_root
from utils.logger import get_log_file, write_to_log  # Import the logger functions
# from neo4j_api.neo4j_api import Neo4j_API
# from openai_api.chat_test import single_chat as gpt_response
# from openai_api.openai_client import OpenAI_API

import sys
from rugged.knowledge_graph.query import Chat
from rugged.knowledge_graph.predict import ExplainablePredictions
from rugged.literature_retrieval.search import LiteratureSearch

# Example of query, predict, and search functions
def query(input_text, log_file=None):
    write_to_log(log_file, 'User Input: ' + input_text)
    chat = Chat(input_text)
    response = chat.conversation()
    return response


def predict(input_text, log_file=None):
    write_to_log(log_file, 'User Input: ' + input_text)
    chat = ExplainablePredictions(input_text)
    response = chat.conversation()
    return response


def search(input_text, log_file=None):
    write_to_log(log_file, 'User Input: ' + input_text)
    chat = LiteratureSearch(input_text)
    response = chat.conversation()
    return response


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


def main(log_dir='./log/'):
    log_file = get_log_file(log_dir)  # Initialize the log file
    print("Welcome to the RUGGED CLI! Type 'help' for instructions.")
    show_help()

    while True:
        user_input = input("> ").strip()
        command = user_input.lower()  # Convert the command to lowercase for case insensitivity
        response = None

        if command.startswith("query "):
            input_text = user_input[len("query "):].strip('"')
            response = query(input_text, log_file=log_file)
            print(response)

        elif command.startswith("predict "):
            input_text = user_input[len("predict "):].strip('"')
            response = predict(input_text, log_file=log_file)
            print(response)

        elif command.startswith("search "):
            input_text = user_input[len("search "):].strip('"')
            response = search(input_text, log_file=log_file)
            print(response)

        elif command == "help":
            show_help()

        elif command == "quit":
            print("Exiting program.")
            if log_file:
                write_to_log(log_file, "Program exited by user.")
            sys.exit()

        else:
            response = "Invalid command. Type 'help' for instructions."
            print(response)

        # Log the user input and the response
        if log_file:
            write_to_log(log_file, "User: " + user_input)
            if response:
                write_to_log(log_file, response)


if __name__ == "__main__":
    main()
