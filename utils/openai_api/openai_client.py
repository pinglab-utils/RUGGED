# NOTE: There has been an update to the Open AI API
from openai import OpenAI

import sys
import os
from transformers import pipeline

sys.path.append('../')
from config import OPENAI_KEY

NODES = ...
EDGES = ...
SUMMARIZER = pipeline('summarization', model="Falconsai/text_summarization")

def get_log_file(directory):
    """
    Find the directory for the logs and return the relevant 
    file that needs to be written to.
    """
    try:
        # Create the output directory if it doesn't exist
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Find the next available log file
        log_file = None
        i = 0
        while True:
            log_file = os.path.join(directory, f"log_{i}.txt")
            if not os.path.exists(log_file):
                break
            i += 1

        return log_file
    except Exception as e:
        print(f"An error occurred: {str(e)}")

def write_to_log(log_file, text):
    """Using a log file, write text to it."""
    try:
        with open(log_file, 'a') as file:
            file.write(text + '\n')
    except Exception as e:
        print(f"An error occured: {str(e)}")

class OpenAI_API():

    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_KEY)
        self.chat_model = None
        temp = self.get_chat_model()
        self.chat_model = temp
        # NOTE: Lang Chain may be useful 
        # TODO: Look at how this can manage context
        # TODO: Having something that the LLM can always access for the kg section if it is relevant
        # NOTE: Token limit may have been expanded in recent editions
        # https://python.langchain.com/docs/get_started/introduction
        self.messages = [{"role": "system", "content": "You are a helpful assistant."}, 
                         {"role": "user", "content": "Here are the node names in the graph: {}".format(NODES)},
                         {"role": "user", "content": "Here are the names of the edges in the graph: {}".format(EDGES)}]


    def get_chat_model(self):
        if self.chat_model:
            return self.chat_model
        messages = [{"role": "user", "content": "Hello"}]
        chat_completion = self.client.chat.completions.create(model="gpt-3.5-turbo", messages=messages)
        return chat_completion

    def call_openai_api(self, user_input):
        try:
            response = self.chat_model['choices'][0]['message']['content']

            return response
        except Exception as e:
            return str(e)
    

    def parse_message(self, chat_completion):
        """
        Parse message after the API has returned a call. 
        Return the contents of the message to be more readable.
        """
        message = chat_completion.choices[0].message

        role = message['role'].capitalize()
        content = message['content']

        return "%s: %s"%(role,content)
    
    def clear_context(self):
        """Use for restarting next instances"""
        self.messages = [{"role": "system", "content": "You are a helpful assistant."}]
    
    def add_context(self,context_message: str):
        """Add additional information to context"""
        self.messages.append({"role": "assistant", "content": context_message})
    
    def get_context(self):
        return self.messages
    
    def get_context_length(self):
        return len(self.messages)
    
    def single_chat(self, summarize=False):
        # For a single chat, just write everything that occurs in a conversation in 1 log file
        log_folder = os.path.join('../chat_log')
        log_file = get_log_file(log_folder)
        write_to_log(log_file, "SUMMARIZER IS {}".format(summarize))
        write_to_log(log_file, "---------------------------------------------------------------------------")
        added_message = ""

        # Continue a conversation indefinitely
        iter = 0
        while True:
            user_input = input("User: ")
            if not summarize:
                # If we are not summarizing, then we simply continue to add contexts
                self.messages.append({"role": "user", "content": user_input})
                added_message = user_input
            else:
                # Summarize the user inputs every time
                if iter == 0:
                    self.messages.append({"role": "user", "content": user_input})
                    added_message = user_input
                else:
                    text = self.messages[-2]["content"] + user_input
                    summary = SUMMARIZER(text, max_length=max(250, len(text.split())), min_length=min(250, len(text.split())), do_sample=False)
                    self.messages[-2]["content"] = summary[0]['summary_text']
                    added_message = summary[0]['summary_text']
                    assistant_response = self.messages[-1]["content"]

            # Complete the chat
            if iter == 0 or iter == 1:
                chat_completion = self.client.chat.completions.create(model="gpt-3.5-turbo", messages=self.messages)
            else:
                chat_completion = self.client.chat.completions.create(model="gpt-3.5-turbo", messages=self.messages[:-1])
            message = self.parse_message(chat_completion)

            # Add the response to the context
            if not summarize:
                self.messages.append({"role": "assistant", "content": message[11:]})
            else:
                if iter == 0 or iter == 1:
                    self.messages.append({"role": "assistant", "content": message[11:]})
                else:
                    text = assistant_response + message
                    summary = SUMMARIZER(text, max_length=max(250, len(text.split())), min_length=min(250, len(text.split())), do_sample=False)
                    self.messages[-1]["content"] = summary[0]['summary_text']

            # Write to log as well
            write_to_log(log_file, "User or summarized query: "+ added_message)
            write_to_log(log_file, message)
            write_to_log(log_file, "---------------------------------------------------------------------------")
            print(self.messages)
            iter += 1

    # def single_chat(self, summarize=False):
    #     # For a single chat, just write everything that occurs in a conversation in 1 log file
    #     log_folder = os.path.join('../chat_log')
    #     log_file = get_log_file(log_folder)
    #     write_to_log(log_file, "SUMMARIZER IS {}".format(summarize))
    #     write_to_log(log_file, "---------------------------------------------------------------------------")
    #     added_message = ""

    #     # Continue a conversation indefinitely
    #     iter = 0
    #     while True:
    #         user_input = input("User: ")
    #         if not summarize:
    #             # If we are not summarizing, then we simply continue to add contexts
    #             self.messages.append({"role": "user", "content": user_input})
    #             added_message = user_input
    #         else:
    #             # Summarize the user inputs every time
    #             if iter == 0 or iter == 1:
    #                 self.messages.append({"role": "user", "content": user_input})
    #                 added_message = user_input
    #             else:
    #                 text = self.messages[-2]["content"] + user_input
    #                 summary = SUMMARIZER(text, max_length=max(250, len(text.split())), min_length=min(250, len(text.split())), do_sample=False)
    #                 self.messages[-2]["content"] = summary[0]['summary_text']
    #                 added_message = summary[0]['summary_text']
    #                 assistant_response = self.messages[-1]["content"]

    #         # Complete the chat
    #         if iter == 0 or iter == 1:
    #             chat_completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=self.messages)
    #         else:
    #             chat_completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=self.messages[:-1])
    #         message = self.parse_message(chat_completion)

    #         # Add the response to the context
    #         if not summarize:
    #             self.messages.append({"role": "assistant", "content": message[11:]})
    #         else:
    #             if iter == 0 or iter == 1:
    #                 self.messages.append({"role": "assistant", "content": message[11:]})
    #             else:
    #                 text = assistant_response + message
    #                 summary = SUMMARIZER(text, max_length=max(250, len(text.split())), min_length=min(250, len(text.split())), do_sample=False)
    #                 self.messages[-1]["content"] = summary[0]['summary_text']

    #         # Write to log as well
    #         write_to_log(log_file, "User or summarized query: "+ added_message)
    #         write_to_log(log_file, message)
    #         write_to_log(log_file, "---------------------------------------------------------------------------")
    #         print(self.messages)
    #         iter += 1

#if __name__ == "__main__":
#    x = OpenAI_API()
#    x.single_chat(summarize=True)

