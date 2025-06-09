import os
import sys

# Add the root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from rugged.literature_retrieval.search import LiteratureSearch
import config

def main():
    # Default directories
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    input_directory = os.path.join(base_dir, 'output')
    log_file = os.path.join(base_dir, 'log', 'test_log.txt')
    corpus = os.path.join(base_dir, 'data', 'text_corpus', 'test_corpus.json')
    full_text_file = os.path.join(base_dir, 'data', 'text_corpus', 'test_full_text.json')

    question = 'Which documents are related to using beta-blockers to treat cardiovascular disease?'
    context = 'This is a test class, testing functionality of this part of the program.'
    try:
        ls = LiteratureSearch(question, context, log_file, corpus=corpus, full_text_file=full_text_file)
        response = ls.run()
        print(response)
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == '__main__':
    main()
