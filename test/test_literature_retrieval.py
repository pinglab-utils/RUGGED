import os, sys, argparse

# Add the root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from rugged.literature_retrieval.search import LiteratureSearch


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run a test query for literature retrieval.")
    parser.add_argument('--query', type=str, help='Literature query string')
    args = parser.parse_args()
    
    # Default directories
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
    log_file = os.path.join(base_dir, 'log', 'test_log.txt')
    corpus = os.path.join(base_dir, 'data', 'text_corpus', 'test_corpus.json')
    full_text_file = os.path.join(base_dir, 'data', 'text_corpus', 'test_full_text.json')
        
    lit_query = args.query or 'Which documents are related to using beta-blockers to treat cardiovascular disease?'
    try:
        # Initialize system
        ls = LiteratureSearch(lit_query, "", log_file, 
                              corpus=corpus, 
                              full_text_file=full_text_file, 
                              query_only=True)

        print("Testing query")
        print(lit_query)

        # Identify keywords from prompt
        keywords = ls.identify_keywords()
        print(keywords)

        # Retrieve results
        original_research_contributions, clinical_case_reports = ls.retrieve_documents(keywords)
        print(str(len(original_research_contributions))," original research contributions")
        print(str([p['PMID'] for p in original_research_contributions]))
        print(str(len(clinical_case_reports))," clinical case reports")
        print(str([p['PMID'] for p in clinical_case_reports]))
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == '__main__':
    main()

