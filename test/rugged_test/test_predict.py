import os
import sys

# Add the root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from rugged.predictive_analysis.predict import PredictionExplorer
import config

def main():
    # Default directories
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    input_directory = os.path.join(base_dir, 'output')
    log_file = os.path.join(base_dir, 'log', 'test_log.txt')
    
    question = 'What drugs are currently being prescribed to treat Arrhythmogenic Cardiomyopathy?'
    
    try:
        pe = PredictionExplorer(input_directory, log_file)
        response = pe.run(question)
        print(response)
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == '__main__':
    main()
