from transformers import pipeline


class BioBERT_NER():
    def __init__(self):
        self.disease_pipe = pipeline("token-classification", model="alvaroalon2/biobert_diseases_ner")
        self.chemical_pipe = pipeline("token-classification", model="alvaroalon2/biobert_chemical_ner")
        self.genetic_pipe = pipeline("token-classification", model="alvaroalon2/biobert_genetic_ner")

    def clean_pipeline_results(self, pipeline_results: list):
        '''
        From the NER pipelines, clean the tokenized results.

        Returns a list of entities based on the BIO method. Results may lack spaces between words.
        '''
        results = list()

        current_word = ''
        for result in pipeline_results:
            if result['entity'] == '0':
                continue
            else:
                if current_word != '' and result['entity'][0] == 'B':
                    results.append(current_word)
                    current_word = ''
                current_word += result['word'].replace('#', '')

        results.append(current_word)

        # Prevent accidentally returning '' as the first string
        if results[0] == '':
            results = results[1:]

        return results

    def get_entities(self, text):
        disease = self.clean_pipeline_results(self.disease_pipe(text))
        gene = self.clean_pipeline_results(self.genetic_pipe(text))
        chemical = self.clean_pipeline_results(self.chemical_pipe(text))
        return disease + chemical + gene

if __name__ == "__main__":
    foo = BioBERT_NER()
    text = "How viable is this hypothesis: Mercuric Chloride interacts with Alpha-Synuclein and other proteins involved in protein misfolding and aggregation pathways, exacerbating neurotoxicity"
    bar = foo.foo(text)
    print(bar)