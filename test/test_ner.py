import unittest
import spacy

class TestSpaCyModels(unittest.TestCase):
    def test_disease_ner_model(self):
        try:
            nlp = spacy.load("en_ner_bc5cdr_md")
            self.assertIsNotNone(nlp, "Disease NER model 'en_ner_bc5cdr_md' loaded successfully.")
        except OSError:
            self.fail("Failed to load Disease NER model 'en_ner_bc5cdr_md'. Make sure the model is downloaded.\nUse "
                      "the following command: pip install "
                      "https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.3/en_ner_bc5cdr_md-0.5.3.tar"
                      ".gz")

    def test_scientific_entity_ner_model(self):
        try:
            nlp = spacy.load("en_ner_bionlp13cg_md")
            self.assertIsNotNone(nlp, "Scientific Entity NER model 'en_ner_bionlp13cg_md' loaded successfully.")
        except OSError:
            self.fail("Failed to load Scientific Entity NER model 'en_ner_bionlp13cg_md'. Make sure the model is "
                      "downloaded.\nUse the following command: pip install "
                      "https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.3/en_ner_bionlp13cg_md-0.5.3"
                      ".tar.gz")

    def test_part_of_speech_model(self):
        try:
            nlp = spacy.load("en_core_web_sm")
            self.assertIsNotNone(nlp, "Part of Speech model 'en_core_web_sm' loaded successfully.")
        except OSError:
            self.fail("Failed to load Part of Speech model 'en_core_web_sm'. Make sure the model is downloaded.")

if __name__ == "__main__":
    unittest.main()
