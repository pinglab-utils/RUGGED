import sys
import json
import re
from utils.ner_utils.biobert_ner import BioBERT_NER
from utils.ner_utils.spacy_ner import SpacyNER


class NamedEntityRecognition():
    def __init__(self):
        self.biobertner = BioBERT_NER()
        self.spacyner = SpacyNER()

    def jaccard_similarity(self, word1, word2):
        """
        Return the Jaccard Similarity score between two words
        """
        # Convert words into sets
        set1 = set(word1)
        set2 = set(word2)

        # Compute intersections and unions
        intersection = set1.intersection(set2)
        union = set1.union(set2)

        # Return Jaccard score
        return len(intersection) / len(union)

    def clean_results(self, all_entites: list, threshold=0.7):
        """
        Return results that are "too similar" to each other
        """
        indexes = list()
        cleaned_results = list()
        for i in range(len(all_entites)):
            for j in range(i, len(all_entites)):
                if i != j and self.jaccard_similarity(all_entites[i], all_entites[j]) > threshold:
                    indexes.append(j)

        for i in range(len(all_entites)):
            if i in indexes:
                continue
            cleaned_results.append(all_entites[i])

        return cleaned_results


    def get_context(self, text):
        # Get all entities
        spacy_res = self.spacyner.get_entities(text)
        # Additional cleaning step to only get entities
        spacy_res = [r[0] for r in spacy_res]
        bert_res = self.biobertner.get_entities(text)
        all_ents = spacy_res + bert_res
        cleaned_ents = self.clean_results(all_ents)
        return cleaned_ents


