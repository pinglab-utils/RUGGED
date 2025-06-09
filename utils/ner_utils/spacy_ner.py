import json
import spacy


from config import NODE_FEATURES_PATH
from config import NODE_RECORDS_PATH

class SpacyNER:
    def __init__(self):
        self.disease_ner_nlp = spacy.load("en_ner_bc5cdr_md")
        self.scientific_entity_nlp = spacy.load("en_ner_bionlp13cg_md")
        self.pos_nlp = spacy.load("en_core_web_sm")
        #self.graph_nodes = json.loads(open(NODE_RECORDS_PATH, "r").read())

        # ADDED RECENTLY
        # Might be able to get rid of some stuff above
        self.node_features = json.loads(open(NODE_FEATURES_PATH, 'r').read())

    def disease_ner(self, text: str):
        document = self.disease_ner_nlp(text)
        return [(ent.text, ent.label_) for ent in document.ents]

    def scientific_entity_ner(self, text: str):
        document = self.scientific_entity_nlp(text)
        return [(ent.text, ent.label_) for ent in document.ents]

    def get_entities(self, text: str):
        """
        The primary function to be used that will gather disease and scientific entites.
        Returns a list of tuples.

        Example usage:
        [(ent0, 'DISEASE'), (ent1, 'CHEMICAL')]
        """
        dis_ent = self.disease_ner(text)
        sci_ent = self.scientific_entity_ner(text)
        return dis_ent + sci_ent


if __name__ == '__main__':
    x = SpacyNER()
    TEXT = "How viable is this hypothesis: Mercuric Chloride interacts with Alpha-Synuclein and other proteins involved in protein misfolding and aggregation pathways, exacerbating neurotoxicity"
    y = x.get_entities(TEXT)
    print(y)
