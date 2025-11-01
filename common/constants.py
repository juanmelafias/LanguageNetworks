column_mapping = {
    "Word ID": "id_palabra",
    "Word": "palabra",
    "Language": "language",
    "Part of Speech": "POS",
    "Ranking": "ranking",
    "Lemma": "lemmatized",
    "PC1": "pc1",
    "PC2": "pc2",
    "PC3": "pc3",
    "RGB1": "rgb1",
    "RGB2": "rgb2",
    "RGB3": "rgb3",
    "PC1 Inflected Spanish": "pc1is",
    "PC2 Inflected Spanish": "pc2is",
    "PC3 Inflected Spanish": "pc3is",
    "NC5": "nc5",
    "NC4": "nc4",
    "NC3": "nc3",
    "NC2": "nc2",
    "Translation": "trans",
    "Frequency": "ranking_inv"
}

relevant_columns = [
    "Word", "Language", "Part of Speech", "Frequency", "Lemma", "Translation"
]

parts_of_speech = ['CCONJ', 'SCONJ', 'ADV', 'PRON', 'AUX', 'DET', 'ADP', 'NOUN', 'PROPN', 'ADJ', 'VERB', 'NUM', 'PART']

pos_mapping = {
    'CCONJ': 'Coordinating Conjunction',
    'SCONJ': 'Subordinating Conjunction',
    'ADV': 'Adverb',
    'PRON': 'Pronoun',
    'AUX': 'Auxiliary',
    'DET': 'Determiner',
    'ADP': 'Adposition',
    'NOUN': 'Noun',
    'PROPN': 'Proper Noun',
    'ADJ': 'Adjective',
    'VERB': 'Verb',
    'NUM': 'Number',
    'PART': 'Particle'
}