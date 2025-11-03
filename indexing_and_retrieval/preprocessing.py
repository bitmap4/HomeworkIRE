import string
from typing import List
from collections import Counter
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from omegaconf import DictConfig

class TextPreprocessor:
    def __init__(self, config: DictConfig):
        self.config = config
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords', quiet=True)
        
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words(config.stopwords))
    
    def tokenize(self, text: str, preprocess: bool = True) -> List[str]:
        if not text:
            return []
        
        if self.config.lowercase:
            text = text.lower()
        
        if self.config.remove_punctuation:
            text = text.translate(str.maketrans('', '', string.punctuation))
        
        tokens = text.split()
        
        if self.config.min_token_length > 0:
            tokens = [t for t in tokens if len(t) >= self.config.min_token_length]
        
        if not preprocess:
            return tokens
        
        tokens = [t for t in tokens if t not in self.stop_words]
        tokens = [self.stemmer.stem(t) for t in tokens]
        
        return tokens
    
    def compute_term_frequencies(self, documents: List[str], preprocess: bool = True) -> Counter:
        term_freq = Counter()
        for doc in documents:
            tokens = self.tokenize(doc, preprocess=preprocess)
            term_freq.update(tokens)
        return term_freq
