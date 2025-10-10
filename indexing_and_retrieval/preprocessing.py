import string
import re
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
        
        # Pre-compile regex for faster punctuation removal
        if self.config.remove_punctuation:
            self.punctuation_pattern = re.compile(f'[{re.escape(string.punctuation)}]')
        
        # Cache for stemmed words to avoid re-stemming common terms
        self._stem_cache = {}
    
    def _stem_cached(self, word: str) -> str:
        """Cached stemming for performance."""
        if word not in self._stem_cache:
            self._stem_cache[word] = self.stemmer.stem(word)
        return self._stem_cache[word]
    
    def tokenize(self, text: str, preprocess: bool = True) -> List[str]:
        if not text:
            return []
        
        if self.config.lowercase:
            text = text.lower()
        
        if self.config.remove_punctuation:
            # Use pre-compiled regex (faster than str.translate for large texts)
            text = self.punctuation_pattern.sub(' ', text)
        
        tokens = text.split()
        
        if self.config.min_token_length > 0:
            tokens = [t for t in tokens if len(t) >= self.config.min_token_length]
        
        if not preprocess:
            return tokens
        
        # Filter stopwords and stem in a single pass (more efficient)
        tokens = [self._stem_cached(t) for t in tokens if t not in self.stop_words]
        
        return tokens
    
    def compute_term_frequencies(self, documents: List[str], preprocess: bool = True) -> Counter:
        term_freq = Counter()
        for doc in documents:
            tokens = self.tokenize(doc, preprocess=preprocess)
            term_freq.update(tokens)
        return term_freq
