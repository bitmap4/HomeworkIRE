"""IRE Assignment 1: Indexing and Retrieval System

This package contains the implementation of various indexing and retrieval systems
including Elasticsearch integration and custom self-built indices.
"""

from indexing_and_retrieval.core.index_base import IndexBase, IndexInfo, DataStore, Compression, QueryProc, Optimizations
from indexing_and_retrieval.utils.preprocessing import TextPreprocessor
from indexing_and_retrieval.utils.data_loader import DataManager
from indexing_and_retrieval.core.es_index import ESIndex
from indexing_and_retrieval.core.self_index import SelfIndex

__version__ = '1.0.0'

__all__ = [
    'IndexBase',
    'IndexInfo',
    'DataStore', 
    'Compression',
    'QueryProc',
    'Optimizations',
    'TextPreprocessor',
    'DataManager',
    'ESIndex',
    'SelfIndex',
]
