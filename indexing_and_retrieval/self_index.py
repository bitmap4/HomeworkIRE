from typing import Dict, List, Set, Tuple, Iterable, Optional
from collections import defaultdict
import math
import json
from pathlib import Path
from omegaconf import DictConfig

from indexing_and_retrieval.index_base import IndexBase, IndexInfo, DataStore, Compression, QueryProc, Optimizations
from indexing_and_retrieval.preprocessing import TextPreprocessor
from indexing_and_retrieval.compression import get_compressor, CompressionBase
from indexing_and_retrieval.datastores import get_datastore, DataStoreBase
from indexing_and_retrieval.query_parser import parse_query, ASTNode, TermNode, PhraseNode, BinaryOpNode, UnaryOpNode

class PostingsList:
    def __init__(self):
        self.doc_ids: List[int] = []
        self.positions: Dict[int, List[int]] = {}
        self.term_freqs: Dict[int, int] = {}
        self.skip_pointers: Dict[int, int] = {}
    
    def add_occurrence(self, doc_id: int, position: int):
        if doc_id not in self.term_freqs:
            self.doc_ids.append(doc_id)
            self.term_freqs[doc_id] = 0
            self.positions[doc_id] = []
        
        self.term_freqs[doc_id] += 1
        self.positions[doc_id].append(position)
    
    def build_skip_pointers(self, skip_distance: int = None):
        n = len(self.doc_ids)
        if n < 2:
            return
        
        if skip_distance is None:
            skip_distance = int(math.sqrt(n))
        
        self.skip_pointers = {}
        for i in range(0, n - skip_distance, skip_distance):
            self.skip_pointers[i] = i + skip_distance
    
    def to_dict(self) -> dict:
        return {
            'doc_ids': self.doc_ids,
            'positions': {str(k): v for k, v in self.positions.items()},
            'term_freqs': {str(k): v for k, v in self.term_freqs.items()},
            'skip_pointers': {str(k): v for k, v in self.skip_pointers.items()}
        }
    
    @staticmethod
    def from_dict(data: dict) -> 'PostingsList':
        pl = PostingsList()
        pl.doc_ids = data['doc_ids']
        pl.positions = {int(k): v for k, v in data['positions'].items()}
        pl.term_freqs = {int(k): v for k, v in data['term_freqs'].items()}
        pl.skip_pointers = {int(k): v for k, v in data.get('skip_pointers', {}).items()}
        return pl

class SelfIndex(IndexBase):
    def __init__(self, config: DictConfig, core='SelfIndex', info='BOOLEAN', 
                 dstore='CUSTOM', qproc='TERMatat', compr='NONE', optim='Null'):
        super().__init__(core, info, dstore, qproc, compr, optim)
        
        self.config = config
        self.preprocessor = TextPreprocessor(config.preprocessing)
        
        self.inverted_index: Dict[str, PostingsList] = {}
        self.documents: Dict[int, str] = {}
        self.doc_id_map: Dict[str, int] = {}
        self.next_doc_id = 0
        
        self.doc_lengths: Dict[int, int] = {}
        self.avg_doc_length = 0
        self.total_docs = 0
        
        self.idf_scores: Dict[str, float] = {}
        self.tfidf_scores: Dict[str, Dict[int, float]] = {}
        
        self.info_type = IndexInfo[info]
        self.datastore_type = DataStore[dstore]
        self.compression_type = Compression[compr]
        self.query_proc_type = QueryProc[qproc]
        self.optim_type = Optimizations[optim]
        
        if self.compression_type == Compression.NONE:
            self.compressor = None
        else:
            self.compressor = get_compressor(compr, config)
        
        self.datastore: Optional[DataStoreBase] = None
    
    def create_index(self, index_id: str, files: Iterable[Tuple[str, str]]) -> None:
        self.index_id = index_id
        
        # Use a more descriptive directory name
        full_identifier = self.identifier_short
        storage_path = Path(self.config.paths.indices_dir) / full_identifier
        storage_path.mkdir(parents=True, exist_ok=True)
        
        self.datastore = get_datastore(
            self.datastore_type.name, 
            self.config, 
            storage_path=str(storage_path),
            table_name=full_identifier,
            key_prefix=f"{full_identifier}:"
        )
        
        # Check if index already exists
        if self.datastore.exists('metadata'):
            print(f"Loading existing index: {full_identifier}")
            try:
                self._load_index()
                return
            except Exception as e:
                # If metadata is missing or corrupted, log and fall back to rebuilding
                print(f"Warning: failed to load existing index metadata ({e}). Rebuilding index...")
                # continue to build index from scratch

        print(f"Building index: {full_identifier}")
        
        doc_count = 0
        for file_id, content in files:
            self._add_document(file_id, content)
            doc_count += 1
            if doc_count % 5000 == 0:
                print(f"  Processed {doc_count} documents...", flush=True)
        
        self.total_docs = len(self.documents)
        print(f"  Total: {self.total_docs} documents indexed", flush=True)
        
        print(f"  Computing statistics...", flush=True)
        self._compute_avg_doc_length()
        
        if self.info_type == IndexInfo.TFIDF:
            print(f"  Computing TF-IDF...", flush=True)
            self._compute_idf()
            self._compute_tfidf()
        
        if self.optim_type == Optimizations.Skipping:
            print(f"  Building skip pointers...", flush=True)
            self._build_skip_pointers()
        
        print(f"  Persisting index...", flush=True)
        self._persist_index()
        print(f"✓ Index created: {self.total_docs} documents, {len(self.inverted_index)} terms", flush=True)
    
    def _load_index(self):
        metadata_obj = self.datastore.get('metadata')
        if not metadata_obj:
            raise FileNotFoundError("Metadata not found in datastore.")

        # metadata_obj might be a dict (already deserialized), a JSON string, or bytes
        if isinstance(metadata_obj, dict):
            metadata = metadata_obj
        else:
            # Try to coerce into a dict via JSON
            try:
                if isinstance(metadata_obj, (bytes, bytearray)):
                    metadata = json.loads(metadata_obj.decode('utf-8'))
                else:
                    metadata = json.loads(str(metadata_obj))
            except Exception as e:
                raise ValueError(f"Unable to parse metadata JSON: {e}")

        # Basic validation of metadata keys
        required_keys = ['next_doc_id', 'doc_lengths', 'avg_doc_length', 'total_docs']
        if not all(k in metadata for k in required_keys):
            raise KeyError(f"Metadata missing required keys: {required_keys}. Found: {list(metadata.keys())}")

        # Don't load documents - they're not stored to save space
        self.documents = {}  # Empty - we don't need full text after indexing
        # doc_id_map is optional (older metadata may not include it); default to empty
        self.doc_id_map = metadata.get('doc_id_map', {})
        self.next_doc_id = metadata.get('next_doc_id', 0)
        self.doc_lengths = {int(k): v for k, v in metadata.get('doc_lengths', {}).items()}
        self.avg_doc_length = metadata.get('avg_doc_length', 0)
        self.total_docs = metadata.get('total_docs', 0)
        self.idf_scores = metadata.get('idf_scores', {})
        self.tfidf_scores = {k: {int(ik): iv for ik, iv in v.items()} for k, v in metadata.get('tfidf_scores', {}).items()}

        raw_index = self.datastore.get_all()
        self.inverted_index = {}
        for term, data in raw_index.items():
            if term == 'metadata':
                continue
            # data may be bytes, a dict, or a JSON string depending on datastore
            try:
                if self.compressor:
                    # compressor expects bytes-like input
                    raw = data
                    decompressed_data = self.compressor.decompress_pl(raw)
                    pl_dict = decompressed_data
                else:
                    if isinstance(data, dict):
                        pl_dict = data
                    elif isinstance(data, (bytes, bytearray)):
                        pl_dict = json.loads(data.decode('utf-8'))
                    else:
                        pl_dict = json.loads(str(data))

                self.inverted_index[term] = PostingsList.from_dict(pl_dict)
            except Exception as e:
                # Skip terms we can't parse but log for debugging
                print(f"Warning: failed to load postings for term '{term}': {e}")
                continue
        
        print(f"Successfully loaded index with {self.total_docs} documents and {len(self.inverted_index)} terms.")

    def load_index(self, index_id: str) -> None:
        self.index_id = index_id
        
        # Reconstruct storage_path based on the identifier
        storage_path = Path(self.config.paths.indices_dir) / self.identifier_short
        if not storage_path.exists():
            raise FileNotFoundError(f"Index directory not found: {storage_path}")
            
        self.datastore = get_datastore(
            self.datastore_type.name, 
            self.config, 
            storage_path=str(storage_path),
            table_name=self.identifier_short,
            key_prefix=f"{self.identifier_short}:"
        )
        self._load_index()

    def _add_document(self, file_id: str, content: str):
        if file_id in self.doc_id_map:
            return
        
        doc_id = self.next_doc_id
        self.doc_id_map[file_id] = doc_id
        self.documents[doc_id] = file_id
        self.next_doc_id += 1
        
        tokens = self.preprocessor.tokenize(content, preprocess=True)
        self.doc_lengths[doc_id] = len(tokens)
        
        for i, token in enumerate(tokens):
            if token not in self.inverted_index:
                self.inverted_index[token] = PostingsList()
            self.inverted_index[token].add_occurrence(doc_id, i)

    def _compute_avg_doc_length(self):
        if self.total_docs > 0:
            self.avg_doc_length = sum(self.doc_lengths.values()) / self.total_docs
        else:
            self.avg_doc_length = 0

    def _compute_idf(self):
        self.idf_scores = {}
        for term, pl in self.inverted_index.items():
            doc_freq = len(pl.doc_ids)
            self.idf_scores[term] = math.log((self.total_docs - doc_freq + 0.5) / (doc_freq + 0.5) + 1)

    def _compute_tfidf(self):
        self.tfidf_scores = defaultdict(dict)
        k1 = 1.2
        b = 0.75
        
        for term, pl in self.inverted_index.items():
            for doc_id in pl.doc_ids:
                tf = pl.term_freqs[doc_id]
                doc_len = self.doc_lengths[doc_id]
                
                numerator = tf * (k1 + 1)
                denominator = tf + k1 * (1 - b + b * (doc_len / self.avg_doc_length))
                
                self.tfidf_scores[term][doc_id] = self.idf_scores[term] * (numerator / denominator)

    def _build_skip_pointers(self):
        for pl in self.inverted_index.values():
            pl.build_skip_pointers()

    def _persist_index(self):
        # Don't store full document text - only store metadata needed for querying
        metadata = {
            'identifier': self.identifier_short,
            'doc_id_map': self.doc_id_map,
            'next_doc_id': self.next_doc_id,
            'doc_lengths': self.doc_lengths,
            'avg_doc_length': self.avg_doc_length,
            'total_docs': self.total_docs,
            'idf_scores': self.idf_scores,
            'tfidf_scores': self.tfidf_scores
        }
        # Serialize metadata to a JSON string before storing
        self.datastore.put('metadata', json.dumps(metadata))

        # Batch persistence: write terms in batches and commit periodically to avoid
        # excessive memory use or extremely long single transactions.
        # For CustomDiskStore with consolidated file: larger batch = fewer writes
        try:
            batch_size = int(self.config.persist.batch_size)
        except Exception:
            # Use much larger batch for disk-based stores to minimize I/O
            batch_size = 5000
        term_count = 0
        total_terms = len(self.inverted_index)

        try:
            batch_terms = 0
            for term, pl in self.inverted_index.items():
                pl_dict = pl.to_dict()
                if self.compressor:
                    compressed_data = self.compressor.compress_pl(pl_dict)
                    self.datastore.put(term, compressed_data)
                else:
                    # Serialize to JSON string if not compressing
                    self.datastore.put(term, json.dumps(pl_dict))

                term_count += 1
                batch_terms += 1

                if batch_terms >= batch_size:
                    # Commit the batch and print progress
                    self.datastore.commit()
                    print(f"  Persisted {term_count}/{total_terms} terms...", flush=True)
                    batch_terms = 0

            # Final commit for any remaining items
            if batch_terms > 0:
                self.datastore.commit()
                print(f"  Persisted {term_count}/{total_terms} terms...", flush=True)
        except Exception as e:
            # Attempt to commit any buffered writes, then re-raise
            try:
                self.datastore.commit()
            except Exception:
                pass
            raise

    def update_index(self, add_files: Iterable[Tuple[str, str]] = None, remove_files: Iterable[str] = None):
        if add_files is None:
            add_files = []
        if remove_files is None:
            remove_files = []
        
        for file_id in remove_files:
            if file_id in self.doc_id_map:
                doc_id = self.doc_id_map[file_id]
                del self.documents[doc_id]
                del self.doc_id_map[file_id]
                if doc_id in self.doc_lengths:
                    del self.doc_lengths[doc_id]
        
        for file_id, content in add_files:
            self._add_document(file_id, content)
        
        self.total_docs = len(self.documents)
        self._compute_avg_doc_length()
        
        if self.info_type == IndexInfo.TFIDF:
            self._compute_idf()
            self._compute_tfidf()
        
        if self.optim_type == Optimizations.Skipping:
            self._build_skip_pointers()
        
        self._persist_index()
    
    def query(self, query_str: str) -> str:
        ast = parse_query(query_str)
        
        if self.query_proc_type == QueryProc.TERMatat:
            result_doc_ids = self._execute_term_at_a_time(ast)
        else:
            result_doc_ids = self._execute_document_at_a_time(ast)
        
        results = []
        for doc_id in result_doc_ids:
            if doc_id in self.documents:
                results.append({
                    'doc_id': self.documents[doc_id],
                    'internal_id': doc_id
                })
        
        return json.dumps({
            'query': query_str,
            'num_results': len(results),
            'results': results
        }, indent=2)
    
    def delete_index(self, index_id: str) -> None:
        if self.datastore:
            self.datastore.close()
        
        storage_path = Path(self.config.paths.indices_dir) / self.identifier_short
        if storage_path.exists():
            import shutil
            shutil.rmtree(storage_path)
            print(f"Deleted index: {self.identifier_short}")

    def list_indices(self) -> List[str]:
        indices_dir = Path(self.config.paths.indices_dir)
        if not indices_dir.exists():
            return []
        return [d.name for d in indices_dir.iterdir() if d.is_dir() and d.name.startswith('SelfIndex-v1.')]

    def list_indexed_files(self, index_id: str) -> List[str]:
        # Assuming the index is loaded, this will return the files for the current index.
        # The index_id parameter is for interface consistency.
        return list(self.doc_id_map.keys())

    def _execute_term_at_a_time(self, node: ASTNode) -> Set[int]:
        if isinstance(node, TermNode):
            term = self.preprocessor.tokenize(node.term, preprocess=True)
            if term and term[0] in self.inverted_index:
                return set(self.inverted_index[term[0]].doc_ids)
            return set()
        
        elif isinstance(node, PhraseNode):
            return self._search_phrase(node.phrase)
        
        elif isinstance(node, BinaryOpNode):
            left_results = self._execute_term_at_a_time(node.left)
            right_results = self._execute_term_at_a_time(node.right)
            
            if node.op == 'AND':
                if self.optim_type == Optimizations.Skipping:
                    return self._intersect_with_skips(left_results, right_results, node.left, node.right)
                return left_results & right_results
            elif node.op == 'OR':
                return left_results | right_results
        
        elif isinstance(node, UnaryOpNode):
            if node.op == 'NOT':
                operand_results = self._execute_term_at_a_time(node.operand)
                all_docs = set(self.documents.keys())
                return all_docs - operand_results
        
        return set()
    
    def _intersect_with_skips(self, left_set: Set[int], right_set: Set[int], 
                               left_node: ASTNode, right_node: ASTNode) -> Set[int]:
        if not isinstance(left_node, TermNode) or not isinstance(right_node, TermNode):
            return left_set & right_set
        
        left_term = self.preprocessor.tokenize(left_node.term, preprocess=True)
        right_term = self.preprocessor.tokenize(right_node.term, preprocess=True)
        
        if not left_term or not right_term:
            return left_set & right_set
        
        left_term = left_term[0]
        right_term = right_term[0]
        
        if left_term not in self.inverted_index or right_term not in self.inverted_index:
            return set()
        
        left_postings = self.inverted_index[left_term]
        right_postings = self.inverted_index[right_term]
        
        result = set()
        i, j = 0, 0
        left_docs = left_postings.doc_ids
        right_docs = right_postings.doc_ids
        
        while i < len(left_docs) and j < len(right_docs):
            if left_docs[i] == right_docs[j]:
                result.add(left_docs[i])
                i += 1
                j += 1
            elif left_docs[i] < right_docs[j]:
                if i in left_postings.skip_pointers and left_docs[left_postings.skip_pointers[i]] <= right_docs[j]:
                    i = left_postings.skip_pointers[i]
                else:
                    i += 1
            else:
                if j in right_postings.skip_pointers and right_docs[right_postings.skip_pointers[j]] <= left_docs[i]:
                    j = right_postings.skip_pointers[j]
                else:
                    j += 1
        
        return result
    
    def _execute_document_at_a_time(self, node: ASTNode) -> Set[int]:
        all_docs = set(self.documents.keys())
        matching_docs = set()
        
        for doc_id in all_docs:
            if self._evaluate_node_for_doc(node, doc_id):
                matching_docs.add(doc_id)
        
        return matching_docs
    
    def _evaluate_node_for_doc(self, node: ASTNode, doc_id: int) -> bool:
        if isinstance(node, TermNode):
            term = self.preprocessor.tokenize(node.term, preprocess=True)
            if term and term[0] in self.inverted_index:
                return doc_id in self.inverted_index[term[0]].doc_ids
            return False
        elif isinstance(node, PhraseNode):
            return doc_id in self._search_phrase(node.phrase)
        elif isinstance(node, BinaryOpNode):
            left = self._evaluate_node_for_doc(node.left, doc_id)
            right = self._evaluate_node_for_doc(node.right, doc_id)
            if node.op == 'AND':
                return left and right
            elif node.op == 'OR':
                return left or right
        elif isinstance(node, UnaryOpNode):
            if node.op == 'NOT':
                return not self._evaluate_node_for_doc(node.operand, doc_id)
        return False
    
    def _search_phrase(self, phrase: str) -> Set[int]:
        tokens = self.preprocessor.tokenize(phrase, preprocess=True)
        if not tokens:
            return set()
        
        if len(tokens) == 1:
            if tokens[0] in self.inverted_index:
                return set(self.inverted_index[tokens[0]].doc_ids)
            return set()
        
        candidate_docs = None
        for token in tokens:
            if token in self.inverted_index:
                docs = set(self.inverted_index[token].doc_ids)
                if candidate_docs is None:
                    candidate_docs = docs
                else:
                    candidate_docs &= docs
            else:
                return set()
        
        if not candidate_docs:
            return set()
        
        result_docs = set()
        for doc_id in candidate_docs:
            if self._check_phrase_in_doc(tokens, doc_id):
                result_docs.add(doc_id)
        
        return result_docs
    
    def _check_phrase_in_doc(self, tokens: List[str], doc_id: int) -> bool:
        first_term = tokens[0]
        if first_term not in self.inverted_index:
            return False
        
        positions = self.inverted_index[first_term].positions.get(doc_id, [])
        
        for start_pos in positions:
            match = True
            for i, token in enumerate(tokens[1:], 1):
                if token not in self.inverted_index or doc_id not in self.inverted_index[token].doc_ids:
                    match = False
                    break
            
            if match:
                return True
        
        return False