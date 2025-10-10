from typing import Iterable, Tuple
import json
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from omegaconf import DictConfig

from indexing_and_retrieval.core.index_base import IndexBase
from indexing_and_retrieval.utils.preprocessing import TextPreprocessor
from indexing_and_retrieval.utils.query_parser import parse_query, ASTNode, TermNode, PhraseNode, BinaryOpNode, UnaryOpNode

class ESIndex(IndexBase):
    def __init__(self, config: DictConfig, core='ESIndex', info='BOOLEAN', 
                 dstore='DB1', qproc='TERMatat', compr='NONE', optim='Null'):
        super().__init__(core, info, dstore, qproc, compr, optim)
        
        self.config = config
        self.preprocessor = TextPreprocessor(config.preprocessing)
        
        self.client = Elasticsearch(
            [f"http://{config.elasticsearch.host}:{config.elasticsearch.port}"],
            request_timeout=30,  # Increase default timeout to 30 seconds
            max_retries=3,
            retry_on_timeout=True
        )
        
        self.index_name = None
    
    def create_index(self, index_id: str, files: Iterable[Tuple[str, str]]) -> None:
        self.index_name = index_id.lower().replace('_', '-')
        
        if self.client.indices.exists(index=self.index_name):
            self.client.indices.delete(index=self.index_name)
        
        index_settings = {
            "settings": {
                "number_of_shards": self.config.index.es_index.number_of_shards,
                "number_of_replicas": self.config.index.es_index.number_of_replicas,
                "refresh_interval": self.config.index.es_index.refresh_interval,
                "analysis": {
                    "analyzer": {
                        "custom_analyzer": {
                            "type": "custom",
                            "tokenizer": "standard",
                            "filter": ["lowercase", "porter_stem", "stop"]
                        }
                    }
                }
            },
            "mappings": {
                "properties": {
                    "doc_id": {"type": "keyword"},
                    "content": {
                        "type": "text",
                        "analyzer": "custom_analyzer"
                    },
                    "content_raw": {
                        "type": "text",
                        "analyzer": "standard"
                    }
                }
            }
        }
        
        # Create index with increased timeout
        print(f"Creating index {self.index_name}...")
        
        # If index exists, delete it first (handles stale/incomplete indices)
        if self.client.indices.exists(index=self.index_name):
            print(f"Deleting existing index {self.index_name}...")
            self.client.indices.delete(index=self.index_name)
        
        try:
            self.client.indices.create(
                index=self.index_name, 
                body=index_settings,
                timeout='60s'  # Increased timeout
            )
            print(f"Index {self.index_name} created successfully")
        except Exception as e:
            print(f"Warning: Index creation had issues: {e}")
            # Check if index was actually created despite the error
            if self.client.indices.exists(index=self.index_name):
                print(f"Index {self.index_name} exists despite error, continuing...")
            else:
                raise
        
        print(f"Bulk indexing {len(files)} documents...")
        def generate_docs():
            for file_id, content in files:
                yield {
                    "_index": self.index_name,
                    "_id": file_id,
                    "doc_id": file_id,
                    "content": content,
                    "content_raw": content
                }
        
        # Use bulk with chunking and timeout
        success, failed = bulk(
            self.client, 
            generate_docs(), 
            chunk_size=500,  # Process in smaller chunks
            request_timeout=30,  # Increase timeout
            raise_on_error=False
        )
        
        self.client.indices.refresh(index=self.index_name)
        
        print(f"ES Index created: {self.index_name}")
        print(f"Successfully indexed: {success} documents")
        if failed:
            print(f"Failed to index: {len(failed)} documents")
    
    def load_index(self, serialized_index_dump: str) -> None:
        self.index_name = serialized_index_dump
        
        if not self.client.indices.exists(index=self.index_name):
            raise ValueError(f"Index {self.index_name} does not exist")
        
        print(f"Loaded ES index: {self.index_name}")
    
    def update_index(self, index_id: str, remove_files: Iterable[Tuple[str, str]], 
                    add_files: Iterable[Tuple[str, str]]) -> None:
        self.index_name = index_id.lower().replace('_', '-')
        
        for file_id, _ in remove_files:
            try:
                self.client.delete(index=self.index_name, id=file_id)
            except:
                pass
        
        def generate_docs():
            for file_id, content in add_files:
                yield {
                    "_index": self.index_name,
                    "_id": file_id,
                    "doc_id": file_id,
                    "content": content,
                    "content_raw": content
                }
        
        bulk(self.client, generate_docs(), raise_on_error=False)
        self.client.indices.refresh(index=self.index_name)
    
    def query(self, query_str: str) -> str:
        ast = parse_query(query_str)
        es_query = self._build_es_query(ast)
        
        response = self.client.search(
            index=self.index_name,
            body={
                "query": es_query,
                "size": 10000
            }
        )
        
        results = []
        for hit in response['hits']['hits']:
            results.append({
                'doc_id': hit['_source']['doc_id'],
                'score': hit['_score']
            })
        
        return json.dumps({
            'query': query_str,
            'num_results': len(results),
            'results': results
        }, indent=2)
    
    def _build_es_query(self, node: ASTNode) -> dict:
        if isinstance(node, TermNode):
            return {
                "match": {
                    "content": node.term
                }
            }
        
        elif isinstance(node, PhraseNode):
            return {
                "match_phrase": {
                    "content": node.phrase
                }
            }
        
        elif isinstance(node, BinaryOpNode):
            left_query = self._build_es_query(node.left)
            right_query = self._build_es_query(node.right)
            
            if node.op == 'AND':
                return {
                    "bool": {
                        "must": [left_query, right_query]
                    }
                }
            elif node.op == 'OR':
                return {
                    "bool": {
                        "should": [left_query, right_query],
                        "minimum_should_match": 1
                    }
                }
        
        elif isinstance(node, UnaryOpNode):
            if node.op == 'NOT':
                operand_query = self._build_es_query(node.operand)
                return {
                    "bool": {
                        "must_not": [operand_query]
                    }
                }
        
        return {"match_all": {}}
    
    def delete_index(self, index_id: str) -> None:
        index_name = index_id.lower().replace('_', '-')
        if self.client.indices.exists(index=index_name):
            self.client.indices.delete(index=index_name)
            print(f"Deleted ES index: {index_name}")
    
    def list_indices(self) -> list:
        indices = self.client.indices.get_alias(index="*")
        return list(indices.keys())
    
    def list_indexed_files(self, index_id: str) -> list:
        index_name = index_id.lower().replace('_', '-')
        result = self.client.search(
            index=index_name,
            body={"query": {"match_all": {}}, "_source": ["doc_id"], "size": 10000}
        )
        return [hit['_source']['doc_id'] for hit in result['hits']['hits']]
