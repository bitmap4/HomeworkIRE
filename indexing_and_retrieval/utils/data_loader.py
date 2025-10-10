import zipfile
import json
import os
from pathlib import Path
from typing import Iterator, Tuple, List
import pandas as pd
from omegaconf import DictConfig
from tqdm import tqdm

class NewsDataLoader:
    def __init__(self, config: DictConfig):
        self.config = config
        self.news_dir = Path(config.paths.news_dir)
    
    def load_documents(self) -> Iterator[Tuple[str, str]]:
        selected_zips = self.config.data.news.selected_zips
        max_zips = self.config.data.news.get('max_zips', None)
        max_docs_per_zip = self.config.data.news.max_docs_per_zip
        
        # If selected_zips is empty, use all available zip files
        if not selected_zips:
            all_zips = sorted([f.name for f in self.news_dir.glob('*.zip')])
            if max_zips:
                selected_zips = all_zips[:max_zips]
            else:
                selected_zips = all_zips
            print(f"No specific zips selected, using {len(selected_zips)} zip files")
        
        for zip_name in selected_zips:
            zip_path = self.news_dir / zip_name
            if not zip_path.exists():
                print(f"Warning: {zip_path} not found, skipping...")
                continue
            
            doc_count = 0
            with zipfile.ZipFile(zip_path, 'r') as zf:
                json_files = [f for f in zf.namelist() if f.endswith('.json')]
                
                for json_file in tqdm(json_files, desc=f"Loading {zip_name}"):
                    if doc_count >= max_docs_per_zip:
                        break
                    
                    try:
                        with zf.open(json_file) as f:
                            data = json.load(f)
                            doc_id = f"{zip_name}_{json_file}"
                            
                            text_parts = []
                            if 'title' in data and data['title']:
                                text_parts.append(data['title'])
                            if 'text' in data and data['text']:
                                text_parts.append(data['text'])
                            
                            if text_parts:
                                text = ' '.join(text_parts)
                                yield (doc_id, text)
                                doc_count += 1
                    except Exception as e:
                        print(f"Error loading {json_file}: {e}")
                        continue

class WikiDataLoader:
    def __init__(self, config: DictConfig):
        self.config = config
        self.wiki_file = Path(config.paths.wiki_file)
    
    def load_documents(self) -> Iterator[Tuple[str, str]]:
        if not self.wiki_file.exists():
            print(f"Warning: {self.wiki_file} not found, skipping wiki data...")
            return
        
        max_docs = self.config.data.wiki.max_docs
        text_col = self.config.data.wiki.text_column
        id_col = self.config.data.wiki.id_column
        
        df = pd.read_parquet(self.wiki_file)
        
        for idx, row in tqdm(df.head(max_docs).iterrows(), total=min(max_docs, len(df)), desc="Loading Wiki"):
            doc_id = f"wiki_{row.get(id_col, idx)}"
            text = row.get(text_col, '')
            
            if text and isinstance(text, str):
                yield (doc_id, text)

class DataManager:
    def __init__(self, config: DictConfig):
        self.config = config
        self.news_loader = NewsDataLoader(config)
        self.wiki_loader = WikiDataLoader(config)
    
    def load_all_documents(self) -> List[Tuple[str, str]]:
        documents = []
        
        print("Loading news documents...")
        for doc in self.news_loader.load_documents():
            documents.append(doc)
        
        # Only load wiki if enabled
        if self.config.data.wiki.get('enabled', True):
            print("Loading wiki documents...")
            for doc in self.wiki_loader.load_documents():
                documents.append(doc)
        else:
            print("Wiki data disabled, skipping...")
        
        print(f"Total documents loaded: {len(documents)}")
        return documents
