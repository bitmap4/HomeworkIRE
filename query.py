import sys
import hydra
from omegaconf import DictConfig
from pathlib import Path
from dotenv import load_dotenv

from indexing_and_retrieval.es_index import ESIndex
from indexing_and_retrieval.self_index import SelfIndex

load_dotenv()

@hydra.main(version_base=None, config_path="config", config_name="config")
def query_cli(cfg: DictConfig):
    if len(sys.argv) < 3:
        print("Usage: python query.py <index_type> <query>")
        print("  index_type: 'es' or 'self'")
        print("  query: Boolean query string (in quotes)")
        print("\nExample:")
        print('  python query.py es \'"machine" AND "learning"\'')
        return
    
    index_type = sys.argv[1]
    query_str = sys.argv[2]
    
    if index_type == 'es':
        print("Loading Elasticsearch index...")
        index = ESIndex(cfg)
        index.load_index("esindex-v1.0")
    elif index_type == 'self':
        print("Loading SelfIndex (BOOLEAN, CUSTOM, NONE)...")
        index = SelfIndex(cfg, 'SelfIndex', 'BOOLEAN', 'CUSTOM', 'TERMatat', 'NONE', 'Null')
        
        indices_dir = Path(cfg.paths.indices_dir)
        index_path = indices_dir / index.identifier_short
        
        if not index_path.exists():
            print(f"Error: Index not found at {index_path}")
            print("Please run 'python main.py' first to create the indices.")
            return
        
        index.load_index(str(index_path))
    else:
        print(f"Unknown index type: {index_type}")
        print("Use 'es' or 'self'")
        return
    
    print(f"\nExecuting query: {query_str}")
    print("-" * 80)
    
    result = index.query(query_str)
    print(result)

if __name__ == "__main__":
    query_cli()
