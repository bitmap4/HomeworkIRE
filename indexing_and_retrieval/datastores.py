import pickle
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional
import psycopg2
from psycopg2.extras import Json, execute_values
import redis
from omegaconf import DictConfig

class DataStoreBase:
    def put(self, key: str, value: Any):
        raise NotImplementedError
    
    def get(self, key: str) -> Optional[Any]:
        raise NotImplementedError
        
    def get_all(self) -> Dict[str, any]:
        raise NotImplementedError

    def exists(self, key: str) -> bool:
        raise NotImplementedError
    
    def close(self):
        pass

class CustomDiskStore(DataStoreBase):
    def __init__(self, storage_path: str):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.cache = {}
        self.key_map_file = self.storage_path / '_key_map.pkl'
        self.key_to_hash = {}
        self.hash_to_key = {}
        self._load_key_map()
    
    def _load_key_map(self):
        if self.key_map_file.exists():
            with open(self.key_map_file, 'rb') as f:
                self.key_to_hash = pickle.load(f)
                self.hash_to_key = {v: k for k, v in self.key_to_hash.items()}
    
    def _save_key_map(self):
        with open(self.key_map_file, 'wb') as f:
            pickle.dump(self.key_to_hash, f)
    
    def _get_hash(self, key: str) -> str:
        if key not in self.key_to_hash:
            key_hash = hashlib.sha256(key.encode('utf-8')).hexdigest()[:16]
            self.key_to_hash[key] = key_hash
            self.hash_to_key[key_hash] = key
        return self.key_to_hash[key]

    def put(self, key: str, value):
        self.cache[key] = value

    def get(self, key: str):
        if key in self.cache:
            return self.cache[key]
        
        key_hash = self._get_hash(key)
        file_path = self.storage_path / f"{key_hash}.pkl"
        if file_path.exists():
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        return None
    
    def get_all(self) -> Dict[str, any]:
        all_data = {}
        for file_path in self.storage_path.glob("*.pkl"):
            if file_path.name == '_key_map.pkl':
                continue
            key_hash = file_path.stem
            if key_hash in self.hash_to_key:
                key = self.hash_to_key[key_hash]
                with open(file_path, 'rb') as f:
                    all_data[key] = pickle.load(f)
        return all_data

    def exists(self, key: str) -> bool:
        if key not in self.key_to_hash:
            return False
        key_hash = self.key_to_hash[key]
        return (self.storage_path / f"{key_hash}.pkl").exists()

    def commit(self):
        for key, value in self.cache.items():
            key_hash = self._get_hash(key)
            with open(self.storage_path / f"{key_hash}.pkl", 'wb') as f:
                pickle.dump(value, f)
        self._save_key_map()
        self.cache.clear()

class PostgresStore(DataStoreBase):
    def __init__(self, config: DictConfig, table_name: str):
        self.config = config
        self.table_name = table_name
        self.conn = psycopg2.connect(
            host=config.postgres.host,
            port=config.postgres.port,
            database=config.postgres.database,
            user=config.postgres.user,
            password=config.postgres.password
        )
        self._create_table()
    
    def _create_table(self):
        with self.conn.cursor() as cur:
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    key TEXT PRIMARY KEY,
                    value BYTEA,
                    is_json BOOLEAN DEFAULT TRUE
                )
            """)
            self.conn.commit()
    
    def put(self, key: str, value: Any):
        # Handle different data types
        if isinstance(value, bytes):
            # Binary data (compressed)
            data_to_store = value
            is_json = False
        elif isinstance(value, str):
            # String data (JSON string)
            data_to_store = value.encode('utf-8')
            is_json = True
        else:
            # Other data - serialize to JSON
            data_to_store = json.dumps(value).encode('utf-8')
            is_json = True
        
        with self.conn.cursor() as cur:
            cur.execute(f"""
                INSERT INTO {self.table_name} (key, value, is_json) 
                VALUES (%s, %s, %s)
                ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, is_json = EXCLUDED.is_json
            """, (key, data_to_store, is_json))
            self.conn.commit()
    
    def get(self, key: str):
        with self.conn.cursor() as cur:
            cur.execute(f"SELECT value, is_json FROM {self.table_name} WHERE key = %s", (key,))
            result = cur.fetchone()
            if result:
                value, is_json = result
                if is_json:
                    # Return as string (JSON string)
                    return value.tobytes().decode('utf-8') if hasattr(value, 'tobytes') else value.decode('utf-8')
                else:
                    # Return as bytes (compressed data)
                    return value.tobytes() if hasattr(value, 'tobytes') else bytes(value)
            return None

    def get_all(self) -> Dict[str, any]:
        with self.conn.cursor() as cur:
            cur.execute(f"SELECT key, value, is_json FROM {self.table_name}")
            result = {}
            for row in cur.fetchall():
                key, value, is_json = row
                if is_json:
                    # Return as string (JSON string)
                    result[key] = value.tobytes().decode('utf-8') if hasattr(value, 'tobytes') else value.decode('utf-8')
                else:
                    # Return as bytes (compressed data)
                    result[key] = value.tobytes() if hasattr(value, 'tobytes') else bytes(value)
            return result

    def exists(self, key: str) -> bool:
        with self.conn.cursor() as cur:
            cur.execute(f"SELECT 1 FROM {self.table_name} WHERE key = %s", (key,))
            return cur.fetchone() is not None
    
    def commit(self):
        # PostgresStore commits after each put operation, so this is a no-op
        pass
    
    def close(self):
        self.conn.close()

class RedisStore(DataStoreBase):
    def __init__(self, config: DictConfig, key_prefix: str = "", db: int = 0):
        self.config = config
        self.key_prefix = key_prefix
        self.client = redis.Redis(
            host=config.redis.host,
            port=config.redis.port,
            db=db,
            decode_responses=False
        )
    
    def _make_key(self, key: str) -> str:
        return f"{self.key_prefix}{key}"
    
    def put(self, key: str, value: Any):
        serialized = pickle.dumps(value)
        self.client.set(self._make_key(key), serialized)
    
    def get(self, key: str):
        value = self.client.get(self._make_key(key))
        if value:
            return pickle.loads(value)
        return None

    def get_all(self) -> Dict[str, any]:
        keys = self.client.keys(f"{self.key_prefix}*")
        if not keys:
            return {}
        values = self.client.mget(keys)
        return {
            key.decode('utf-8').replace(self.key_prefix, '', 1): pickle.loads(value)
            for key, value in zip(keys, values) if value
        }

    def exists(self, key: str) -> bool:
        return self.client.exists(self._make_key(key)) > 0
    
    def commit(self):
        # RedisStore writes immediately on put, so this is a no-op
        pass
    
    def close(self):
        self.client.close()

def get_datastore(store_type: str, config: DictConfig, **kwargs) -> DataStoreBase:
    if store_type == 'CUSTOM':
        storage_path = kwargs.get('storage_path', config.index.datastores.custom.storage_path)
        return CustomDiskStore(storage_path)
    elif store_type == 'DB1':
        table_name = kwargs.get('table_name', config.index.datastores.postgres.table_name)
        # Sanitize table name - replace invalid characters with underscores
        table_name = table_name.replace('-', '_').replace('.', '_')
        return PostgresStore(config, table_name)
    elif store_type == 'DB2':
        key_prefix = kwargs.get('key_prefix', config.index.datastores.redis.key_prefix)
        return RedisStore(config, key_prefix, config.index.datastores.redis.db)
    else:
        raise ValueError(f"Unknown datastore type: {store_type}")
