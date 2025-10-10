import pickle
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional
import psycopg2
from psycopg2.extras import Json, execute_values
from psycopg2 import Binary
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
        self.data_file = self.storage_path / 'index_data.pkl'
        self.persistent_data = {}
        self._load_data()
    
    def _load_data(self):
        """Load all data from the single consolidated file."""
        if self.data_file.exists():
            try:
                with open(self.data_file, 'rb') as f:
                    self.persistent_data = pickle.load(f)
            except Exception as e:
                print(f"Warning: Failed to load index data: {e}")
                self.persistent_data = {}

    def put(self, key: str, value):
        self.cache[key] = value

    def get(self, key: str):
        # Check cache first, then persistent data
        if key in self.cache:
            return self.cache[key]
        return self.persistent_data.get(key)
    
    def get_all(self) -> Dict[str, any]:
        # Merge persistent data with cache (cache takes precedence)
        all_data = dict(self.persistent_data)
        all_data.update(self.cache)
        return all_data

    def exists(self, key: str) -> bool:
        return key in self.cache or key in self.persistent_data

    def commit(self):
        """Write all cached data to the single consolidated file."""
        # Merge cache into persistent data
        self.persistent_data.update(self.cache)
        
        # Write everything to disk in one operation
        with open(self.data_file, 'wb') as f:
            pickle.dump(self.persistent_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        self.cache.clear()

class PostgresStore(DataStoreBase):
    def __init__(self, config: DictConfig, table_name: str):
        self.config = config
        self.table_name = table_name
        
        # Debug: print connection parameters
        print(f"  PostgreSQL connection: host={config.postgres.host}, port={config.postgres.port}, "
              f"db={config.postgres.database}, user={config.postgres.user}", flush=True)
        
        try:
            self.conn = psycopg2.connect(
                host=config.postgres.host,
                port=config.postgres.port,
                database=config.postgres.database,
                user=config.postgres.user,
                password=config.postgres.password
            )
        except Exception as e:
            print(f"  ERROR: Failed to connect to PostgreSQL: {e}", flush=True)
            raise
        
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
        # Buffer the put operations and commit in bulk in commit()
        # Remove NUL bytes from key (PostgreSQL TEXT can't contain them)
        key = key.replace('\x00', '')
        
        if isinstance(value, bytes):
            data_to_store = value
            is_json = False
        elif isinstance(value, str):
            data_to_store = value.encode('utf-8')
            is_json = True
        else:
            data_to_store = json.dumps(value).encode('utf-8')
            is_json = True

        # Store in-memory for batch commit
        if not hasattr(self, '_batch'):
            self._batch = []
        # Wrap bytes in psycopg2.Binary to handle NUL bytes properly
        self._batch.append((key, Binary(data_to_store), is_json))
    
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
        # Flush any buffered puts using executemany (handles binary data properly)
        if not hasattr(self, '_batch') or not self._batch:
            return

        sql = f"INSERT INTO {self.table_name} (key, value, is_json) VALUES (%s, %s, %s) " \
              f"ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, is_json = EXCLUDED.is_json"

        with self.conn.cursor() as cur:
            try:
                # executemany handles Binary() properly, unlike execute_values with template=None
                cur.executemany(sql, self._batch)
                self.conn.commit()
            except Exception:
                self.conn.rollback()
                raise
            finally:
                self._batch = []
    
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
