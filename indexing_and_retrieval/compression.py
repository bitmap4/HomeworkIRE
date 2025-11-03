import struct
from typing import List, Optional
import zstandard as zstd
from omegaconf import DictConfig
import json

class CompressionBase:
    def compress(self, data: List[int]) -> bytes:
        raise NotImplementedError
    
    def decompress(self, data: bytes) -> List[int]:
        raise NotImplementedError

    def compress_pl(self, data: dict) -> bytes:
        return json.dumps(data).encode('utf-8')

    def decompress_pl(self, data: bytes) -> dict:
        return json.loads(data.decode('utf-8'))

class NoCompression(CompressionBase):
    def compress(self, data: List[int]) -> bytes:
        return struct.pack(f'{len(data)}I', *data)
    
    def decompress(self, data: bytes) -> List[int]:
        num_ints = len(data) // 4
        return list(struct.unpack(f'{num_ints}I', data))

class VarByteCompression(CompressionBase):
    def compress(self, data: List[int]) -> bytes:
        result = bytearray()
        for num in data:
            while num >= 128:
                result.append((num & 0x7F) | 0x80)
                num >>= 7
            result.append(num & 0x7F)
        return bytes(result)
    
    def decompress(self, data: bytes) -> List[int]:
        result = []
        current = 0
        shift = 0
        
        for byte in data:
            if byte & 0x80:
                current |= (byte & 0x7F) << shift
                shift += 7
            else:
                current |= byte << shift
                result.append(current)
                current = 0
                shift = 0
        
        return result

class ZstdCompression(CompressionBase):
    def __init__(self, level: int = 3):
        self.level = level
        self.compressor = zstd.ZstdCompressor(level=level)
        self.decompressor = zstd.ZstdDecompressor()
    
    def compress(self, data: List[int]) -> bytes:
        raw_bytes = struct.pack(f'{len(data)}I', *data)
        return self.compressor.compress(raw_bytes)
    
    def decompress(self, data: bytes) -> List[int]:
        raw_bytes = self.decompressor.decompress(data)
        num_ints = len(raw_bytes) // 4
        return list(struct.unpack(f'{num_ints}I', raw_bytes))

    def compress_pl(self, data: dict) -> bytes:
        return self.compressor.compress(json.dumps(data).encode('utf-8'))
    
    def decompress_pl(self, data: bytes) -> dict:
        return json.loads(self.decompressor.decompress(data).decode('utf-8'))

def get_compressor(method: str, config: Optional[DictConfig] = None) -> CompressionBase:
    if method == 'NONE':
        return NoCompression()
    elif method == 'CODE':
        return VarByteCompression()
    elif method == 'CLIB':
        level = config.index.compression.zstd.level if config else 3
        return ZstdCompression(level=level)
    else:
        raise ValueError(f"Unknown compression method: {method}")
