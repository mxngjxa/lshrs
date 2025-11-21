# import pytest
import numpy as np
from typing import List, Set, Tuple, Any
from lshrs import LSHRS
from lshrs.storage.redis import RedisStorage

class MockStorage(RedisStorage):
    def __init__(self):
        self.batches: List[Any] = []
        self.data: dict = {}

    def batch_add(self, operations: List[Tuple[int, bytes, int]]) -> None:
        self.batches.append(operations)
        for band_id, hash_val, index in operations:
            key = (band_id, hash_val)
            if key not in self.data:
                self.data[key] = set()
            self.data[key].add(index)

    def get_bucket(self, band_id: int, hash_val: bytes) -> Set[int]:
        return self.data.get((band_id, hash_val), set())
        
    def close(self) -> None:
        pass

def test_single_ingest_not_immediately_queryable():
    """Test that single ingestion is buffered and not immediately searchable."""
    storage = MockStorage()
    # Large buffer ensures no auto-flush
    lsh = LSHRS(dim=4, num_bands=2, rows_per_band=2, num_perm=4, buffer_size=100, storage=storage)
    
    vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    
    # Ingest one vector
    lsh.ingest(0, vec)
    
    # Buffer should not have flushed yet
    assert len(storage.batches) == 0
    
    # Query should return empty results because storage is empty
    results = lsh.query(vec, top_k=1)
    assert results == []
    
    # Manually flush
    lsh.flush()
    
    # Now storage should have data
    assert len(storage.batches) == 1
    
    # Query should work
    results = lsh.query(vec, top_k=1)
    assert results == [0]

def test_batch_index_auto_flushes():
    """Test that batch indexing forces a flush at the end."""
    storage = MockStorage()
    lsh = LSHRS(dim=4, num_bands=2, rows_per_band=2, num_perm=4, buffer_size=100, storage=storage)
    
    vecs = np.eye(4, dtype=np.float32)
    indices = [0, 1, 2, 3]
    
    # Batch index
    lsh.index(indices, vecs)
    
    # Should have flushed automatically
    assert len(storage.batches) == 1
    # 4 vectors * 2 bands = 8 operations
    assert len(storage.batches[0]) == 8 
    
    # Immediately queryable
    results = lsh.query(vecs[0], top_k=1)
    assert results == [0]

def test_buffer_flush_on_full():
    """Test that buffer flushes automatically when capacity is reached."""
    storage = MockStorage()
    # buffer_size=4. Each vector adds 2 ops (2 bands).
    # 2 vectors = 4 ops, which should trigger flush.
    lsh = LSHRS(dim=4, num_bands=2, rows_per_band=2, num_perm=4, buffer_size=4, storage=storage)
    
    vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    
    # Ingest 1 (2 ops)
    lsh.ingest(0, vec)
    assert len(storage.batches) == 0
    
    # Ingest 2 (Total 4 ops) -> Should trigger flush
    lsh.ingest(1, vec)
    assert len(storage.batches) == 1
    assert len(storage.batches[0]) == 4

def test_close_flushes_buffer():
    """Test that closing the LSH instance flushes any pending operations."""
    storage = MockStorage()
    lsh = LSHRS(dim=4, num_bands=2, rows_per_band=2, num_perm=4, buffer_size=100, storage=storage)
    
    lsh.ingest(0, np.zeros(4, dtype=np.float32) + 1) # Non-zero vector
    assert len(storage.batches) == 0
    
    lsh.close()
    assert len(storage.batches) == 1