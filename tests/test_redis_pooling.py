# import pytest
# import redis
from unittest.mock import MagicMock, patch

from lshrs import LSHRS
from lshrs.storage.redis import RedisStorage


def test_redis_storage_uses_connection_pool():
    """Test that RedisStorage initializes a connection pool with correct parameters."""
    with patch("lshrs.storage.redis.redis.ConnectionPool") as mock_pool_cls:
        with patch("lshrs.storage.redis.redis.Redis") as mock_redis_cls:
            storage = RedisStorage(
                host="localhost",
                port=6379,
                max_connections=100,
                prefix="test"
            )
            
            # Check ConnectionPool initialization
            mock_pool_cls.assert_called_once()
            _, kwargs = mock_pool_cls.call_args
            assert kwargs["host"] == "localhost"
            assert kwargs["port"] == 6379
            assert kwargs["max_connections"] == 100
            
            # Check Redis client initialization using pool
            mock_redis_cls.assert_called_once_with(connection_pool=mock_pool_cls.return_value)
            
            # Check close disconnects pool
            mock_pool_instance = mock_pool_cls.return_value
            storage.close()
            mock_pool_instance.disconnect.assert_called_once()


def test_lshrs_passes_max_connections():
    """Test that LSHRS passes max_connections to RedisStorage."""
    with patch("lshrs.core.main.RedisStorage") as mock_storage_cls:
        
        # Check RedisStorage was called with max_connections
        _, kwargs = mock_storage_cls.call_args
        assert kwargs["max_connections"] == 75


def test_lshrs_context_manager():
    """Test that LSHRS context manager closes resources."""
    mock_storage = MagicMock(spec=RedisStorage)
    
    with LSHRS(
        dim=64,
        num_bands=4,
        rows_per_band=4,
        num_perm=16,
        storage=mock_storage
    ) as lsh:
        assert isinstance(lsh, LSHRS)
        # Simulate some buffer usage
        lsh._buffer.append((0, b'hash', 1))
        
    # Verify close was called on exit
    # This should trigger flush_buffer and storage.close
    mock_storage.batch_add.assert_called_once()
    mock_storage.close.assert_called_once()


def test_lshrs_close_flushes_buffer():
    """Test that calling close() manually flushes the buffer."""
    mock_storage = MagicMock(spec=RedisStorage)
    lsh = LSHRS(
        dim=64,
        num_bands=4,
        rows_per_band=4,
        num_perm=16,
        storage=mock_storage,
        buffer_size=100
    )
    
    # Add item to buffer
    lsh._buffer.append((0, b'hash', 1))
    
    lsh.close()
    
    mock_storage.batch_add.assert_called_once()
    mock_storage.close.assert_called_once()