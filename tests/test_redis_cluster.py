"""Tests for Redis Cluster support with band-aware hash-tag sharding.

These tests use mocking so no live Redis Cluster is required. They cover:
- single-node key format is unchanged (regression),
- cluster-mode bucket keys include the {band_id} hash tag,
- the storage constructs a RedisCluster client when cluster=True,
- member_key hash-tag format,
- LSHRS threads cluster options through to RedisStorage.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from lshrs import LSHRS
from lshrs.storage.redis import RedisStorage


def _single_node_storage(prefix: str = "lsh") -> RedisStorage:
    """Build a single-node RedisStorage without touching a real Redis."""
    with (
        patch("lshrs.storage.redis.redis.ConnectionPool"),
        patch("lshrs.storage.redis.redis.Redis"),
    ):
        return RedisStorage(prefix=prefix)


def _cluster_storage(prefix: str = "lsh", **kwargs) -> RedisStorage:
    """Build a cluster RedisStorage with a mocked RedisCluster client."""
    with patch("lshrs.storage.redis.redis.cluster.RedisCluster"):
        return RedisStorage(prefix=prefix, cluster=True, **kwargs)


# ---------------------------------------------------------------------------
# Key-format regression / cluster behavior
# ---------------------------------------------------------------------------


def test_single_node_bucket_key_format_unchanged():
    """Single-node mode keeps the legacy flat key format (no hash tag)."""
    storage = _single_node_storage(prefix="search")
    key = storage.bucket_key(band_id=5, hash_val=b"\xab\xcd\xef")
    assert key == "search:5:bucket:abcdef"
    assert "{" not in key  # no hash tag in single-node mode


def test_cluster_bucket_key_has_band_hash_tag():
    """Cluster mode wraps the band id in a {} hash tag for slot co-location."""
    storage = _cluster_storage(prefix="search")
    key = storage.bucket_key(band_id=5, hash_val=b"\xab\xcd\xef")
    assert key == "search:{5}:bucket:abcdef"
    # The hash tag must contain exactly the band id.
    assert "{5}" in key


def test_cluster_buckets_of_same_band_share_hash_tag():
    """All buckets in a band share the same hash tag (same slot/node)."""
    storage = _cluster_storage(prefix="lsh")
    k1 = storage.bucket_key(band_id=3, hash_val=b"\x01")
    k2 = storage.bucket_key(band_id=3, hash_val=b"\x02\x03")
    # Same band -> identical {tag}, so identical cluster slot.
    assert k1.split(":bucket:")[0] == k2.split(":bucket:")[0] == "lsh:{3}"


def test_single_node_member_key_format():
    """Single-node member key has no hash tag."""
    storage = _single_node_storage(prefix="search")
    assert storage.member_key(42) == "search:member:42"


def test_cluster_member_key_has_hash_tag():
    """Cluster member key wraps the index in a {} hash tag."""
    storage = _cluster_storage(prefix="search")
    assert storage.member_key(42) == "search:member:{42}"


# ---------------------------------------------------------------------------
# Client construction
# ---------------------------------------------------------------------------


def test_cluster_constructs_rediscluster_client():
    """cluster=True builds a RedisCluster client, not redis.Redis+pool."""
    with (
        patch("lshrs.storage.redis.redis.cluster.RedisCluster") as mock_cluster_cls,
        patch("lshrs.storage.redis.redis.ConnectionPool") as mock_pool_cls,
        patch("lshrs.storage.redis.redis.Redis") as mock_redis_cls,
    ):
        storage = RedisStorage(host="node1", port=7000, cluster=True, prefix="c")

        mock_cluster_cls.assert_called_once()
        # Single-node primitives must NOT be used in cluster mode.
        mock_pool_cls.assert_not_called()
        mock_redis_cls.assert_not_called()

        assert storage.cluster is True
        assert storage._pool is None
        assert storage.client is mock_cluster_cls.return_value


def test_cluster_default_seed_node_from_host_port():
    """With no startup_nodes, the single host/port becomes the seed node."""
    with (
        patch("lshrs.storage.redis.redis.cluster.RedisCluster") as mock_cluster_cls,
        patch("lshrs.storage.redis.redis.cluster.ClusterNode") as mock_node_cls,
    ):
        RedisStorage(host="seed-host", port=7777, cluster=True)

        mock_node_cls.assert_called_once_with("seed-host", 7777)
        _, kwargs = mock_cluster_cls.call_args
        assert kwargs["startup_nodes"] == [mock_node_cls.return_value]


def test_cluster_startup_nodes_tuples_and_mappings():
    """startup_nodes accepts (host, port) tuples and {host, port} mappings."""
    with (
        patch("lshrs.storage.redis.redis.cluster.RedisCluster"),
        patch("lshrs.storage.redis.redis.cluster.ClusterNode") as mock_node_cls,
    ):
        RedisStorage(
            cluster=True,
            startup_nodes=[("a", 7000), {"host": "b", "port": 7001}],
        )
        calls = [c.args for c in mock_node_cls.call_args_list]
        assert ("a", 7000) in calls
        assert ("b", 7001) in calls


def test_single_node_mode_unchanged_construction():
    """Default (cluster=False) still uses ConnectionPool + redis.Redis."""
    with (
        patch("lshrs.storage.redis.redis.ConnectionPool") as mock_pool_cls,
        patch("lshrs.storage.redis.redis.Redis") as mock_redis_cls,
        patch("lshrs.storage.redis.redis.cluster.RedisCluster") as mock_cluster_cls,
    ):
        storage = RedisStorage(prefix="x")

        mock_pool_cls.assert_called_once()
        mock_redis_cls.assert_called_once_with(connection_pool=mock_pool_cls.return_value)
        mock_cluster_cls.assert_not_called()
        assert storage.cluster is False


def test_cluster_close_closes_client():
    """close() in cluster mode closes the RedisCluster client (no pool)."""
    with patch("lshrs.storage.redis.redis.cluster.RedisCluster") as mock_cluster_cls:
        storage = RedisStorage(cluster=True)
        storage.close()
        mock_cluster_cls.return_value.close.assert_called_once()


# ---------------------------------------------------------------------------
# Cluster-safe maintenance operations
# ---------------------------------------------------------------------------


def test_cluster_clear_uses_pipeline_per_key():
    """clear() in cluster mode deletes per-key via pipeline (avoids CROSSSLOT)."""
    with patch("lshrs.storage.redis.redis.cluster.RedisCluster") as mock_cluster_cls:
        storage = RedisStorage(cluster=True, prefix="c")
        client = mock_cluster_cls.return_value
        client.scan_iter.return_value = iter(["c:{0}:bucket:aa", "c:{1}:bucket:bb"])
        pipe = MagicMock()
        client.pipeline.return_value = pipe

        storage.clear()

        # Each key deleted individually through the cluster pipeline.
        assert pipe.delete.call_count == 2
        # A single multi-key delete must NOT be used in cluster mode.
        client.delete.assert_not_called()


def test_cluster_bucket_and_member_count():
    """bucket_count / member_count scan the keyspace under cluster mode."""
    with patch("lshrs.storage.redis.redis.cluster.RedisCluster") as mock_cluster_cls:
        storage = RedisStorage(cluster=True, prefix="c")
        client = mock_cluster_cls.return_value
        client.scan_iter.side_effect = [
            iter(["c:{0}:bucket:aa", "c:{1}:bucket:bb", "c:{2}:bucket:cc"]),
            iter(["c:member:{1}", "c:member:{2}"]),
        ]
        assert storage.bucket_count() == 3
        assert storage.member_count() == 2


# ---------------------------------------------------------------------------
# LSHRS threading of cluster options
# ---------------------------------------------------------------------------


def test_lshrs_passes_cluster_options_to_storage():
    """LSHRS forwards redis_cluster / redis_startup_nodes to RedisStorage."""
    with patch("lshrs.core.main.RedisStorage") as mock_storage_cls:
        LSHRS(
            dim=64,
            redis_cluster=True,
            redis_startup_nodes=[("a", 7000)],
        )
        _, kwargs = mock_storage_cls.call_args
        assert kwargs["cluster"] is True
        assert kwargs["startup_nodes"] == [("a", 7000)]


def test_lshrs_stats_reports_cluster_flag():
    """stats() includes the redis_cluster flag (additive)."""
    mock_storage = MagicMock(spec=RedisStorage)
    lsh = LSHRS(
        dim=64,
        num_bands=4,
        rows_per_band=4,
        num_perm=16,
        storage=mock_storage,
        redis_cluster=True,
    )
    assert lsh.stats()["redis_cluster"] is True
