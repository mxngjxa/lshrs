# src/lshrs/utils/save.py

"""
Comprehensive save/load system for LSH recommendation system.
Handles packaging and restoration of all system components.
"""

import json
import logging
import os
import pickle
import tarfile
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
from scipy.sparse import load_npz, save_npz

from ..core.config import RecommenderConfig
from ..core.exceptions import PersistenceError

logger = logging.getLogger(__name__)

class LSHSystemSaver:
    """
    Comprehensive system for saving and loading LSH recommendation system state.
    Creates compressed archives containing all necessary components.
    """

    def __init__(self, compression_level: int = 6, use_joblib: bool = True):
        self.compression_level = compression_level
        self.use_joblib = use_joblib  # joblib is better for numpy arrays [10]
        self.manifest_version = "1.0"

    def save_complete_system(
        self,
        filepath: str,
        recommender_instance: Any,
        dataloader_instance: Any = None,
        metadata: Dict[str, Any] = None
    ) -> None:
        """
        Save complete LSH recommendation system to compressed archive.

        Args:
            filepath: Path for the output archive (.tar.gz)
            recommender_instance: Main recommender object
            dataloader_instance: Optional dataloader object
            metadata: Additional metadata to save
        """
        try:
            # Create temporary directory structure
            temp_dir = Path(filepath).parent / f"temp_save_{os.getpid()}"
            temp_dir.mkdir(exist_ok=True)

            # Save all components
            manifest = self._create_manifest(
                recommender_instance, dataloader_instance, metadata
            )

            # Save core components
            self._save_config(temp_dir, recommender_instance.config)
            self._save_pipeline_components(temp_dir, recommender_instance.pipeline)

            # Save dataloader if provided
            if dataloader_instance:
                self._save_dataloader(temp_dir, dataloader_instance)

            # Save raw data and indices
            self._save_data_components(temp_dir, recommender_instance.pipeline)

            # Save LSH structures
            self._save_lsh_structures(temp_dir, recommender_instance.pipeline)

            # Save manifest
            self._save_manifest(temp_dir, manifest)

            # Create compressed archive
            self._create_archive(filepath, temp_dir)

            # Cleanup
            self._cleanup_temp_dir(temp_dir)

            logger.info(f"System saved successfully to {filepath}")

        except Exception as e:
            logger.error(f"Failed to save system: {str(e)}")
            raise PersistenceError(f"System save failed: {str(e)}")

    def _create_manifest(
        self,
        recommender_instance: Any,
        dataloader_instance: Any,
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create manifest with system information."""
        manifest = {
            "version": self.manifest_version,
            "timestamp": str(np.datetime64('now')),
            "components": {
                "config": True,
                "encoder": True,
                "hasher": True,
                "similarity": True,
                "dataloader": dataloader_instance is not None,
                "raw_data": True,
                "encoded_data": True,
                "signatures": True,
                "hash_buckets": True
            },
            "config_info": {
                "encoding_type": (
                    recommender_instance.config.encoding_config.encoding_type
                ),
                "lsh_type": recommender_instance.config.lsh_config.hash_type,
                "num_bands": recommender_instance.config.lsh_config.num_bands,
                "rows_per_band": recommender_instance.config.lsh_config.rows_per_band
            },
            "data_info": {
                "num_items": (
                    len(recommender_instance.pipeline.item_index)
                    if recommender_instance.pipeline.item_index
                    else 0
                ),
                "encoding_shape": (
                    recommender_instance.pipeline.encoded_data.shape
                    if hasattr(recommender_instance.pipeline.encoded_data, "shape")
                    else None
                ),
                "signature_shape": (
                    recommender_instance.pipeline.signatures.shape
                    if hasattr(recommender_instance.pipeline.signatures, "shape")
                    else None
                ),
            },
            "metadata": metadata or {}
        }
        return manifest

    def _save_config(self, temp_dir: Path, config: RecommenderConfig) -> None:
        """Save configuration to JSON.""" [9]
        config_path = temp_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config.dict(), f, indent=2, default=str)

    def _save_pipeline_components(self, temp_dir: Path, pipeline: Any) -> None:
        """Save encoder, hasher, and similarity components.""" [10]
        components_dir = temp_dir / "components"
        components_dir.mkdir(exist_ok=True)

        # Save encoder
        if pipeline.encoder:
            encoder_path = components_dir / "encoder.joblib"
            if self.use_joblib:
                joblib.dump(pipeline.encoder, encoder_path)
            else:
                with open(encoder_path.with_suffix('.pkl'), 'wb') as f:
                    pickle.dump(pipeline.encoder, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Save hasher
        if pipeline.hasher:
            hasher_path = components_dir / "hasher.joblib"
            if self.use_joblib:
                joblib.dump(pipeline.hasher, hasher_path)
            else:
                with open(hasher_path.with_suffix('.pkl'), 'wb') as f:
                    pickle.dump(pipeline.hasher, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Save similarity calculator
        if pipeline.similarity:
            similarity_path = components_dir / "similarity.joblib"
            if self.use_joblib:
                joblib.dump(pipeline.similarity, similarity_path)
            else:
                with open(similarity_path.with_suffix('.pkl'), 'wb') as f:
                    pickle.dump(
                        pipeline.similarity, f, protocol=pickle.HIGHEST_PROTOCOL
                    )

    def _save_dataloader(self, temp_dir: Path, dataloader: Any) -> None:
        """Save dataloader state efficiently.""" [7]
        dataloader_dir = temp_dir / "dataloader"
        dataloader_dir.mkdir(exist_ok=True)

        # Save indices and mappings
        indices_path = dataloader_dir / "indices.json"
        with open(indices_path, 'w') as f:
            json.dump({
                "indices": dataloader._indices,
                "id_mapping": dataloader._id_mapping,
                "metadata": dataloader._metadata
            }, f, indent=2, default=str)

        # Save compressed raw texts
        texts_path = dataloader_dir / "raw_texts.pkl"
        with open(texts_path, 'wb') as f:
            pickle.dump(dataloader._raw_texts, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Save vectorized data if available
        if dataloader._vectors is not None:
            vectors_path = dataloader_dir / "vectors.npz"
            if hasattr(dataloader._vectors, 'toarray'):  # sparse matrix
                save_npz(vectors_path, dataloader._vectors)
            else:  # dense array
                np.savez_compressed(vectors_path, vectors=dataloader._vectors)

        # Save dataloader configuration
        config_path = dataloader_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump({
                "encoding_method": dataloader.encoding_method,
                "compression_level": dataloader.compression_level,
                "lazy_loading": dataloader.lazy_loading,
                "compress_text": dataloader._compress_text,
                "use_sparse_vectors": dataloader._use_sparse_vectors,
                "batch_size": dataloader._batch_size,
                "is_fitted": dataloader._is_fitted
            }, f, indent=2)

    def _save_data_components(self, temp_dir: Path, pipeline: Any) -> None:
        """Save data components (raw data, encoded data, item indices)."""
        data_dir = temp_dir / "data"
        data_dir.mkdir(exist_ok=True)

        # Save raw item data
        if pipeline.item_data is not None:
            item_data_path = data_dir / "item_data.pkl"
            with open(item_data_path, 'wb') as f:
                pickle.dump(pipeline.item_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Save item indices
        if pipeline.item_index is not None:
            index_path = data_dir / "item_index.json"
            with open(index_path, 'w') as f:
                json.dump(pipeline.item_index, f, indent=2, default=str)

        # Save encoded data
        if pipeline.encoded_data is not None:
            encoded_path = data_dir / "encoded_data.npz"
            if hasattr(pipeline.encoded_data, 'toarray'):  # sparse matrix [10]
                save_npz(encoded_path, pipeline.encoded_data)
            else:  # dense array
                np.savez_compressed(encoded_path, encoded_data=pipeline.encoded_data)

    def _save_lsh_structures(self, temp_dir: Path, pipeline: Any) -> None:
        """Save LSH signatures and hash bucket structures."""
        lsh_dir = temp_dir / "lsh"
        lsh_dir.mkdir(exist_ok=True)

        # Save signatures
        if pipeline.signatures is not None:
            signatures_path = lsh_dir / "signatures.npz"
            np.savez_compressed(signatures_path, signatures=pipeline.signatures)

        # Save hash buckets if available
        if hasattr(pipeline.hasher, 'buckets') and pipeline.hasher.buckets:
            buckets_path = lsh_dir / "hash_buckets.pkl"
            with open(buckets_path, 'wb') as f:
                pickle.dump(
                    pipeline.hasher.buckets, f, protocol=pickle.HIGHEST_PROTOCOL
                )

        # Save reverse bucket mapping if available
        if (
            hasattr(pipeline.hasher, "reverse_buckets")
            and pipeline.hasher.reverse_buckets
        ):
            reverse_buckets_path = lsh_dir / "reverse_buckets.pkl"
            with open(reverse_buckets_path, 'wb') as f:
                pickle.dump(
                    pipeline.hasher.reverse_buckets,
                    f,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )

    def _save_manifest(self, temp_dir: Path, manifest: Dict[str, Any]) -> None:
        """Save system manifest."""
        manifest_path = temp_dir / "manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2, default=str)

    def _create_archive(self, filepath: str, temp_dir: Path) -> None:
        """Create compressed tar archive.""" [13]
        with tarfile.open(
            filepath, "w:gz", compresslevel=self.compression_level
        ) as tar:
            tar.add(temp_dir, arcname="lsh_system")

    def _cleanup_temp_dir(self, temp_dir: Path) -> None:
        """Clean up temporary directory."""
        import shutil
        shutil.rmtree(temp_dir)

class LSHSystemLoader:
    """
    System for loading LSH recommendation system from saved archives.
    """

    def __init__(self):
        self.temp_dir = None

    def load_complete_system(
        self,
        filepath: str,
        recreate_components: bool = True
    ) -> Dict[str, Any]:
        """
        Load complete LSH recommendation system from archive.

        Args:
            filepath: Path to the saved archive
            recreate_components: Whether to recreate component instances

        Returns:
            Dictionary containing all loaded components
        """
        try:
            # Extract archive
            self.temp_dir = self._extract_archive(filepath)

            # Load and validate manifest
            manifest = self._load_manifest()

            # Load configuration
            config = self._load_config()

            # Load pipeline components
            components = self._load_pipeline_components()

            # Load data components
            data_components = self._load_data_components()

            # Load LSH structures
            lsh_structures = self._load_lsh_structures()

            # Load dataloader if available
            dataloader = None
            if manifest["components"]["dataloader"]:
                dataloader = self._load_dataloader()

            # Recreate system if requested
            if recreate_components:
                system = self._recreate_system(
                    config, components, data_components, lsh_structures, dataloader
                )
            else:
                system = {
                    "config": config,
                    "components": components,
                    "data": data_components,
                    "lsh": lsh_structures,
                    "dataloader": dataloader,
                    "manifest": manifest
                }

            # Cleanup
            self._cleanup_temp_dir()

            logger.info(f"System loaded successfully from {filepath}")
            return system

        except Exception as e:
            logger.error(f"Failed to load system: {str(e)}")
            if self.temp_dir:
                self._cleanup_temp_dir()
            raise PersistenceError(f"System load failed: {str(e)}")

    def _extract_archive(self, filepath: str) -> Path:
        """Extract archive to temporary directory.""" [13]
        temp_dir = Path(filepath).parent / f"temp_load_{os.getpid()}"
        temp_dir.mkdir(exist_ok=True)

        with tarfile.open(filepath, "r:gz") as tar:
            tar.extractall(temp_dir)

        return temp_dir / "lsh_system"

    def _load_manifest(self) -> Dict[str, Any]:
        """Load and validate manifest."""
        manifest_path = self.temp_dir / "manifest.json"
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)

        # Validate manifest version
        if manifest.get("version") != "1.0":
            raise PersistenceError(
                f"Unsupported manifest version: {manifest.get('version')}"
            )

        return manifest

    def _load_config(self) -> RecommenderConfig:
        """Load configuration."""
        config_path = self.temp_dir / "config.json"
        with open(config_path, 'r') as f:
            config_dict = json.load(f)

        return RecommenderConfig.from_dict(config_dict)

    def _load_pipeline_components(self) -> Dict[str, Any]:
        """Load pipeline components.""" [10]
        components_dir = self.temp_dir / "components"
        components = {}

        # Load encoder
        encoder_path = components_dir / "encoder.joblib"
        if encoder_path.exists():
            components["encoder"] = joblib.load(encoder_path)
        else:
            encoder_path = components_dir / "encoder.pkl"
            if encoder_path.exists():
                with open(encoder_path, 'rb') as f:
                    components["encoder"] = pickle.load(f)

        # Load hasher
        hasher_path = components_dir / "hasher.joblib"
        if hasher_path.exists():
            components["hasher"] = joblib.load(hasher_path)
        else:
            hasher_path = components_dir / "hasher.pkl"
            if hasher_path.exists():
                with open(hasher_path, 'rb') as f:
                    components["hasher"] = pickle.load(f)

        # Load similarity
        similarity_path = components_dir / "similarity.joblib"
        if similarity_path.exists():
            components["similarity"] = joblib.load(similarity_path)
        else:
            similarity_path = components_dir / "similarity.pkl"
            if similarity_path.exists():
                with open(similarity_path, 'rb') as f:
                    components["similarity"] = pickle.load(f)

        return components

    def _load_data_components(self) -> Dict[str, Any]:
        """Load data components."""
        data_dir = self.temp_dir / "data"
        data_components = {}

        # Load raw item data
        item_data_path = data_dir / "item_data.pkl"
        if item_data_path.exists():
            with open(item_data_path, 'rb') as f:
                data_components["item_data"] = pickle.load(f)

        # Load item index
        index_path = data_dir / "item_index.json"
        if index_path.exists():
            with open(index_path, 'r') as f:
                data_components["item_index"] = json.load(f)

        # Load encoded data
        encoded_path = data_dir / "encoded_data.npz"
        if encoded_path.exists():
            try:
                # Try loading as sparse matrix first
                data_components["encoded_data"] = load_npz(encoded_path)
            except Exception:
                # Fall back to dense array
                loaded = np.load(encoded_path)
                data_components["encoded_data"] = loaded["encoded_data"]

        return data_components

    def _load_lsh_structures(self) -> Dict[str, Any]:
        """Load LSH structures."""
        lsh_dir = self.temp_dir / "lsh"
        lsh_structures = {}

        # Load signatures
        signatures_path = lsh_dir / "signatures.npz"
        if signatures_path.exists():
            loaded = np.load(signatures_path)
            lsh_structures["signatures"] = loaded["signatures"]

        # Load hash buckets
        buckets_path = lsh_dir / "hash_buckets.pkl"
        if buckets_path.exists():
            with open(buckets_path, 'rb') as f:
                lsh_structures["buckets"] = pickle.load(f)

        # Load reverse buckets
        reverse_buckets_path = lsh_dir / "reverse_buckets.pkl"
        if reverse_buckets_path.exists():
            with open(reverse_buckets_path, 'rb') as f:
                lsh_structures["reverse_buckets"] = pickle.load(f)

        return lsh_structures

    def _load_dataloader(self) -> Any:
        """Load dataloader state."""
        from .dataloader import LSHDataLoader

        dataloader_dir = self.temp_dir / "dataloader"

        # Load configuration
        config_path = dataloader_dir / "config.json"
        with open(config_path, 'r') as f:
            dl_config = json.load(f)

        # Create new dataloader instance
        dataloader = LSHDataLoader(
            encoding_method=dl_config["encoding_method"],
            compression_level=dl_config["compression_level"],
            lazy_loading=dl_config["lazy_loading"]
        )

        # Restore internal state
        dataloader._compress_text = dl_config["compress_text"]
        dataloader._use_sparse_vectors = dl_config["use_sparse_vectors"]
        dataloader._batch_size = dl_config["batch_size"]
        dataloader._is_fitted = dl_config["is_fitted"]

        # Load indices and mappings
        indices_path = dataloader_dir / "indices.json"
        with open(indices_path, 'r') as f:
            indices_data = json.load(f)

        dataloader._indices = indices_data["indices"]
        dataloader._id_mapping = indices_data["id_mapping"]
        dataloader._metadata = indices_data["metadata"]

        # Load raw texts
        texts_path = dataloader_dir / "raw_texts.pkl"
        with open(texts_path, 'rb') as f:
            dataloader._raw_texts = pickle.load(f)

        # Load vectors if available
        vectors_path = dataloader_dir / "vectors.npz"
        if vectors_path.exists():
            try:
                dataloader._vectors = load_npz(vectors_path)
            except Exception:
                loaded = np.load(vectors_path)
                dataloader._vectors = loaded["vectors"]

        return dataloader

    def _recreate_system(
        self,
        config: RecommenderConfig,
        components: Dict[str, Any],
        data_components: Dict[str, Any],
        lsh_structures: Dict[str, Any],
        dataloader: Any
    ) -> Dict[str, Any]:
        """Recreate complete system from loaded components."""
        from ..core.main import LSHRecommender

        # Create recommender instance
        recommender = LSHRecommender(config)

        # Set up pipeline components
        recommender.pipeline.encoder = components.get("encoder")
        recommender.pipeline.hasher = components.get("hasher")
        recommender.pipeline.similarity = components.get("similarity")

        # Restore data
        recommender.pipeline.item_data = data_components.get("item_data")
        recommender.pipeline.item_index = data_components.get("item_index")
        recommender.pipeline.encoded_data = data_components.get("encoded_data")

        # Restore LSH structures
        recommender.pipeline.signatures = lsh_structures.get("signatures")
        if "buckets" in lsh_structures:
            recommender.pipeline.hasher.buckets = lsh_structures["buckets"]
        if "reverse_buckets" in lsh_structures:
            recommender.pipeline.hasher.reverse_buckets = lsh_structures[
                "reverse_buckets"
            ]

        # Mark as fitted
        recommender.pipeline.is_fitted = True

        return {
            "recommender": recommender,
            "dataloader": dataloader,
            "config": config
        }

    def _cleanup_temp_dir(self) -> None:
        """Clean up temporary directory."""
        if self.temp_dir and self.temp_dir.exists():
            import shutil
            shutil.rmtree(self.temp_dir.parent)

# Convenience functions
def save_lsh_system(
    filepath: str,
    recommender_instance: Any,
    dataloader_instance: Any = None,
    metadata: Dict[str, Any] = None,
    compression_level: int = 6
) -> None:
    """
    Convenience function to save LSH system.

    Args:
        filepath: Output archive path (.tar.gz)
        recommender_instance: Main recommender object
        dataloader_instance: Optional dataloader object
        metadata: Additional metadata
        compression_level: Compression level (1-9)
    """
    saver = LSHSystemSaver(compression_level=compression_level)
    saver.save_complete_system(
        filepath, recommender_instance, dataloader_instance, metadata
    )

def load_lsh_system(filepath: str, recreate_components: bool = True) -> Dict[str, Any]:
    """
    Convenience function to load LSH system.

    Args:
        filepath: Path to saved archive
        recreate_components: Whether to recreate component instances

    Returns:
        Dictionary containing loaded system components
    """
    loader = LSHSystemLoader()
    return loader.load_complete_system(filepath, recreate_components)

# Updated dataloader integration
def save_dataloader_state(dataloader: Any, filepath: str) -> None:
    """Save dataloader state to individual file."""
    saver = LSHSystemSaver()
    temp_dir = Path(filepath).parent / f"temp_dl_{os.getpid()}"
    temp_dir.mkdir(exist_ok=True)

    try:
        saver._save_dataloader(temp_dir, dataloader)
        saver._create_archive(filepath, temp_dir)
        saver._cleanup_temp_dir(temp_dir)
    except Exception:
        saver._cleanup_temp_dir(temp_dir)
        raise

def load_dataloader_state(filepath: str) -> Any:
    """Load dataloader state from individual file."""
    loader = LSHSystemLoader()
    temp_dir = loader._extract_archive(filepath)
    loader.temp_dir = temp_dir

    try:
        dataloader = loader._load_dataloader()
        loader._cleanup_temp_dir()
        return dataloader
    except Exception:
        loader._cleanup_temp_dir()
        raise
