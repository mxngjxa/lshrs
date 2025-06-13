# src/lshrs/utils/dataloader.py

class LSHDataLoader:
    """
    Space-efficient dataloader for LSH recommendation system.
    Stores arbitrary indices, raw text, vectorized representations, and metadata.
    """

    def __init__(self, encoding_method='tfidf', compression_level=6, lazy_loading=True):
        # Core data storage - using memory-efficient structures
        self._indices = []  # Original arbitrary indices
        self._id_mapping = {}  # Maps arbitrary index -> internal sequential ID
        self._raw_texts = []  # Compressed raw text storage
        self._vectors = None  # Sparse matrix for vectorized representations
        self._metadata = {}  # Additional item metadata

        # Configuration
        self.encoding_method = encoding_method  # 'tfidf', 'embedding', 'onehot'
        self.compression_level = compression_level
        self.lazy_loading = lazy_loading

        # Lazy loading components
        self._encoder = None
        self._preprocessor = None
        self._is_fitted = False

        # Memory optimization flags
        self._compress_text = True
        self._use_sparse_vectors = True
        self._batch_size = 1000

    def add_items(self, items_data):
        """
        Add items to the dataloader efficiently.
        items_data: List of dicts with keys: 'index', 'text', 'metadata'
        """
        for item in items_data:
            internal_id = len(self._indices)
            arbitrary_idx = item['index']

            # Store index mapping
            self._id_mapping[arbitrary_idx] = internal_id
            self._indices.append(arbitrary_idx)

            # Compress and store raw text
            if self._compress_text:
                compressed_text = self._compress_text_data(item['text'])
                self._raw_texts.append(compressed_text)
            else:
                self._raw_texts.append(item['text'])

            # Store metadata efficiently
            if 'metadata' in item:
                self._metadata[internal_id] = item['metadata']

    def _compress_text_data(self, text):
        """Compress text data to save memory"""
        import zlib
        return zlib.compress(text.encode('utf-8'), self.compression_level)

    def _decompress_text_data(self, compressed_text):
        """Decompress text data when needed"""
        import zlib
        return zlib.decompress(compressed_text).decode('utf-8')

    def fit_encoder(self):
        """Fit the encoding method on all text data"""
        if self._is_fitted:
            return

        # Initialize encoder based on method
        if self.encoding_method == 'tfidf':
            from ..encoding.tfidf import TFIDFEncoder
            self._encoder = TFIDFEncoder()
        elif self.encoding_method == 'embedding':
            from ..encoding.embedding import EmbeddingEncoder
            self._encoder = EmbeddingEncoder()
        elif self.encoding_method == 'onehot':
            from ..encoding.onehot import OneHotEncoder
            self._encoder = OneHotEncoder()

        # Initialize preprocessor
        from ..preprocessing.main import TextPreprocessor
        self._preprocessor = TextPreprocessor()

        # Process texts in batches to manage memory
        all_texts = self._get_all_raw_texts()
        processed_texts = []

        for i in range(0, len(all_texts), self._batch_size):
            batch_texts = all_texts[i:i + self._batch_size]
            # Apply lemmatization and preprocessing
            preprocessed_batch = [
                self._preprocessor.lemmatize_text(text)
                for text in batch_texts
            ]
            processed_texts.extend(preprocessed_batch)

        # Fit encoder on all preprocessed text
        self._encoder.fit(processed_texts)

        # Generate vectorized representations
        if self._use_sparse_vectors:
            from scipy.sparse import csr_matrix, vstack
            vector_batches = []

            for i in range(0, len(processed_texts), self._batch_size):
                batch = processed_texts[i:i + self._batch_size]
                batch_vectors = self._encoder.transform(batch)
                vector_batches.append(csr_matrix(batch_vectors))

            self._vectors = vstack(vector_batches)
        else:
            self._vectors = self._encoder.transform(processed_texts)

        self._is_fitted = True

    def _get_all_raw_texts(self):
        """Retrieve all raw texts, decompressing if necessary"""
        if self._compress_text:
            return [
                self._decompress_text_data(compressed_text)
                for compressed_text in self._raw_texts
            ]
        else:
            return self._raw_texts

    def get_vector_by_index(self, arbitrary_index):
        """Get vectorized representation by arbitrary index"""
        if not self._is_fitted:
            self.fit_encoder()

        internal_id = self._id_mapping[arbitrary_index]
        return self._vectors[internal_id]

    def get_text_by_index(self, arbitrary_index):
        """Get raw text by arbitrary index"""
        internal_id = self._id_mapping[arbitrary_index]

        if self._compress_text:
            return self._decompress_text_data(self._raw_texts[internal_id])
        else:
            return self._raw_texts[internal_id]

    def get_metadata_by_index(self, arbitrary_index):
        """Get metadata by arbitrary index"""
        internal_id = self._id_mapping[arbitrary_index]
        return self._metadata.get(internal_id, {})

    def get_vectors_batch(self, arbitrary_indices):
        """Get multiple vectors efficiently"""
        if not self._is_fitted:
            self.fit_encoder()

        internal_ids = [self._id_mapping[idx] for idx in arbitrary_indices]
        return self._vectors[internal_ids]

    def get_all_vectors(self):
        """Get all vectorized representations"""
        if not self._is_fitted:
            self.fit_encoder()
        return self._vectors

    def save_to_disk(self, filepath: str) -> None:
        """Save dataloader state to disk for persistence."""
        from lshrc.utils.save import save_dataloader_state
        save_dataloader_state(self, filepath)

    def load_from_disk(self, filepath: str) -> "LSHDataLoader":
        """Load dataloader state from disk."""
        from lshrc.utils.save import load_dataloader_state
        return load_dataloader_state(filepath)

    def get_memory_usage(self) -> dict:
        """Get memory usage statistics"""
        import sys
        return {
            'indices_size': sys.getsizeof(self._indices),
            'mapping_size': sys.getsizeof(self._id_mapping),
            'texts_size': sys.getsizeof(self._raw_texts),
            'vectors_size': (
                self._vectors.data.nbytes
                if hasattr(self._vectors, 'data')
                else sys.getsizeof(self._vectors)
            ),
            'metadata_size': sys.getsizeof(self._metadata),
            'total_items': len(self._indices)
        }

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, arbitrary_index):
        """Enable direct access by arbitrary index"""
        return {
            'index': arbitrary_index,
            'text': self.get_text_by_index(arbitrary_index),
            'vector': self.get_vector_by_index(arbitrary_index),
            'metadata': self.get_metadata_by_index(arbitrary_index)
        }
