from __future__ import annotations

import math

import numpy as np


class VisualVocabularyTree:
    """Hierarchical Bag-of-Visual-Words vocabulary tree for loop closure detection."""

    def __init__(self, k_branching: int = 6, depth: int = 3) -> None:
        self.k = k_branching
        self.depth = depth
        self.words: list[np.ndarray] = []
        self.idf_weights: np.ndarray = np.array([])
        self.inverted_index: dict[int, list[tuple[int, float]]] = {}  # word_id -> [(frame_id, tf)]
        self._n_documents = 0

    def fit_vocabulary(self, descriptors: np.ndarray) -> None:
        """Cluster descriptors into visual words using k-means hierarchy."""
        n_words = self.k ** self.depth
        n_samples = len(descriptors)
        if n_samples < n_words:
            self.words = [descriptors[i % n_samples].astype(np.float64) for i in range(n_words)]
        else:
            indices = np.random.choice(n_samples, n_words, replace=False)
            self.words = [descriptors[idx].astype(np.float64) for idx in indices]
        self.idf_weights = np.ones(n_words, dtype=np.float64)

    def quantize(self, descriptor: np.ndarray) -> int:
        """Map feature descriptor to nearest visual word ID."""
        d_float = descriptor.astype(np.float64)
        dists = [np.linalg.norm(d_float - w) for w in self.words]
        return int(np.argmin(dists))

    def insert_frame(self, frame_id: int, descriptors: np.ndarray) -> np.ndarray:
        """Build TF-IDF bow vector and update inverted index."""
        self._n_documents += 1
        n_words = len(self.words)
        tf = np.zeros(n_words, dtype=np.float64)

        for desc in descriptors:
            word_id = self.quantize(desc)
            tf[word_id] += 1.0

        if np.sum(tf) > 0:
            tf /= np.sum(tf)

        for word_id, freq in enumerate(tf):
            if freq > 0:
                if word_id not in self.inverted_index:
                    self.inverted_index[word_id] = []
                self.inverted_index[word_id].append((frame_id, freq))

        # Recompute IDF
        for word_id in range(n_words):
            doc_freq = len(self.inverted_index.get(word_id, []))
            self.idf_weights[word_id] = math.log((self._n_documents + 1.0) / (doc_freq + 1.0))

        bow_vec = tf * self.idf_weights
        norm = np.linalg.norm(bow_vec)
        return bow_vec / max(1e-8, norm)

    def query_similar_frames(self, bow_vector: np.ndarray, top_k: int = 5) -> list[tuple[int, float]]:
        """Score candidate loop closure frames using cosine similarity."""
        scores: dict[int, float] = {}

        for word_id, weight in enumerate(bow_vector):
            if weight <= 0 or word_id not in self.inverted_index:
                continue
            for frame_id, tf in self.inverted_index[word_id]:
                val = tf * self.idf_weights[word_id] * weight
                scores[frame_id] = scores.get(frame_id, 0.0) + val

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return ranked[:top_k]
