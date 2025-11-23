import numpy as np
import logging

from memo_mcp.rag.config.rag_config import DocumentMetadata, RAGConfig
from memo_mcp.rag.vector.database.vector_backend import VectorDatabase
from typing import Any


class VectorStore:
    """
    Vector storage interface with multiple backend implementations.

    Supports FAISS, Chroma, Qdrant, and simple in-memory storage.
    """

    def __init__(self, config: RAGConfig):
        """
        Initialize the vector store.

        Args:
            config: RAG configuration specifying backend type.
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # from memo_mcp.rag.vector.database.qdrant import QdrantBackend
        from memo_mcp.rag.vector.database.chromadb import ChromaBackend
        from memo_mcp.rag.vector.database.faiss import FAISSBackend
        from memo_mcp.rag.vector.database.simple import SimpleBackend

        self.backend: FAISSBackend | ChromaBackend | SimpleBackend

        backend_type = config.vector_store_type.lower()
        if backend_type == "faiss":
            self.backend = FAISSBackend(config)
        elif backend_type == "chroma":
            self.backend = ChromaBackend(config)
        # elif backend_type == "qdrant":
        #     self.backend = QdrantBackend(config)
        else:
            self.backend = SimpleBackend(config)

    async def initialize(self) -> None:
        """
        Initialize the vector store backend.

        Delegates to the specific backend implementation.
        """
        await self.backend.initialize()

    async def add_documents(
        self,
        embeddings: list[np.ndarray],
        texts: list[str],
        metadatas: list[DocumentMetadata],
    ) -> None:
        """
        Add document embeddings to the store.

        Args:
            embeddings: List of embedding vectors.
            texts: List of document texts.
            metadatas: List of document metadata.
        """
        await self.backend.add_documents(embeddings, texts, metadatas)

    def search(
        self, query_embedding: np.ndarray, top_k: int, similarity_threshold: float = 0.0
    ) -> list[dict[str, Any]]:
        """
        Search for similar documents.

        Args:
            query_embedding: Query embedding vector.
            top_k: Number of results to return.
            similarity_threshold: Minimum similarity score.

        Returns:
            List of search results with metadata and scores.
        """
        return self.backend.search(query_embedding, top_k, similarity_threshold)

    async def remove_document(self, file_path: str) -> bool:
        """
        Remove all chunks of a document.

        Args:
            file_path: Path of the document to remove.

        Returns:
            True if document was removed successfully.
        """
        return await self.backend.remove_document(file_path)

    def get_document_count(self) -> int:
        """
        Get the number of unique documents.

        Returns:
            Count of unique documents in the store.
        """
        return self.backend.get_document_count()

    def get_chunk_count(self) -> int:
        """
        Get the total number of chunks.

        Returns:
            Total number of chunks across all documents.
        """
        return self.backend.get_chunk_count()

    def is_empty(self) -> bool:
        """
        Check if the vector store is empty.

        Returns:
            True if store contains no documents.
        """
        return self.backend.is_empty()

    async def clear(self) -> None:
        """
        Clear all data from the vector store.

        Removes all documents and resets the store.
        """
        await self.backend.clear()

    async def close(self) -> None:
        """
        Close the vector store and cleanup resources.

        Persists any pending changes and releases resources.
        """
        if self.backend:
            await self.backend.close()

    def health_check(self) -> dict[str, Any]:
        """
        Perform a health check of the vector store.

        Returns:
            Dictionary with health status and statistics.
        """
        return self.backend.health_check()

    def get_stats(self) -> dict[str, Any]:
        """
        Get statistical information about the vector store.

        Returns:
            Dictionary with store statistics.
        """
        return self.backend.get_stats()

    def get_backend(self) -> VectorDatabase:
        """
        Get direct access to the underlying backend for advanced operations.

        Use this when you need backend-specific features:

        Example:
            store = VectorStore(config)
            if isinstance(store.get_backend(), QdrantBackend):
                results = await store.get_backend().search_with_filter(...)
        """
        return self.backend

    def get_backend_type(self) -> str:
        """
        Get the type of the current backend.

        Returns:
            Backend class name as string.
        """
        return self.backend.__class__.__name__


def create_vector_store(config: RAGConfig) -> VectorStore:
    """
    Factory function to create vector store instances.

    Args:
        config: RAG configuration

    Returns:
        VectorStore instance
    """
    return VectorStore(config)
