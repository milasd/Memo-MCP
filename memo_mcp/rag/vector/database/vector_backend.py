import numpy as np
import logging

from abc import ABC, abstractmethod
from memo_mcp.rag.config.rag_config import DocumentMetadata, RAGConfig
from typing import Any


class VectorDatabase(ABC):
    """
    Abstract base class defining the interface for all vector store backends.
    """

    def __init__(self, config: RAGConfig):
        """
        Initialize the vector database backend.

        Args:
            config: RAG configuration.
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

    @abstractmethod
    async def initialize(self) -> None:
        """
        Initialize the vector store backend.

        Sets up necessary connections and data structures.
        """
        pass

    @abstractmethod
    async def add_documents(
        self,
        embeddings: list[np.ndarray],
        texts: list[str],
        metadatas: list[DocumentMetadata],
    ) -> None:
        """
        Add document embeddings to the store.

        Args:
            embeddings: List of document embeddings
            texts: List of document texts
            metadatas: List of document metadata
        """
        pass

    @abstractmethod
    def search(
        self, query_embedding: np.ndarray, top_k: int, similarity_threshold: float = 0.0
    ) -> list[dict[str, Any]]:
        """
        Search for similar documents.

        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            similarity_threshold: Minimum similarity score

        Returns:
            List of search results with text, metadata, and similarity scores
        """
        pass

    @abstractmethod
    async def remove_document(self, file_path: str) -> bool:
        """
        Remove all chunks of a document.

        Args:
            file_path: Path of the document to remove

        Returns:
            True if document was found and removed successfully
        """
        pass

    @abstractmethod
    def get_document_count(self) -> int:
        """
        Get the number of unique documents in the store.

        Returns:
            Count of unique documents.
        """
        pass

    @abstractmethod
    def get_chunk_count(self) -> int:
        """
        Get the total number of chunks in the store.

        Returns:
            Total number of chunks.
        """
        pass

    @abstractmethod
    def is_empty(self) -> bool:
        """
        Check if the vector store is empty.

        Returns:
            True if store contains no documents.
        """
        pass

    @abstractmethod
    async def clear(self) -> None:
        """
        Clear all data from the vector store.

        Removes all documents and resets the store.
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """
        Close the vector store and cleanup resources.

        Persists pending changes and releases resources.
        """
        pass

    def health_check(self) -> dict[str, Any]:
        """
        Perform a health check of the backend.
        Default implementation provides basic status.
        """
        try:
            chunk_count = self.get_chunk_count()
            doc_count = self.get_document_count()
            is_empty = self.is_empty()

            return {
                "status": "healthy",
                "backend_type": self.__class__.__name__,
                "chunk_count": chunk_count,
                "document_count": doc_count,
                "is_empty": is_empty,
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "backend_type": self.__class__.__name__,
                "error": str(e),
            }

    def get_stats(self) -> dict[str, Any]:
        """
        Get basic stats about the vector store, such as document count.
        """
        return {
            "document_count": self.get_document_count(),
            "chunk_count": self.get_chunk_count(),
            "is_empty": self.is_empty(),
            "backend_type": self.__class__.__name__,
        }
