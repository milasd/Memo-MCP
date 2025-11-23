import hashlib
import logging
import pickle
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from memo_mcp.rag.config.rag_config import RAGConfig


class EmbeddingManager:
    """
    Manages text embeddings with caching and GPU acceleration.

    Supports multiple embedding models and automatic device selection.
    """

    def __init__(self, config: RAGConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Model and device state
        self.model: Any = None
        self.tokenizer: Any = None
        self.device: str | None = None
        self.model_name = config.embedding_model
        self.embedding_dimension = config.embedding_dimension

        # Caching
        self._cache: dict[str, NDArray[np.float32]] = {}
        self._cache_dirty = False

    def initialize(self) -> None:
        """Initialize the embedding model and determine device."""
        self.logger.info(f"Initializing embedding model: {self.model_name}")

        # Determine device
        self.device = self._get_best_device()
        self.logger.info(f"Using device: {self.device}")

        # Load model based on type
        if "sentence-transformers" in self.model_name:
            self._load_sentence_transformer()
        elif "bge" in self.model_name.lower():
            self._load_fastembed()
        else:
            self._load_huggingface_model()

        # Load cache if it exists
        self._load_cache()

        self.logger.info("Embedding manager initialized successfully")

    def _get_best_device(self) -> str:
        """Determine the best available device."""
        if not self.config.use_gpu:
            return "cpu"

        try:
            import torch

            if torch.cuda.is_available():
                device = f"cuda:{torch.cuda.current_device()}"
                gpu_name = torch.cuda.get_device_name()
                self.logger.info(f"CUDA GPU available: {gpu_name}")
                return device

            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.logger.info("Apple MPS GPU available")
                return "mps"

        except ImportError:
            self.logger.warning("PyTorch not available, falling back to CPU")

        self.logger.info("Using CPU for embeddings")
        return "cpu"

    def _load_sentence_transformer(self) -> None:
        """Load a sentence transformer model."""
        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(
                self.model_name,
                device=self.device if self.device != "mps" else "cpu",
            )

            # Update embedding dimension from model
            if self.model:
                self.embedding_dimension = self.model.get_sentence_embedding_dimension()

        except ImportError as e:
            raise ImportError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            ) from e

    def _load_fastembed(self) -> None:
        """Load a BGE model using FastEmbed for optimal performance."""
        try:
            from fastembed import TextEmbedding

            # Map model names to FastEmbed model names
            fastembed_model_map = {
                "BAAI/bge-base-en-v1.5": "BAAI/bge-base-en-v1.5",
                "BAAI/bge-small-en-v1.5": "BAAI/bge-small-en-v1.5", 
                "BAAI/bge-large-en-v1.5": "BAAI/bge-large-en-v1.5",
                "BAAI/bge-m3": "BAAI/bge-m3",
            }

            model_name = fastembed_model_map.get(self.model_name, self.model_name)
            
            self.model = TextEmbedding(
                model_name=model_name,
                max_length=512,  # BGE models work well with 512 tokens
                cache_dir=str(self.config.index_path / "fastembed_cache"),
            )

            # Update embedding dimension from model
            # BGE models have known dimensions
            if "bge-base" in self.model_name:
                self.embedding_dimension = 768
            elif "bge-small" in self.model_name:
                self.embedding_dimension = 384
            elif "bge-large" in self.model_name:
                self.embedding_dimension = 1024
            elif "bge-m3" in self.model_name:
                self.embedding_dimension = 1024
            else:
                self.embedding_dimension = self.config.embedding_dimension

            self.logger.info(f"Loaded BGE model with FastEmbed: {model_name} (dim: {self.embedding_dimension})")

        except ImportError as e:
            raise ImportError(
                "FastEmbed not installed. "
                "Install with: uv add fastembed"
            ) from e

    def _load_flag_embedding(self) -> None:
        """Load a BGE model using FlagEmbedding for optimal performance."""
        try:
            from FlagEmbedding import BGEM3FlagModel, FlagModel

            # Check if it's a BGE-M3 model or regular BGE model
            if "m3" in self.model_name.lower():
                self.model = BGEM3FlagModel(
                    self.model_name,
                    device=self.device if self.device != "mps" else "cpu",
                )
                # BGE-M3 has fixed dimension of 1024
                self.embedding_dimension = 1024
            else:
                self.model = FlagModel(
                    self.model_name,
                    query_instruction_for_retrieval="Represent this sentence for searching relevant passages: ",
                    use_fp16=(self.device != "cpu" and self.device != "mps"),
                )
                # Keep the configured dimension for regular BGE models
                self.embedding_dimension = self.config.embedding_dimension

            self.logger.info(f"Loaded BGE model with FlagEmbedding: {self.model_name}")

        except ImportError as e:
            raise ImportError(
                "FlagEmbedding not installed. "
                "Install with: uv add FlagEmbedding"
            ) from e

    def _load_huggingface_model(self) -> None:
        """Load a HuggingFace transformers model."""
        try:
            from transformers import AutoModel, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)

            if self.device != "cpu" and self.model:
                self.model = self.model.to(self.device)

        except ImportError as e:
            raise ImportError(
                "transformers not installed. "
                "Install with: pip install transformers torch"
            ) from e

    def _load_cache(self) -> None:
        """Load embedding cache from disk."""
        if not self.config.cache_embeddings:
            return

        cache_path = self.config.embeddings_cache_path
        if cache_path.exists():
            try:
                with open(cache_path, "rb") as f:
                    self._cache = pickle.load(f)
                self.logger.info(f"Loaded {len(self._cache)} cached embeddings")
            except Exception as e:
                self.logger.warning(f"Failed to load embedding cache: {e}")
                self._cache = {}

    def _save_cache(self) -> None:
        """Save embedding cache to disk."""
        if not self.config.cache_embeddings or not self._cache_dirty:
            return

        try:
            cache_path = self.config.embeddings_cache_path
            cache_path.parent.mkdir(parents=True, exist_ok=True)

            with open(cache_path, "wb") as f:
                pickle.dump(self._cache, f)

            self._cache_dirty = False
            self.logger.debug(f"Saved {len(self._cache)} embeddings to cache")

        except Exception as e:
            self.logger.warning(f"Failed to save embedding cache: {e}")

    def _get_cache_key(self, text: str) -> str:
        """Generate a cache key for text."""
        return hashlib.md5(f"{self.model_name}:{text}".encode()).hexdigest()

    def _apply_bge_prefix(self, text: str, is_query: bool) -> str:
        """Apply BGE model query prefix for better retrieval performance."""
        if not is_query or "bge" not in self.model_name.lower():
            return text
        
        # BGE models benefit from query prefixes for retrieval tasks
        return f"Represent this sentence for searching relevant passages: {text}"

    def embed_text(self, text: str, is_query: bool = False) -> NDArray[np.float32]:
        """
        Generate embedding for a single text.

        Args:
            text: Input text
            is_query: Whether this is a query (for BGE models, adds "Represent this sentence for searching relevant passages:")

        Returns:
            Embedding vector as numpy array
        """
        if not text.strip():
            return np.zeros(self.embedding_dimension, dtype=np.float32)

        # Apply BGE query prefix if needed
        processed_text = self._apply_bge_prefix(text, is_query)

        # Check cache first
        cache_key = self._get_cache_key(processed_text)
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Generate embedding
        embedding: NDArray[np.float32]
        if hasattr(self.model, "embed"):  # FastEmbed
            embeddings = self._embed_with_fastembed([processed_text])
            embedding = embeddings[0]
        elif hasattr(self.model, "encode") and not hasattr(self.model, "encode_queries"):  # SentenceTransformer
            embeddings = self._embed_with_sentence_transformer([processed_text])
            embedding = embeddings[0]
        elif hasattr(self.model, "encode_queries"):  # FlagEmbedding
            embeddings = self._embed_with_flag_embedding([processed_text], is_query)
            embedding = embeddings[0]
        else:  # HuggingFace model
            embeddings = self._embed_with_huggingface([processed_text])
            embedding = embeddings[0]

        # Cache the result
        if self.config.cache_embeddings:
            self._cache[cache_key] = embedding
            self._cache_dirty = True

        return embedding

    def embed_texts(self, texts: list[str]) -> list[NDArray[np.float32]]:
        """
        Generate embeddings for multiple texts efficiently.

        Args:
            texts: List of input texts

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        # Separate cached and uncached texts
        cached_embeddings: dict[int, NDArray[np.float32]] = {}
        uncached_texts: list[str] = []
        uncached_indices: list[int] = []

        for i, text in enumerate(texts):
            if not text.strip():
                cached_embeddings[i] = np.zeros(
                    self.embedding_dimension, dtype=np.float32
                )
                continue

            cache_key = self._get_cache_key(text)
            if cache_key in self._cache:
                cached_embeddings[i] = self._cache[cache_key]
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)

        # Generate embeddings for uncached texts
        if uncached_texts:
            if hasattr(self.model, "embed"):  # FastEmbed
                new_embeddings = self._embed_with_fastembed(uncached_texts)
            elif hasattr(self.model, "encode") and not hasattr(self.model, "encode_queries"):  # SentenceTransformer
                new_embeddings = self._embed_with_sentence_transformer(uncached_texts)
            elif hasattr(self.model, "encode_queries"):  # FlagEmbedding
                new_embeddings = self._embed_with_flag_embedding(uncached_texts, is_query=False)
            else:  # HuggingFace model
                new_embeddings = self._embed_with_huggingface(uncached_texts)

            # Cache new embeddings
            if self.config.cache_embeddings:
                for text, embedding in zip(
                    uncached_texts, new_embeddings, strict=False
                ):
                    cache_key = self._get_cache_key(text)
                    self._cache[cache_key] = embedding
                self._cache_dirty = True

            # Add to results
            for i, embedding in zip(uncached_indices, new_embeddings, strict=False):
                cached_embeddings[i] = embedding

        # Return in original order
        return [cached_embeddings[i] for i in range(len(texts))]

    def _embed_with_sentence_transformer(
        self, texts: list[str]
    ) -> list[NDArray[np.float32]]:
        """Generate embeddings using SentenceTransformer."""
        if not self.model:
            return []

        embeddings = self.model.encode(
            texts,
            batch_size=self.config.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return [cast(NDArray[np.float32], emb.astype(np.float32)) for emb in embeddings]

    def _embed_with_fastembed(self, texts: list[str]) -> list[NDArray[np.float32]]:
        """Generate embeddings using FastEmbed."""
        if not self.model:
            return []

        try:
            # FastEmbed's embed method returns a generator of numpy arrays
            embeddings_generator = self.model.embed(texts)
            embeddings = list(embeddings_generator)
            
            return [cast(NDArray[np.float32], emb.astype(np.float32)) for emb in embeddings]

        except Exception as e:
            self.logger.error(f"Failed to generate embeddings with FastEmbed: {e}")
            raise

    def _embed_with_flag_embedding(self, texts: list[str], is_query: bool = False) -> list[NDArray[np.float32]]:
        """Generate embeddings using FlagEmbedding."""
        if not self.model:
            return []

        try:
            # Check if it's BGE-M3 model (has encode method that returns dict)
            if hasattr(self.model, "encode") and "m3" in self.model_name.lower():
                # BGE-M3 model returns dict with dense_vecs
                result = self.model.encode(texts, batch_size=self.config.batch_size)
                if isinstance(result, dict) and "dense_vecs" in result:
                    embeddings = result["dense_vecs"]
                else:
                    embeddings = result
            else:
                # Regular BGE model with FlagModel
                if is_query:
                    embeddings = self.model.encode_queries(texts, batch_size=self.config.batch_size)
                else:
                    embeddings = self.model.encode_corpus(texts, batch_size=self.config.batch_size)

            # Ensure embeddings is a numpy array
            import numpy as np
            if not isinstance(embeddings, np.ndarray):
                embeddings = np.array(embeddings)
            
            # Convert to list of individual embeddings
            if len(embeddings.shape) == 1:
                # Single embedding
                return [cast(NDArray[np.float32], embeddings.astype(np.float32))]
            else:
                # Multiple embeddings
                return [cast(NDArray[np.float32], emb.astype(np.float32)) for emb in embeddings]

        except Exception as e:
            self.logger.error(f"Failed to generate embeddings with FlagEmbedding: {e}")
            raise

    def _embed_with_huggingface(self, texts: list[str]) -> list[NDArray[np.float32]]:
        """Generate embeddings using HuggingFace transformers."""
        try:
            import torch
            import torch.nn.functional as F

            if not self.model or not self.tokenizer:
                return []

            embeddings: list[NDArray[np.float32]] = []

            # Process in batches
            for i in range(0, len(texts), self.config.batch_size):
                batch_texts = texts[i : i + self.config.batch_size]

                # Tokenize
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=2000,
                )

                if self.device != "cpu":
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # Generate embeddings
                with torch.no_grad():
                    outputs = self.model(**inputs)

                    # Use mean pooling
                    attention_mask = inputs["attention_mask"]
                    token_embeddings = outputs.last_hidden_state

                    input_mask_expanded = (
                        attention_mask.unsqueeze(-1)
                        .expand(token_embeddings.size())
                        .float()
                    )
                    batch_embeddings = torch.sum(
                        token_embeddings * input_mask_expanded, 1
                    ) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

                    # Normalize
                    batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)

                    # Convert to numpy
                    np_embeddings = batch_embeddings.cpu().numpy().astype(np.float32)
                    embeddings.extend(np_embeddings)

            return embeddings

        except Exception as e:
            self.logger.error(
                f"Failed to generate embeddings with HuggingFace model: {e}"
            )
            raise

    async def close(self) -> None:
        """Clean up resources and save cache."""
        self._save_cache()

        # Clean up model resources
        self.model = None
        self.tokenizer = None

        # Clear GPU memory if using CUDA
        if self.device and "cuda" in self.device:
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

        # Force garbage collection
        import gc

        gc.collect()

        self.logger.info("Embedding manager closed")
