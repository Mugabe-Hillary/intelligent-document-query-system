import logging
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Protocol, Any
from functools import lru_cache

from langchain_community.document_loaders import TextLoader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()


# ==================== EXCEPTIONS ====================
class RAGException(Exception):
    """Base exception for RAG pipeline errors."""

    pass


class DocumentLoadError(RAGException):
    """Raised when document loading fails."""

    pass


class VectorStoreError(RAGException):
    """Raised when vector store operations fail."""

    pass


class QueryError(RAGException):
    """Raised when query processing fails."""

    pass


# ==================== CONFIGURATION ====================
@dataclass
class RAGConfig:
    """Configuration for the RAG pipeline."""

    # Document processing
    chunk_size: int = field(
        default_factory=lambda: _get_config_value("CHUNK_SIZE", 1000)
    )
    chunk_overlap: int = field(
        default_factory=lambda: _get_config_value("CHUNK_OVERLAP", 200)
    )

    # Embedding model
    embedding_model: str = field(
        default_factory=lambda: _get_config_value(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
    )
    embedding_device: str = field(
        default_factory=lambda: _get_config_value("EMBEDDING_DEVICE", "cpu")
    )

    # Vector store
    db_persist_directory: Path = field(
        default_factory=lambda: Path(
            _get_config_value("DB_PERSIST_DIRECTORY", "./chroma_db")
        )
    )
    collection_name: str = field(
        default_factory=lambda: _get_config_value(
            "CHROMA_COLLECTION_NAME", "rag_collection"
        )
    )

    # Retrieval
    k_retrieved_chunks: int = field(
        default_factory=lambda: _get_config_value("K_RETRIEVED_CHUNKS", 4)
    )

    # LLM
    llm_model: str = field(
        default_factory=lambda: os.getenv("LLM_MODEL", "gemini-1.5-flash")
    )
    llm_temperature: float = field(
        default_factory=lambda: float(os.getenv("LLM_TEMPERATURE", "0.1"))
    )

    # Source document path
    source_document_path: Optional[Path] = field(
        default_factory=lambda: _get_config_path("SOURCE_DOCUMENT_PATH")
    )

    # API Keys
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    google_api_key: str = field(default_factory=lambda: os.getenv("GOOGLE_API_KEY", ""))

    def __post_init__(self):
        """Validate configuration after initialization."""
        self.validate()

    def validate(self) -> None:
        """Validate configuration values."""
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        if self.k_retrieved_chunks <= 0:
            raise ValueError("k_retrieved_chunks must be positive")

        if "gemini" in self.llm_model.lower() and not self.google_api_key:
            raise ValueError(
                "GOOGLE_API_KEY environment variable is required for Gemini models"
            )
        elif "gpt" in self.llm_model.lower() and not self.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable is required for GPT models"
            )

        self.db_persist_directory.mkdir(parents=True, exist_ok=True)


def _get_config_value(key: str, default: Any) -> Any:
    """Get configuration value from config.py first, then environment, then default."""
    try:
        import config

        if hasattr(config, key):
            return getattr(config, key)
    except ImportError:
        pass

    # Fall back to environment variable
    env_value = os.getenv(key)
    if env_value is not None:
        # Convert to appropriate type based on default
        if isinstance(default, int):
            return int(env_value)
        elif isinstance(default, float):
            return float(env_value)
        elif isinstance(default, bool):
            return env_value.lower() in ("true", "1", "yes")
        else:
            return env_value

    return default


def _get_config_path(key: str) -> Optional[Path]:
    """Get path configuration value from config.py first, then environment."""
    try:
        import config

        if hasattr(config, key):
            path_value = getattr(config, key)
            return Path(path_value) if path_value else None
    except ImportError:
        pass

    # Fall back to environment variable
    env_value = os.getenv(key)
    return Path(env_value) if env_value else None


# ==================== UTILITIES ====================
def setup_logging(level: str = "INFO") -> logging.Logger:
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler("rag_pipeline.log")],
    )
    return logging.getLogger(__name__)


def validate_query(query: str) -> str:
    """Validate and sanitize query input."""
    if not query or not isinstance(query, str):
        raise ValueError("Query must be a non-empty string")

    query = query.strip()
    if not query:
        raise ValueError("Query cannot be empty after stripping whitespace")

    if len(query) > 1000:  # Reasonable limit
        raise ValueError("Query is too long (max 1000 characters)")

    return query


def validate_file_path(file_path: str) -> Path:
    """Validate file path and convert to Path object."""
    if not file_path or not isinstance(file_path, str):
        raise ValueError("File path must be a non-empty string")

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if not path.is_file():
        raise ValueError(f"Path is not a file: {file_path}")

    # Check file extension
    allowed_extensions = {".txt", ".md", ".rst"}
    if path.suffix.lower() not in allowed_extensions:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    return path


# ==================== DOCUMENT HANDLING ====================
class DocumentLoader(Protocol):
    """Protocol for document loaders."""

    def load(self, file_path: Path) -> List[Document]:
        """Load documents from file path."""
        ...


class TextDocumentLoader:
    """Concrete implementation for text document loading."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def load(self, file_path: Path) -> List[Document]:
        """Load documents from text file."""
        try:
            self.logger.info(f"Loading document from: {file_path}")

            loader = TextLoader(str(file_path), encoding="utf-8")
            documents = loader.load()

            if not documents:
                raise DocumentLoadError(f"No content loaded from {file_path}")

            self.logger.info(f"Successfully loaded {len(documents)} document(s)")
            return documents

        except Exception as e:
            self.logger.error(f"Failed to load document from {file_path}: {e}")
            raise DocumentLoadError(f"Failed to load document: {e}") from e


class DocumentProcessor:
    """Handle document chunking and processing."""

    def __init__(self, config: RAGConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self._text_splitter = None

    @property
    def text_splitter(self) -> RecursiveCharacterTextSplitter:
        """Lazy initialization of text splitter."""
        if self._text_splitter is None:
            self._text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                length_function=len,
                add_start_index=True,
            )
        return self._text_splitter

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks."""
        if not documents:
            raise ValueError("Cannot chunk empty document list")

        try:
            self.logger.info("Chunking documents...")

            chunked_documents = self.text_splitter.split_documents(documents)

            if not chunked_documents:
                raise DocumentLoadError("Document chunking resulted in no chunks")

            self.logger.info(
                f"Successfully chunked documents into {len(chunked_documents)} chunks"
            )
            return chunked_documents

        except Exception as e:
            self.logger.error(f"Failed to chunk documents: {e}")
            raise DocumentLoadError(f"Failed to chunk documents: {e}") from e


# ==================== VECTOR STORE ====================
class VectorStoreManager:
    """Manage vector store operations."""

    def __init__(self, config: RAGConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self._embeddings = None

    @property
    def embeddings(self) -> HuggingFaceEmbeddings:
        """Lazy initialization of embeddings model."""
        if self._embeddings is None:
            try:
                self.logger.info("Initializing embedding model...")
                self._embeddings = HuggingFaceEmbeddings(
                    model_name=self.config.embedding_model,
                    model_kwargs={"device": self.config.embedding_device},
                )
                self.logger.info("Embedding model initialized successfully")
            except Exception as e:
                self.logger.error(f"Failed to initialize embedding model: {e}")
                raise VectorStoreError(
                    f"Failed to initialize embedding model: {e}"
                ) from e

        return self._embeddings

    def create_vector_store(self, documents: List[Document]) -> Chroma:
        """Create and persist vector store from documents."""
        if not documents:
            raise ValueError("Cannot create vector store from empty document list")

        try:
            self.logger.info(
                f"Creating vector store with {len(documents)} documents..."
            )

            vectordb = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=str(self.config.db_persist_directory),
                collection_name=self.config.collection_name,
            )

            # Verify the store was created properly
            doc_count = vectordb._collection.count()
            if doc_count == 0:
                raise VectorStoreError("Vector store created but contains no documents")

            self.logger.info(
                f"Successfully created vector store with {doc_count} documents"
            )
            return vectordb

        except Exception as e:
            self.logger.error(f"Failed to create vector store: {e}")
            raise VectorStoreError(f"Failed to create vector store: {e}") from e

    def load_vector_store(self) -> Chroma:
        """Load existing vector store."""
        try:
            self.logger.info("Loading existing vector store...")

            vectordb = Chroma(
                persist_directory=str(self.config.db_persist_directory),
                embedding_function=self.embeddings,
                collection_name=self.config.collection_name,
            )

            # Verify the store has documents
            doc_count = vectordb._collection.count()
            if doc_count == 0:
                raise VectorStoreError("Vector store exists but contains no documents")

            self.logger.info(
                f"Successfully loaded vector store with {doc_count} documents"
            )
            return vectordb

        except Exception as e:
            self.logger.error(f"Failed to load vector store: {e}")
            raise VectorStoreError(f"Failed to load vector store: {e}") from e


# ==================== LLM FACTORY ====================
class LLMFactory:
    """Factory for creating LLM instances."""

    @staticmethod
    def create_llm(config: RAGConfig) -> ChatOpenAI:
        """Create LLM instance based on configuration."""
        if "gemini" in config.llm_model.lower():
            return ChatGoogleGenerativeAI(
                model=config.llm_model,
                temperature=config.llm_temperature,
                google_api_key=config.google_api_key,
                convert_system_message_to_human=True,
            )
        elif "gpt" in config.llm_model.lower():
            return ChatOpenAI(
                model_name=config.llm_model,
                temperature=config.llm_temperature,
                api_key=config.openai_api_key,
            )
        else:
            raise RAGException(f"Unsupported LLM model specified: {config.llm_model}")


# ==================== PERFORMANCE MONITORING ====================
@contextmanager
def performance_monitoring(logger: logging.Logger):
    """Context manager for performance monitoring."""
    start_time = time.time()
    start_memory = _get_memory_usage()

    try:
        yield
    finally:
        end_time = time.time()
        end_memory = _get_memory_usage()

        logger.info(f"Operation completed in {end_time - start_time:.2f} seconds")
        logger.info(f"Memory usage: {end_memory - start_memory:.2f} MB")


def _get_memory_usage() -> float:
    """Get current memory usage in MB."""
    try:
        import psutil

        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    except ImportError:
        return 0.0  # Return 0 if psutil is not available
