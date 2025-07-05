import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Protocol, Any
from functools import lru_cache
import time

from langchain_community.document_loaders import TextLoader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()


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


# Configuration management
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
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")

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

    # Use default
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


# Logging setup
def setup_logging(level: str = "INFO") -> logging.Logger:
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler("rag_pipeline.log")],
    )
    return logging.getLogger(__name__)


# Input validation
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


# Document loading interface
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


# Document processing
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


# Vector store management
class VectorStoreManager:
    """Manage vector store operations."""

    def __init__(self, config: RAGConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self._embeddings = None
        self._vectordb = None

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


# RAG query processor
class RAGProcessor:
    """Handle RAG query processing."""

    def __init__(self, config: RAGConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self._llm = None
        self._vector_store_manager = VectorStoreManager(config, logger)

    @property
    def llm(self) -> ChatOpenAI:
        """Lazy initialization of LLM."""
        if self._llm is None:
            try:
                self.logger.info(f"Initializing LLM: {self.config.llm_model}")

                # --- FACTORY LOGIC ---
                if "gemini" in self.config.llm_model.lower():
                    self._llm = ChatGoogleGenerativeAI(
                        model=self.config.llm_model,
                        temperature=self.config.llm_temperature,
                        google_api_key=self.config.google_api_key,
                        # Gemini can be aggressive with safety settings
                        convert_system_message_to_human=True,
                    )
                elif "gpt" in self.config.llm_model.lower():
                    self._llm = ChatOpenAI(
                        model_name=self.config.llm_model,
                        temperature=self.config.llm_temperature,
                        api_key=self.config.openai_api_key,
                    )
                else:
                    raise RAGException(
                        f"Unsupported LLM model specified: {self.config.llm_model}"
                    )

                self.logger.info("LLM initialized successfully")

            except Exception as e:
                self.logger.error(f"Failed to initialize LLM: {e}")
                raise QueryError(f"Failed to initialize LLM: {e}") from e

        return self._llm

    def create_prompt_template(self) -> PromptTemplate:
        """Create the prompt template for RAG."""
        template = """
        You are an AI assistant helping with questions about a document.
        Answer the user's question based only on the following context.
        If the answer is not found in the context, respond with "I cannot answer this question based on the provided document."
        
        Context:
        {context}
        
        Question:
        {input}
        
        Answer:
        """

        return PromptTemplate(template=template, input_variables=["context", "input"])

    @lru_cache(maxsize=100)
    def _cached_query(self, query: str) -> str:
        """Cached query processing (for identical queries)."""
        return self._process_query_uncached(query)

    def _process_query_uncached(self, query: str) -> str:
        """Process query without caching."""
        try:
            # Load vector store
            vectordb = self._vector_store_manager.load_vector_store()

            # Create retriever
            retriever = vectordb.as_retriever(
                search_kwargs={"k": self.config.k_retrieved_chunks}
            )

            # Create chains
            prompt = self.create_prompt_template()
            question_answer_chain = create_stuff_documents_chain(self.llm, prompt)
            rag_chain = create_retrieval_chain(retriever, question_answer_chain)

            # Process query
            self.logger.info(f"Processing query: '{query[:50]}...'")
            start_time = time.time()

            response = rag_chain.invoke({"input": query})

            end_time = time.time()
            self.logger.info(f"Query processed in {end_time - start_time:.2f} seconds")

            return response["answer"]

        except Exception as e:
            self.logger.error(f"Failed to process query: {e}")
            raise QueryError(f"Failed to process query: {e}") from e

    def query(self, query: str, use_cache: bool = True) -> str:
        """Process a query through the RAG pipeline."""
        validated_query = validate_query(query)

        if use_cache:
            return self._cached_query(validated_query)
        else:
            return self._process_query_uncached(validated_query)


# Main RAG pipeline orchestrator
class RAGPipeline:
    """Main RAG pipeline orchestrator."""

    def __init__(self, config: Optional[RAGConfig] = None):
        self.config = config or RAGConfig()
        self.logger = setup_logging()

        # Initialize components
        self.document_loader = TextDocumentLoader(self.logger)
        self.document_processor = DocumentProcessor(self.config, self.logger)
        self.vector_store_manager = VectorStoreManager(self.config, self.logger)
        self.rag_processor = RAGProcessor(self.config, self.logger)

    def setup_pipeline(self, file_path: str) -> None:
        """Set up the RAG pipeline with a document."""
        try:
            self.logger.info("Setting up RAG pipeline...")

            # Validate and load document
            validated_path = validate_file_path(file_path)
            documents = self.document_loader.load(validated_path)

            # Process documents
            chunked_docs = self.document_processor.chunk_documents(documents)

            # Create vector store
            self.vector_store_manager.create_vector_store(chunked_docs)

            self.logger.info("RAG pipeline setup completed successfully")

        except Exception as e:
            self.logger.error(f"Failed to setup RAG pipeline: {e}")
            raise

    def query(self, query: str, use_cache: bool = True) -> str:
        """Query the RAG pipeline."""
        try:
            return self.rag_processor.query(query, use_cache)
        except Exception as e:
            self.logger.error(f"Query failed: {e}")
            raise

    @contextmanager
    def performance_monitoring(self):
        """Context manager for performance monitoring."""
        start_time = time.time()
        start_memory = self._get_memory_usage()

        try:
            yield
        finally:
            end_time = time.time()
            end_memory = self._get_memory_usage()

            self.logger.info(
                f"Operation completed in {end_time - start_time:.2f} seconds"
            )
            self.logger.info(f"Memory usage: {end_memory - start_memory:.2f} MB")

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        import psutil

        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024


def main():
    """Example usage of the refactored RAG pipeline."""
    try:
        pipeline = RAGPipeline()

        if pipeline.config.source_document_path:
            print(
                f"Setting up pipeline with document: {pipeline.config.source_document_path}"
            )
            pipeline.setup_pipeline(str(pipeline.config.source_document_path))
        else:
            print(
                "No SOURCE_DOCUMENT_PATH found in config. Please call setup_pipeline() manually."
            )
            print("Example: pipeline.setup_pipeline('path/to/your/document.txt')")
            return

        # Query the pipeline
        with pipeline.performance_monitoring():
            query = "What challenges did the expedition face with the natives?"
            answer = pipeline.query(query)

            print(f"\nQuery: {query}")
            print(f"Answer: {answer}")

    except Exception as e:
        logging.error(f"Pipeline execution failed: {e}")
        raise


if __name__ == "__main__":
    main()
