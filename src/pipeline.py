import logging
import time
from functools import lru_cache
from typing import Optional

from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

from core import (
    RAGConfig,
    RAGException,
    QueryError,
    TextDocumentLoader,
    DocumentProcessor,
    VectorStoreManager,
    LLMFactory,
    setup_logging,
    validate_query,
    validate_file_path,
    performance_monitoring,
)


# ==================== RAG PROCESSOR ====================
class RAGProcessor:
    """Handle RAG query processing."""

    def __init__(self, config: RAGConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self._llm = None
        self._vector_store_manager = VectorStoreManager(config, logger)

    @property
    def llm(self):
        """Lazy initialization of LLM."""
        if self._llm is None:
            try:
                self.logger.info(f"Initializing LLM: {self.config.llm_model}")
                self._llm = LLMFactory.create_llm(self.config)
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


# ==================== MAIN PIPELINE ====================
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

    def query_with_monitoring(self, query: str, use_cache: bool = True) -> str:
        """Query the RAG pipeline with performance monitoring."""
        with performance_monitoring(self.logger):
            return self.query(query, use_cache)


# ==================== MAIN FUNCTION ====================
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
        query = "What challenges did the expedition face with the natives?"
        answer = pipeline.query_with_monitoring(query)

        print(f"\nQuery: {query}")
        print(f"Answer: {answer}")

    except Exception as e:
        logging.error(f"Pipeline execution failed: {e}")
        raise


if __name__ == "__main__":
    main()
