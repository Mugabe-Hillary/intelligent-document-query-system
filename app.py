import streamlit as st
import logging
import tempfile
import os
import shutil
import atexit
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from src.rag_pipeline import RAGPipeline, RAGConfig


# --- Configuration ---
class UIConfig:
    PAGE_CONFIG = {
        "page_title": "Intelligent Document Query",
        "page_icon": "🧮",
        "layout": "wide",
        "initial_sidebar_state": "expanded",
    }

    MESSAGES = {
        "default_doc": "Through the Dark Continent by Henry M. Stanley",
        "upload_help": "Upload a .txt file to chat with it.",
        "query_placeholder": "Ask a question about the document...",
        "processing_spinner": "Processing '{filename}'... This may take a moment.",
        "analyzing_spinner": "Analyzing document...",
        "connection_error": "⚠️ Connection error. Please check your internet connection and try again.",
        "timeout_error": "⏱️ Request timed out. Please try a simpler question or try again later.",
        "general_error": "❌ I encountered an error: {error}",
        "system_not_ready": "System not ready. Please wait for initialization to complete.",
        "file_too_large": "File too large. Maximum size: {max_size}MB",
        "unsupported_file": "Unsupported file type. Supported: {supported_types}",
        "processing_failed": "Failed to process uploaded file: {error}",
        "retry_button": "🔄 Retry",
        "clear_chat": "Clear Chat History",
        "switch_default": "Switch back to Default Document",
        "currently_chatting": "Currently chatting with: **{doc_name}**",
        "questions_asked": "Questions Asked",
        "upload_section": "⬆️ Upload Your Own Document",
        "document_info": "📋 Document Information",
        "query_tips": "💡 Query Tips",
        "actions": "🔧 Actions",
        "chat_title": "💬 Chat: {doc_name}",
        "main_title": "🧮 Intelligent Document Query System",
        "main_description": """
        Ask questions about **'Through the Dark Continent'** by Henry M. Stanley. 
        This system uses AI to provide accurate, contextual answers based on the document content.
        """,
        "default_doc_info": "**Document:** Through the Dark Continent by Henry M. Stanley",
        "tips_content": """
        - Ask specific questions about events, people, or places
        - Use clear, complete sentences
        - Try questions like:
          - "What challenges did Stanley face?"
          - "Describe the expedition route"
          - "Who were the key people mentioned?"
        """,
    }

    SUPPORTED_FILE_TYPES = ["txt"]
    MAX_FILE_SIZE_MB = 10
    CACHE_TTL = 300  # 5 minutes


# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --- Resource Management ---
def cleanup_temp_resources() -> None:
    """Clean up temporary resources when switching documents."""
    for key in list(st.session_state.keys()):
        if key.startswith("temp_dir_"):
            temp_dir = st.session_state[key]
            if os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    logger.info(f"Cleaned up temporary directory: {temp_dir}")
                except Exception as e:
                    logger.warning(f"Failed to cleanup temp directory {temp_dir}: {e}")
            del st.session_state[key]


def cleanup_on_session_end() -> None:
    """Register cleanup function for session end."""
    cleanup_temp_resources()


# Register cleanup for session end
atexit.register(cleanup_on_session_end)


# --- Caching the RAG Pipeline ---
@st.cache_resource
def load_rag_pipeline() -> RAGPipeline:
    """
    Loads and caches the RAG pipeline.

    Returns:
        RAGPipeline: Initialized pipeline instance

    Raises:
        RuntimeError: If pipeline initialization fails
    """
    try:
        config = RAGConfig()
        pipeline = RAGPipeline(config=config)
        logger.info("RAG Pipeline initialized successfully")
        return pipeline
    except Exception as e:
        logger.error(f"Failed to initialize RAG Pipeline: {e}")
        raise RuntimeError(f"Failed to initialize RAG Pipeline: {e}")


# --- Query Caching ---
@st.cache_data(ttl=UIConfig.CACHE_TTL)
def cached_query(pipeline_hash: str, query: str) -> str:
    """
    Cache frequent queries to improve response time.

    Args:
        pipeline_hash: Unique identifier for the pipeline
        query: User query string

    Returns:
        str: Cached response
    """
    # This is a placeholder - actual implementation would depend on your RAG pipeline
    # For now, we'll skip caching to avoid complexity
    return None


# --- Session State Management ---
def initialize_session_state() -> None:
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pipeline_loaded" not in st.session_state:
        st.session_state.pipeline_loaded = False


# --- Chat Functions ---
def display_chat_history() -> None:
    """Display chat messages from history."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def add_message_to_history(role: str, content: str) -> None:
    """
    Add a message to chat history.

    Args:
        role: Message role ('user' or 'assistant')
        content: Message content
    """
    st.session_state.messages.append({"role": role, "content": content})


def export_chat_history() -> str:
    """
    Export chat history as markdown.

    Returns:
        str: Chat history formatted as markdown
    """
    if not st.session_state.messages:
        return "# Chat History\n\n*No messages yet*"

    chat_md = "# Chat History\n\n"
    for msg in st.session_state.messages:
        role = "**User**" if msg["role"] == "user" else "**Assistant**"
        chat_md += f"{role}: {msg['content']}\n\n"
    return chat_md


# --- Enhanced Query Handler ---
def handle_query(pipeline: RAGPipeline, prompt: str) -> None:
    """
    Handle user query and display response with enhanced error handling.

    Args:
        pipeline: RAG pipeline instance
        prompt: User query
    """
    # Add user message to chat history
    add_message_to_history("user", prompt)

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response
    with st.chat_message("assistant"):
        try:
            with st.spinner(UIConfig.MESSAGES["analyzing_spinner"]):
                response = pipeline.query(prompt)
                st.markdown(response)
                add_message_to_history("assistant", response)
                logger.info(f"Query processed successfully: {prompt[:50]}...")

        except ConnectionError:
            error_msg = UIConfig.MESSAGES["connection_error"]
            st.error(error_msg)
            add_message_to_history("assistant", error_msg)
            logger.error(f"Connection error during query: {prompt[:50]}...")

        except TimeoutError:
            error_msg = UIConfig.MESSAGES["timeout_error"]
            st.error(error_msg)
            add_message_to_history("assistant", error_msg)
            logger.error(f"Timeout error during query: {prompt[:50]}...")

        except Exception as e:
            error_msg = UIConfig.MESSAGES["general_error"].format(error=str(e))
            st.error(error_msg)
            add_message_to_history("assistant", error_msg)
            logger.error(f"Query processing failed: {e}")

            # Add retry button
            if st.button(
                UIConfig.MESSAGES["retry_button"],
                key=f"retry_{len(st.session_state.messages)}",
            ):
                st.rerun()


# --- File Validation ---
def validate_file_upload(uploaded_file) -> Tuple[bool, str]:
    """
    Validate uploaded file before processing.

    Args:
        uploaded_file: Streamlit uploaded file object

    Returns:
        Tuple[bool, str]: (is_valid, error_message)
    """
    if uploaded_file.size > UIConfig.MAX_FILE_SIZE_MB * 1024 * 1024:
        return False, UIConfig.MESSAGES["file_too_large"].format(
            max_size=UIConfig.MAX_FILE_SIZE_MB
        )

    if not uploaded_file.name.lower().endswith(tuple(UIConfig.SUPPORTED_FILE_TYPES)):
        supported_types = ", ".join(UIConfig.SUPPORTED_FILE_TYPES)
        return False, UIConfig.MESSAGES["unsupported_file"].format(
            supported_types=supported_types
        )

    return True, ""


# --- Enhanced File Upload Handler ---
def handle_file_upload(uploaded_file) -> bool:
    """
    Processes an uploaded file with proper resource management.

    Args:
        uploaded_file: Streamlit uploaded file object

    Returns:
        bool: True if processing successful, False otherwise
    """
    # Validate file first
    is_valid, error_message = validate_file_upload(uploaded_file)
    if not is_valid:
        st.error(error_message)
        return False

    # Validate uploaded file object
    if not uploaded_file or not uploaded_file.name:
        st.error("Invalid file upload. Please try again.")
        return False

    # Cleanup previous resources if switching files
    if f"temp_dir_{uploaded_file.name}" in st.session_state:
        cleanup_temp_resources()

    # Create session-specific temp directory
    session_id = hash(uploaded_file.name + str(uploaded_file.size))
    temp_dir = tempfile.mkdtemp(prefix=f"rag_session_{abs(session_id)}_")

    try:
        spinner_text = UIConfig.MESSAGES["processing_spinner"].format(
            filename=uploaded_file.name
        )
        with st.spinner(spinner_text):
            # Add progress bar
            progress_bar = st.progress(0)

            # Save file
            progress_bar.progress(25)
            temp_filepath = os.path.join(temp_dir, uploaded_file.name)

            # Validate file content
            file_content = uploaded_file.getvalue()
            if not file_content:
                raise ValueError("Uploaded file is empty")

            with open(temp_filepath, "wb") as f:
                f.write(file_content)

            # Verify file was saved correctly
            if not os.path.exists(temp_filepath) or os.path.getsize(temp_filepath) == 0:
                raise ValueError("Failed to save uploaded file")

            logger.info(
                f"File saved to: {temp_filepath}, size: {os.path.getsize(temp_filepath)} bytes"
            )

            # Setup pipeline
            progress_bar.progress(50)
            db_persist_dir = Path(temp_dir) / "chroma_db_session"
            temp_config = RAGConfig(db_persist_directory=db_persist_dir)

            progress_bar.progress(75)
            custom_pipeline = RAGPipeline(config=temp_config)

            # Convert to string for the pipeline (some RAG pipelines expect string paths)
            file_path_str = str(temp_filepath)
            logger.info(f"Setting up pipeline with file path: {file_path_str}")

            custom_pipeline.setup_pipeline(file_path=file_path_str)

            # Store in session state
            progress_bar.progress(100)
            st.session_state["custom_pipeline"] = custom_pipeline
            st.session_state["active_document"] = uploaded_file.name
            st.session_state[f"temp_dir_{uploaded_file.name}"] = temp_dir
            st.session_state.messages = []

            progress_bar.empty()
            logger.info(f"Successfully processed file: {uploaded_file.name}")
            return True

    except Exception as e:
        # Cleanup on failure
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)

        error_msg = UIConfig.MESSAGES["processing_failed"].format(error=str(e))
        st.error(error_msg)
        logger.error(f"Error processing {uploaded_file.name}: {e}")
        logger.error(f"Full error details: {repr(e)}")
        return False


# --- Enhanced Sidebar ---
def create_sidebar() -> None:
    """Create sidebar with enhanced features and better organization."""
    with st.sidebar:
        # Document Information
        st.header(UIConfig.MESSAGES["document_info"])
        if "active_document" in st.session_state:
            current_doc = st.session_state["active_document"]
            st.info(
                UIConfig.MESSAGES["currently_chatting"].format(doc_name=current_doc)
            )
        else:
            st.info(UIConfig.MESSAGES["default_doc_info"])

        # Query Tips
        st.header(UIConfig.MESSAGES["query_tips"])
        st.markdown(UIConfig.MESSAGES["tips_content"])

        # Actions
        st.header(UIConfig.MESSAGES["actions"])

        # Clear chat button
        if st.button(UIConfig.MESSAGES["clear_chat"], type="secondary"):
            st.session_state.messages = []
            st.rerun()

        # Export chat history
        if st.session_state.messages:
            chat_export = export_chat_history()
            st.download_button(
                label="📥 Export Chat History",
                data=chat_export,
                file_name="chat_history.md",
                mime="text/markdown",
                type="secondary",
            )

        # Display chat statistics
        if st.session_state.messages:
            user_messages = len(
                [m for m in st.session_state.messages if m["role"] == "user"]
            )
            st.metric(UIConfig.MESSAGES["questions_asked"], user_messages)

        # File Upload Section
        st.header(UIConfig.MESSAGES["upload_section"])
        uploaded_file = st.file_uploader(
            UIConfig.MESSAGES["upload_help"],
            type=UIConfig.SUPPORTED_FILE_TYPES,
            accept_multiple_files=False,
            help=f"Maximum file size: {UIConfig.MAX_FILE_SIZE_MB}MB",
        )

        # Handle file upload
        if uploaded_file is not None:
            if st.button("🔄 Process File", type="primary"):
                if handle_file_upload(uploaded_file):
                    st.success(f"✅ Successfully processed {uploaded_file.name}")
                    st.rerun()

        # Switch back to default document
        if "custom_pipeline" in st.session_state:
            if st.button(UIConfig.MESSAGES["switch_default"], type="secondary"):
                cleanup_temp_resources()
                if "custom_pipeline" in st.session_state:
                    del st.session_state["custom_pipeline"]
                if "active_document" in st.session_state:
                    del st.session_state["active_document"]
                st.session_state.messages = []
                st.rerun()


# --- Main Application ---
def main() -> None:
    """Main application function with enhanced error handling and UX."""
    # --- Page Configuration ---
    st.set_page_config(**UIConfig.PAGE_CONFIG)

    # --- Initialize Session State ---
    initialize_session_state()

    # --- Header ---
    st.title(UIConfig.MESSAGES["main_title"])
    st.markdown(UIConfig.MESSAGES["main_description"])

    # --- Create Sidebar ---
    create_sidebar()

    # --- Load Pipeline ---
    pipeline: Optional[RAGPipeline] = None
    active_doc_name = UIConfig.MESSAGES["default_doc"]

    # Determine which pipeline to use
    if "custom_pipeline" in st.session_state:
        pipeline = st.session_state["custom_pipeline"]
        active_doc_name = st.session_state["active_document"]
    else:
        # Load the default, cached pipeline
        try:
            pipeline = load_rag_pipeline()
            st.session_state.pipeline_loaded = True
        except RuntimeError as e:
            st.error(f"❌ Failed to load the default system: {str(e)}")
            st.stop()

    # --- Chat Interface ---
    st.subheader(UIConfig.MESSAGES["chat_title"].format(doc_name=active_doc_name))

    # Display chat history
    display_chat_history()

    # User input
    if prompt := st.chat_input(UIConfig.MESSAGES["query_placeholder"]):
        if pipeline:
            handle_query(pipeline, prompt)
        else:
            st.error(UIConfig.MESSAGES["system_not_ready"])

    # Add help text for new users
    if not st.session_state.messages:
        st.info(
            "👋 Welcome! Start by asking a question about the document, or upload your own file to analyze."
        )


if __name__ == "__main__":
    main()
