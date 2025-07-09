from typing import Dict, Any, List
import os


class UIConfig:
    """Configuration class for UI settings and messages."""

    # Streamlit page configuration
    PAGE_CONFIG: Dict[str, Any] = {
        "page_title": "Intelligent Document Query",
        "page_icon": "🧮",
        "layout": "wide",
        "initial_sidebar_state": "expanded",
    }

    # File upload configuration
    SUPPORTED_FILE_TYPES: List[str] = ["txt"]
    MAX_FILE_SIZE_MB: int = 10
    CACHE_TTL: int = 300  # 5 minutes

    # Docker and local paths
    DOCKER_TEMP_DIR: str = "/app/temp"
    DOCKER_DB_PATH: str = "/app/chroma_db"
    DOCKER_DATA_PATH: str = "/app/data/through_the_dark_continent.txt"
    LOCAL_TEMP_DIR: str = "temp"
    LOCAL_DB_PATH: str = "chroma_db"
    LOCAL_DATA_PATH: str = "data/through_the_dark_continent.txt"

    # UI Messages
    MESSAGES: Dict[str, str] = {
        # Document related
        "default_doc": "Through the Dark Continent by Henry M. Stanley",
        "currently_chatting": "Currently chatting with: **{doc_name}**",
        "default_doc_info": "**Document:** Through the Dark Continent by Henry M. Stanley",
        # Upload related
        "upload_help": "Upload a .txt file to chat with it.",
        "upload_section": "⬆️ Upload Your Own Document",
        "processing_spinner": "Processing '{filename}'... This may take a moment.",
        "file_too_large": "File too large. Maximum size: {max_size}MB",
        "unsupported_file": "Unsupported file type. Supported: {supported_types}",
        "processing_failed": "Failed to process uploaded file: {error}",
        # Query related
        "query_placeholder": "Ask a question about the document...",
        "analyzing_spinner": "Analyzing document...",
        "system_not_ready": "System not ready. Please wait for initialization to complete.",
        # Error messages
        "connection_error": "⚠️ Connection error. Please check your internet connection and try again.",
        "timeout_error": "⏱️ Request timed out. Please try a simpler question or try again later.",
        "general_error": "❌ I encountered an error: {error}",
        # Action buttons
        "retry_button": "🔄 Retry",
        "clear_chat": "Clear Chat History",
        "switch_default": "Switch back to Default Document",
        # Stats and info
        "questions_asked": "Questions Asked",
        # Section headers
        "document_info": "📋 Document Information",
        "query_tips": "💡 Query Tips",
        "actions": "🔧 Actions",
        "chat_title": "💬 Chat: {doc_name}",
        "main_title": "🧮 Intelligent Document Query System",
        # Main description
        "main_description": """
        Ask questions about **'Through the Dark Continent'** by Henry M. Stanley. 
        This system uses AI to provide accurate, contextual answers based on the document content.
        """,
        # Tips content
        "tips_content": """
        - Ask specific questions about events, people, or places
        - Use clear, complete sentences
        - Try questions like:
          - "What challenges did Stanley face?"
          - "Describe the expedition route"
          - "Who were the key people mentioned?"
        """,
    }

    # Streamlit component keys
    COMPONENT_KEYS: Dict[str, str] = {
        "retry_template": "retry_{message_count}",
        "process_file": "process_file_button",
        "clear_chat": "clear_chat_button",
        "switch_default": "switch_default_button",
        "export_chat": "export_chat_button",
    }

    # Session state keys
    SESSION_KEYS: Dict[str, str] = {
        "messages": "messages",
        "pipeline_loaded": "pipeline_loaded",
        "custom_pipeline": "custom_pipeline",
        "active_document": "active_document",
        "temp_dir_prefix": "temp_dir_",
    }

    @classmethod
    def get_supported_file_types_str(cls) -> str:
        """Get supported file types as a comma-separated string."""
        return ", ".join(cls.SUPPORTED_FILE_TYPES)

    @classmethod
    def get_retry_key(cls, message_count: int) -> str:
        """Generate a retry button key based on message count."""
        return cls.COMPONENT_KEYS["retry_template"].format(message_count=message_count)

    @classmethod
    def get_temp_dir_key(cls, filename: str) -> str:
        """Generate a temporary directory key for a filename."""
        return f"{cls.SESSION_KEYS['temp_dir_prefix']}{filename}"

    @classmethod
    def get_data_path(cls) -> str:
        """Return the data path depending on environment (Docker or local)."""
        import os
        if os.path.exists(cls.DOCKER_DATA_PATH):
            return cls.DOCKER_DATA_PATH
        elif os.path.exists(cls.LOCAL_DATA_PATH):
            return cls.LOCAL_DATA_PATH
        else:
            return cls.DOCKER_DATA_PATH  # Default to Docker path for error messages


    @classmethod
    def get_db_path(cls) -> str:
        if os.path.exists(cls.LOCAL_DB_PATH):
            return cls.LOCAL_DB_PATH
        elif os.path.exists(cls.DOCKER_DB_PATH):
            return cls.DOCKER_DB_PATH
        else:
            return cls.LOCAL_DB_PATH  # Default to local, not Docker

