import streamlit as st
import logging
from typing import Optional, List, Dict, Any
from pathlib import Path

from .config import UIConfig
from src.pipeline import RAGPipeline, RAGConfig

logger = logging.getLogger(__name__)


class UIComponents:
    """Handles all UI component creation and rendering."""

    def __init__(self):
        self.config = UIConfig()

    def setup_page_config(self) -> None:
        """Configure Streamlit page settings."""
        st.set_page_config(**self.config.PAGE_CONFIG)

    def render_header(self) -> None:
        """Render the main application header."""
        st.title(self.config.MESSAGES["main_title"])
        st.markdown(self.config.MESSAGES["main_description"])

    def render_chat_title(self, doc_name: str) -> None:
        """Render the chat section title."""
        st.subheader(self.config.MESSAGES["chat_title"].format(doc_name=doc_name))

    def render_welcome_message(self) -> None:
        """Render welcome message for new users."""
        if not st.session_state.get(self.config.SESSION_KEYS["messages"], []):
            st.info(
                "👋 Welcome! Start by asking a question about the document, or upload your own file to analyze."
            )

    def render_chat_input(self) -> Optional[str]:
        """Render chat input field and return user input."""
        return st.chat_input(self.config.MESSAGES["query_placeholder"])

    def render_chat_history(self) -> None:
        """Display chat messages from history."""
        messages = st.session_state.get(self.config.SESSION_KEYS["messages"], [])
        for message in messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    def render_document_info_section(self) -> None:
        """Render document information section in sidebar."""
        st.header(self.config.MESSAGES["document_info"])

        active_doc_key = self.config.SESSION_KEYS["active_document"]
        if active_doc_key in st.session_state:
            current_doc = st.session_state[active_doc_key]
            st.info(
                self.config.MESSAGES["currently_chatting"].format(doc_name=current_doc)
            )
        else:
            st.info(self.config.MESSAGES["default_doc_info"])

    def render_query_tips_section(self) -> None:
        """Render query tips section in sidebar."""
        st.header(self.config.MESSAGES["query_tips"])
        st.markdown(self.config.MESSAGES["tips_content"])

    def render_actions_section(self) -> Dict[str, bool]:
        """Render actions section in sidebar and return button states."""
        st.header(self.config.MESSAGES["actions"])

        actions = {}

        # Clear chat button
        actions["clear_chat"] = st.button(
            self.config.MESSAGES["clear_chat"],
            type="secondary",
            key=self.config.COMPONENT_KEYS["clear_chat"],
        )

        # Export chat history
        messages = st.session_state.get(self.config.SESSION_KEYS["messages"], [])
        if messages:
            chat_export = self._export_chat_history(messages)
            st.download_button(
                label="📥 Export Chat History",
                data=chat_export,
                file_name="chat_history.md",
                mime="text/markdown",
                type="secondary",
                key=self.config.COMPONENT_KEYS["export_chat"],
            )

        # Display chat statistics
        if messages:
            user_messages = len([m for m in messages if m["role"] == "user"])
            st.metric(self.config.MESSAGES["questions_asked"], user_messages)

        return actions

    def render_file_upload_section(self) -> Dict[str, Any]:
        """Render file upload section and return upload state."""
        st.header(self.config.MESSAGES["upload_section"])

        uploaded_file = st.file_uploader(
            self.config.MESSAGES["upload_help"],
            type=self.config.SUPPORTED_FILE_TYPES,
            accept_multiple_files=False,
            help=f"Maximum file size: {self.config.MAX_FILE_SIZE_MB}MB",
        )

        process_clicked = False
        if uploaded_file is not None:
            process_clicked = st.button(
                "🔄 Process File",
                type="primary",
                key=self.config.COMPONENT_KEYS["process_file"],
            )

        return {"uploaded_file": uploaded_file, "process_clicked": process_clicked}

    def render_switch_default_button(self) -> bool:
        """Render switch to default document button."""
        custom_pipeline_key = self.config.SESSION_KEYS["custom_pipeline"]
        if custom_pipeline_key in st.session_state:
            return st.button(
                self.config.MESSAGES["switch_default"],
                type="secondary",
                key=self.config.COMPONENT_KEYS["switch_default"],
            )
        return False

    def render_sidebar(self) -> Dict[str, Any]:
        """Render complete sidebar and return all interaction states."""
        with st.sidebar:
            # Document information
            self.render_document_info_section()

            # Query tips
            self.render_query_tips_section()

            # Actions
            actions = self.render_actions_section()

            # File upload
            upload_state = self.render_file_upload_section()

            # Switch default button
            actions["switch_default"] = self.render_switch_default_button()

            return {"actions": actions, "upload_state": upload_state}

    def show_error(self, message: str) -> None:
        """Display error message."""
        st.error(message)

    def show_success(self, message: str) -> None:
        """Display success message."""
        st.success(message)

    def show_info(self, message: str) -> None:
        """Display info message."""
        st.info(message)

    def show_warning(self, message: str) -> None:
        """Display warning message."""
        st.warning(message)

    def show_spinner(self, message: str):
        """Create and return a spinner context manager."""
        return st.spinner(message)

    def create_progress_bar(self) -> Any:
        """Create and return a progress bar."""
        return st.progress(0)

    def render_retry_button(self, message_count: int) -> bool:
        """Render retry button with unique key."""
        return st.button(
            self.config.MESSAGES["retry_button"],
            key=self.config.get_retry_key(message_count),
        )

    def render_chat_message(self, role: str, content: str) -> None:
        """Render a single chat message."""
        with st.chat_message(role):
            st.markdown(content)

    def _export_chat_history(self, messages: List[Dict[str, str]]) -> str:
        """Export chat history as markdown."""
        if not messages:
            return "# Chat History\n\n*No messages yet*"

        chat_md = "# Chat History\n\n"
        for msg in messages:
            role = "**User**" if msg["role"] == "user" else "**Assistant**"
            chat_md += f"{role}: {msg['content']}\n\n"
        return chat_md

    @st.cache_resource
    def load_rag_pipeline(_self) -> RAGPipeline:
        """Load and cache the RAG pipeline with Docker-compatible paths."""
        try:
            # Use environment-aware paths
            db_path = Path(_self.config.get_db_path())
            config = RAGConfig(db_persist_directory=db_path)
            pipeline = RAGPipeline(config=config)

            # Check if we need to setup the default document
            data_path = Path(_self.config.get_data_path())
            if data_path.exists():
                logger.info(f"Setting up pipeline with default document: {data_path}")
                pipeline.setup_pipeline(file_path=str(data_path))
            else:
                logger.warning(f"Default document not found at: {data_path}")

            logger.info("RAG Pipeline initialized successfully")
            return pipeline
        except Exception as e:
            logger.error(f"Failed to initialize RAG Pipeline: {e}")
            raise RuntimeError(f"Failed to initialize RAG Pipeline: {e}")

    def get_active_document_name(self) -> str:
        """Get the name of the currently active document."""
        active_doc_key = self.config.SESSION_KEYS["active_document"]
        if active_doc_key in st.session_state:
            return st.session_state[active_doc_key]
        return self.config.MESSAGES["default_doc"]
