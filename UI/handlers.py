import streamlit as st
import logging
import tempfile
import os
import shutil
import atexit
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

from .config import UIConfig
from .components import UIComponents
from src.pipeline import RAGPipeline, RAGConfig

logger = logging.getLogger(__name__)


class UIHandlers:
    """Handles all UI event processing and business logic."""

    def __init__(self):
        self.config = UIConfig()
        self.components = UIComponents()

    def initialize_session_state(self) -> None:
        """Initialize session state variables."""
        if self.config.SESSION_KEYS["messages"] not in st.session_state:
            st.session_state[self.config.SESSION_KEYS["messages"]] = []
        if self.config.SESSION_KEYS["pipeline_loaded"] not in st.session_state:
            st.session_state[self.config.SESSION_KEYS["pipeline_loaded"]] = False

    def add_message_to_history(self, role: str, content: str) -> None:
        """Add a message to chat history."""
        messages_key = self.config.SESSION_KEYS["messages"]
        if messages_key not in st.session_state:
            st.session_state[messages_key] = []
        st.session_state[messages_key].append({"role": role, "content": content})

    def clear_chat_history(self) -> None:
        """Clear chat history."""
        st.session_state[self.config.SESSION_KEYS["messages"]] = []

    def handle_query(self, pipeline: RAGPipeline, prompt: str) -> None:
        """Handle user query and display response with enhanced error handling."""
        # Add user message to chat history
        self.add_message_to_history("user", prompt)

        # Display user message
        self.components.render_chat_message("user", prompt)

        # Display assistant response
        try:
            with self.components.show_spinner(
                self.config.MESSAGES["analyzing_spinner"]
            ):
                response = pipeline.query(prompt)
                self.components.render_chat_message("assistant", response)
                self.add_message_to_history("assistant", response)
                logger.info(f"Query processed successfully: {prompt[:50]}...")

        except ConnectionError:
            error_msg = self.config.MESSAGES["connection_error"]
            self.components.show_error(error_msg)
            self.add_message_to_history("assistant", error_msg)
            logger.error(f"Connection error during query: {prompt[:50]}...")

        except TimeoutError:
            error_msg = self.config.MESSAGES["timeout_error"]
            self.components.show_error(error_msg)
            self.add_message_to_history("assistant", error_msg)
            logger.error(f"Timeout error during query: {prompt[:50]}...")

        except Exception as e:
            error_msg = self.config.MESSAGES["general_error"].format(error=str(e))
            self.components.show_error(error_msg)
            self.add_message_to_history("assistant", error_msg)
            logger.error(f"Query processing failed: {e}")

            # Add retry button
            messages_count = len(
                st.session_state.get(self.config.SESSION_KEYS["messages"], [])
            )
            if self.components.render_retry_button(messages_count):
                st.rerun()

    def validate_file_upload(self, uploaded_file) -> Tuple[bool, str]:
        """Validate uploaded file before processing."""
        if uploaded_file.size > self.config.MAX_FILE_SIZE_MB * 1024 * 1024:
            return False, self.config.MESSAGES["file_too_large"].format(
                max_size=self.config.MAX_FILE_SIZE_MB
            )

        if not uploaded_file.name.lower().endswith(
            tuple(self.config.SUPPORTED_FILE_TYPES)
        ):
            return False, self.config.MESSAGES["unsupported_file"].format(
                supported_types=self.config.get_supported_file_types_str()
            )

        return True, ""

    def get_temp_dir(self) -> str:
        """Get a temporary directory that works in Docker containers."""
        if os.path.exists(self.config.DOCKER_TEMP_DIR) and os.access(
            self.config.DOCKER_TEMP_DIR, os.W_OK
        ):
            return self.config.DOCKER_TEMP_DIR
        return tempfile.gettempdir()

    def cleanup_temp_resources(self) -> None:
        """Clean up temporary resources when switching documents."""
        temp_prefix = self.config.SESSION_KEYS["temp_dir_prefix"]
        for key in list(st.session_state.keys()):
            if key.startswith(temp_prefix):
                temp_dir = st.session_state[key]
                if os.path.exists(temp_dir):
                    try:
                        shutil.rmtree(temp_dir, ignore_errors=True)
                        logger.info(f"Cleaned up temporary directory: {temp_dir}")
                    except Exception as e:
                        logger.warning(
                            f"Failed to cleanup temp directory {temp_dir}: {e}"
                        )
                del st.session_state[key]

    def handle_file_upload(self, uploaded_file) -> bool:
        """Process an uploaded file with proper resource management."""
        # Validate file first
        is_valid, error_message = self.validate_file_upload(uploaded_file)
        if not is_valid:
            self.components.show_error(error_message)
            return False

        # Validate uploaded file object
        if not uploaded_file or not uploaded_file.name:
            self.components.show_error("Invalid file upload. Please try again.")
            return False

        # Cleanup previous resources if switching files
        temp_dir_key = self.config.get_temp_dir_key(uploaded_file.name)
        if temp_dir_key in st.session_state:
            self.cleanup_temp_resources()

        # Create session-specific temp directory
        session_id = hash(uploaded_file.name + str(uploaded_file.size))
        temp_base = self.get_temp_dir()
        temp_dir = os.path.join(temp_base, f"rag_session_{abs(session_id)}")
        os.makedirs(temp_dir, exist_ok=True)

        try:
            spinner_text = self.config.MESSAGES["processing_spinner"].format(
                filename=uploaded_file.name
            )
            with self.components.show_spinner(spinner_text):
                # Add progress bar
                progress_bar = self.components.create_progress_bar()

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
                if (
                    not os.path.exists(temp_filepath)
                    or os.path.getsize(temp_filepath) == 0
                ):
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

                file_path_str = str(temp_filepath)
                logger.info(f"Setting up pipeline with file path: {file_path_str}")

                custom_pipeline.setup_pipeline(file_path=file_path_str)

                # Store in session state
                progress_bar.progress(100)
                st.session_state[self.config.SESSION_KEYS["custom_pipeline"]] = (
                    custom_pipeline
                )
                st.session_state[self.config.SESSION_KEYS["active_document"]] = (
                    uploaded_file.name
                )
                st.session_state[temp_dir_key] = temp_dir
                self.clear_chat_history()

                progress_bar.empty()
                logger.info(f"Successfully processed file: {uploaded_file.name}")
                return True

        except Exception as e:
            # Cleanup on failure
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)

            error_msg = self.config.MESSAGES["processing_failed"].format(error=str(e))
            self.components.show_error(error_msg)
            logger.error(f"Error processing {uploaded_file.name}: {e}")
            return False

    def handle_clear_chat(self) -> None:
        """Handle clear chat action."""
        self.clear_chat_history()
        st.rerun()

    def handle_switch_default(self) -> None:
        """Handle switch to default document action."""
        self.cleanup_temp_resources()

        # Remove custom pipeline and active document from session state
        custom_pipeline_key = self.config.SESSION_KEYS["custom_pipeline"]
        active_doc_key = self.config.SESSION_KEYS["active_document"]

        if custom_pipeline_key in st.session_state:
            del st.session_state[custom_pipeline_key]
        if active_doc_key in st.session_state:
            del st.session_state[active_doc_key]

        self.clear_chat_history()
        st.rerun()

    def get_active_pipeline(self) -> Optional[RAGPipeline]:
        """Get the currently active RAG pipeline."""
        custom_pipeline_key = self.config.SESSION_KEYS["custom_pipeline"]

        # Check if custom pipeline exists
        if custom_pipeline_key in st.session_state:
            return st.session_state[custom_pipeline_key]

        # Load default pipeline
        try:
            pipeline = self.components.load_rag_pipeline()
            st.session_state[self.config.SESSION_KEYS["pipeline_loaded"]] = True
            return pipeline
        except RuntimeError as e:
            self.components.show_error(
                f"❌ Failed to load the default system: {str(e)}"
            )
            return None

    def process_sidebar_interactions(self, sidebar_state: Dict[str, Any]) -> None:
        """Process all sidebar interactions."""
        actions = sidebar_state.get("actions", {})
        upload_state = sidebar_state.get("upload_state", {})

        # Handle clear chat
        if actions.get("clear_chat"):
            self.handle_clear_chat()

        # Handle file upload
        if upload_state.get("process_clicked") and upload_state.get("uploaded_file"):
            if self.handle_file_upload(upload_state["uploaded_file"]):
                self.components.show_success(
                    f"✅ Successfully processed {upload_state['uploaded_file'].name}"
                )
                st.rerun()

        # Handle switch to default document
        if actions.get("switch_default"):
            self.handle_switch_default()

    def setup_cleanup_on_exit(self) -> None:
        """Register cleanup function for session end."""
        atexit.register(self.cleanup_temp_resources)
