import logging
from typing import Optional

from UI import UIConfig, UIComponents, UIHandlers, pipeline_loader
from src.pipeline import RAGPipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    """Main application function with enhanced error handling and UX."""

    # Initialize UI components and handlers
    components = UIComponents()
    handlers = UIHandlers()

    # Setup page configuration
    components.setup_page_config()

    # Initialize session state
    handlers.initialize_session_state()

    # Setup cleanup on exit
    handlers.setup_cleanup_on_exit()

    # Render header
    components.render_header()

    # Render sidebar and get interaction states
    sidebar_state = components.render_sidebar()

    # Process sidebar interactions
    handlers.process_sidebar_interactions(sidebar_state)

    # Get active pipeline
    from pipeline_loader import load_rag_pipeline
    pipeline = load_rag_pipeline()
    if pipeline is None:
        components.show_error(
            "❌ System initialization failed. Please refresh the page."
        )
        return

    # Get active document name
    active_doc_name = components.get_active_document_name()

    # Render chat interface
    components.render_chat_title(active_doc_name)

    # Display chat history
    components.render_chat_history()

    # Handle user input
    user_input = components.render_chat_input()
    if user_input:
        if pipeline:
            handlers.handle_query(pipeline, user_input)
        else:
            components.show_error(UIConfig.MESSAGES["system_not_ready"])

    # Show welcome message for new users
    components.render_welcome_message()


if __name__ == "__main__":
    main()
