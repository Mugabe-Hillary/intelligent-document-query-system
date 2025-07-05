from pathlib import Path

# Project Root
PROJECT_ROOT = Path(__file__).parent.parent

# Data Configuration
SOURCE_DOCUMENT_PATH = PROJECT_ROOT / "data" / "through_the_dark_continent.txt"

# Vector Store Configuration
DB_PERSIST_DIRECTORY = PROJECT_ROOT / "db"
CHROMA_COLLECTION_NAME = "dark_continent_collection"

# RAG Pipeline Configuration
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

K_RETRIEVED_CHUNKS = 4
