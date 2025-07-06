FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install OS-level dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user
RUN addgroup --system app && adduser --system --ingroup app app

# Copy requirements.txt first (for better Docker layer caching)
COPY requirements.txt .

# Install Python packages as root (system-wide)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Create necessary directories with proper permissions
RUN mkdir -p /app/data /app/chroma_db /app/temp /home/app/.cache/huggingface /home/app/.streamlit && \
    chown -R app:app /app /home/app

# Copy application files (excluding chroma_db and temp directories)
COPY --chown=app:app src/ ./src/
COPY --chown=app:app data/ ./data/
COPY --chown=app:app app.py .

# Switch to non-root user
USER app

# Set environment variables
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_ENABLE_CORS=false
ENV STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
ENV PYTHONPATH=/app
ENV TMPDIR=/app/temp
# Set HuggingFace cache directory to a writable location
ENV HF_HOME=/home/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/home/app/.cache/huggingface/transformers
ENV SENTENCE_TRANSFORMERS_HOME=/home/app/.cache/huggingface/sentence-transformers

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Run the application
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
