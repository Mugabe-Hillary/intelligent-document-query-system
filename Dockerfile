# Multi-stage build for optimal image size
FROM python:3.11-slim AS builder

# Set working directory
WORKDIR /app

# Install build dependencies in a single layer
RUN apk add --no-cache --virtual .build-deps \
    build-base \
    libffi-dev \
    && apk add --no-cache \
    curl

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install Python packages to user directory and clean up in same layer
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --user -r requirements.txt

# Final stage - runtime image
FROM python:3.11-slim

# Install runtime dependencies only
RUN apk add --no-cache \
    curl \
    && addgroup -S app \
    && adduser -S app -G app

# Set working directory
WORKDIR /app

# Copy Python packages from builder stage
COPY --from=builder /root/.local /home/app/.local

# Create necessary directories with proper permissions
RUN mkdir -p /app/data /app/chroma_db /app/temp /home/app/.cache/huggingface /home/app/.streamlit && \
    chown -R app:app /app /home/app

# Copy application files (use specific COPY commands)
COPY --chown=app:app src/ ./src/
COPY --chown=app:app data/ ./data/
COPY --chown=app:app app.py .

# Switch to non-root user
USER app

# Set environment variables
ENV STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_ENABLE_CORS=false \
    STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    PYTHONPATH=/app \
    TMPDIR=/app/temp \
    HF_HOME=/home/app/.cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/home/app/.cache/huggingface/sentence-transformers \
    PATH=/home/app/.local/bin:$PATH

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Run the application
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]