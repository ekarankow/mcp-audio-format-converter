# Audio Format Converter MCP Server Dockerfile
# Multi-stage build for optimized production image

# Build stage
FROM python:3.11-slim as builder

# Set build arguments
ARG DEBIAN_FRONTEND=noninteractive

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy dependency files and source code
COPY pyproject.toml ./
COPY src/ ./src/

# Create virtual environment and install Python dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e .

# Production stage
FROM python:3.11-slim as production

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PATH="/opt/venv/bin:$PATH"

# Set build arguments
ARG DEBIAN_FRONTEND=noninteractive

# Create non-root user for security
RUN groupadd --gid 1000 audioconv && \
    useradd --uid 1000 --gid audioconv --shell /bin/bash --create-home audioconv

# Install runtime system dependencies
RUN apt-get update && apt-get install -y \
    # Audio processing libraries (for pydub and audio format support)
    libsndfile1 \
    libavcodec-extra \
    libavformat-dev \
    libavutil-dev \
    libswresample-dev \
    # Network tools for debugging and health checks
    curl \
    netcat-openbsd \
    # Clean up
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy Python virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Set working directory
WORKDIR /app

# Copy application code and configuration files
COPY --chown=audioconv:audioconv src/ ./src/
COPY --chown=audioconv:audioconv pyproject.toml ./
COPY --chown=audioconv:audioconv run_server.py ./
COPY --chown=audioconv:audioconv config.example ./

# Create directories for logs and data
RUN mkdir -p /app/logs /app/data && \
    chown -R audioconv:audioconv /app

# Switch to non-root user
USER audioconv

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD sh -c 'curl -f http://localhost:$PORT/health || exit 1'

# Default environment variables
ENV HOST=0.0.0.0
ENV PORT=8080
ENV LOG_LEVEL=INFO
ENV DEFAULT_SAMPLE_RATE=16000
ENV DEFAULT_SAMPLE_WIDTH=2
ENV DEFAULT_CHANNELS=1

# Default command
CMD ["sh", "-c", "python -m src.audio_format_converter_mcp.server --host $HOST --port $PORT"]