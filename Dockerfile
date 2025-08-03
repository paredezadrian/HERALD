# Multi-stage build for HERALD
# Stage 1: Builder
FROM python:3.11-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libopenblas-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Production
FROM python:3.11-slim

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    libopenblas-base \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user with minimal permissions
RUN useradd -r -s /bin/false herald && \
    mkdir -p /app && \
    chown -R herald:herald /app

# Set working directory
WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /root/.local /home/herald/.local

# Copy application code
COPY --chown=herald:herald . .

# Create necessary directories
RUN mkdir -p /app/logs /app/models /app/config /app/data && \
    chown -R herald:herald /app

# Switch to non-root user
USER herald

# Set environment variables
ENV PYTHONPATH=/app
ENV HERALD_LOG_LEVEL=INFO
ENV HERALD_HOST=0.0.0.0
ENV HERALD_PORT=8000

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["python", "-m", "api.server"] 