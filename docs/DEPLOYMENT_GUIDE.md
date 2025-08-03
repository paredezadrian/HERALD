# HERALD Deployment Guide

## Overview

This guide provides comprehensive instructions for deploying HERALD (Hybrid Efficient Reasoning Architecture for Local Deployment) in various environments, from local development to production systems.

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Local Development Setup](#local-development-setup)
3. [Docker Deployment](#docker-deployment)
4. [Production Deployment](#production-deployment)
5. [Cloud Deployment](#cloud-deployment)
6. [Performance Tuning](#performance-tuning)
7. [Monitoring and Logging](#monitoring-and-logging)
8. [Security Considerations](#security-considerations)
9. [Troubleshooting](#troubleshooting)

## System Requirements

### Minimum Requirements

- **CPU**: Intel i5-11320H or equivalent (4+ cores)
- **RAM**: 16GB minimum, 32GB recommended
- **Storage**: 50GB available space
- **OS**: Windows 10/11, Linux (Ubuntu 20.04+), macOS 12+
- **Python**: 3.11 or higher

### Recommended Requirements

- **CPU**: Intel i7/i9 or AMD Ryzen 7/9 (8+ cores)
- **RAM**: 32GB or higher
- **Storage**: 100GB SSD
- **GPU**: Optional (CPU-optimized architecture)
- **Network**: 1Gbps+ for API serving

### Hardware Optimization

HERALD is optimized for CPU inference with the following features:

- **Intel MKL**: Math Kernel Library for optimized BLAS operations
- **AVX-512**: Advanced Vector Extensions for SIMD operations
- **Memory Mapping**: Efficient memory usage for large models
- **Quantization**: int8/bf16 precision for reduced memory usage

## Local Development Setup

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/your-org/herald.git
cd herald

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration Setup

Create configuration files:

```bash
# Create config directory
mkdir -p config

# Copy example configurations
cp config/example_model_config.yaml config/model_config.yaml
cp config/example_runtime_config.yaml config/runtime_config.yaml
cp config/example_deployment_config.yaml config/deployment_config.yaml
```

### 3. Model Preparation

```bash
# Download or prepare your model file
# Model should be in .herald format
# Example model structure:
# model.herald
# ├── config/          # Model configuration
# ├── weights/         # Model weights
# ├── tokenizer/       # Tokenizer data
# └── metadata/        # Model metadata
```

### 4. Basic Testing

```bash
# Run unit tests
python -m pytest tests/unit/

# Run integration tests
python -m pytest tests/integration/

# Run performance benchmarks
python run_benchmarks.py
```

## Docker Deployment

### 1. Dockerfile

The project includes a `Dockerfile` optimized for production:

```dockerfile
# Use Python 3.11 slim image
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 herald && chown -R herald:herald /app
USER herald

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["python", "-m", "api.server"]
```

### 2. Docker Compose

Create `docker-compose.yml` for easy deployment:

```yaml
version: '3.8'

services:
  herald-api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
      - ./config:/app/config
      - ./logs:/app/logs
    environment:
      - HERALD_HOST=0.0.0.0
      - HERALD_PORT=8000
      - HERALD_LOG_LEVEL=INFO
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  herald-cli:
    build: .
    volumes:
      - ./models:/app/models
      - ./config:/app/config
      - ./data:/app/data
    environment:
      - HERALD_LOG_LEVEL=INFO
    command: ["python", "cli.py", "chat"]
    depends_on:
      - herald-api
```

### 3. Building and Running

```bash
# Build the Docker image
docker build -t herald:latest .

# Run with Docker Compose
docker-compose up -d

# Check logs
docker-compose logs -f herald-api

# Stop services
docker-compose down
```

### 4. Production Docker Setup

For production, use multi-stage builds and security best practices:

```dockerfile
# Multi-stage build for production
FROM python:3.11-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libopenblas-dev

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.11-slim

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    libopenblas-base \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

# Set working directory
WORKDIR /app

# Copy application code
COPY . .

# Create non-root user with minimal permissions
RUN useradd -r -s /bin/false herald && \
    chown -R herald:herald /app

USER herald

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["python", "-m", "api.server"]
```

## Production Deployment

### 1. System Preparation

```bash
# Update system packages
sudo apt-get update && sudo apt-get upgrade -y

# Install system dependencies
sudo apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    build-essential \
    libopenblas-dev \
    nginx \
    supervisor

# Create application user
sudo useradd -r -s /bin/false herald
sudo mkdir -p /opt/herald
sudo chown herald:herald /opt/herald
```

### 2. Application Deployment

```bash
# Clone repository
sudo -u herald git clone https://github.com/your-org/herald.git /opt/herald/app

# Create virtual environment
sudo -u herald python3.11 -m venv /opt/herald/venv

# Install dependencies
sudo -u herald /opt/herald/venv/bin/pip install -r /opt/herald/app/requirements.txt

# Create directories
sudo -u herald mkdir -p /opt/herald/{models,config,logs,data}
```

### 3. Configuration

Create production configuration files:

```bash
# Model configuration
sudo -u herald tee /opt/herald/config/model_config.yaml > /dev/null <<EOF
model:
  name: "HERALD-v1.0"
  version: "1.0.0"
  architecture: "hybrid_transformer_mamba"

transformer:
  num_layers: 12
  hidden_dim: 768
  num_heads: 12

memory:
  active_size: 8192
  compressed_size: 32768

performance:
  max_context_length: 1000000
  peak_ram_usage: 11.8
EOF

# Runtime configuration
sudo -u herald tee /opt/herald/config/runtime_config.yaml > /dev/null <<EOF
inference:
  max_new_tokens: 100
  temperature: 0.7
  top_p: 0.9

server:
  host: "127.0.0.1"
  port: 8000
  workers: 4
EOF
```

### 4. Service Configuration

Create systemd service:

```bash
sudo tee /etc/systemd/system/herald.service > /dev/null <<EOF
[Unit]
Description=HERALD AI Service
After=network.target

[Service]
Type=simple
User=herald
Group=herald
WorkingDirectory=/opt/herald/app
Environment=PATH=/opt/herald/venv/bin
ExecStart=/opt/herald/venv/bin/python -m api.server
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable herald
sudo systemctl start herald
```

### 5. Nginx Configuration

```bash
sudo tee /etc/nginx/sites-available/herald > /dev/null <<EOF
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        
        # Timeout settings
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }
}
EOF

# Enable site
sudo ln -s /etc/nginx/sites-available/herald /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

### 6. SSL/TLS Setup

```bash
# Install Certbot
sudo apt-get install -y certbot python3-certbot-nginx

# Obtain SSL certificate
sudo certbot --nginx -d your-domain.com

# Auto-renewal
sudo crontab -e
# Add: 0 12 * * * /usr/bin/certbot renew --quiet
```

## Cloud Deployment

### 1. AWS Deployment

#### EC2 Setup

```bash
# Launch EC2 instance (t3.2xlarge or larger)
# Ubuntu 22.04 LTS recommended

# Connect to instance
ssh -i your-key.pem ubuntu@your-instance-ip

# Follow production deployment steps above
```

#### AWS Lambda (Serverless)

Create `lambda_function.py`:

```python
import json
from api.server import create_app
from mangum import Mangum

app = create_app()
handler = Mangum(app)
```

Deploy with AWS SAM:

```yaml
# template.yaml
AWSTemplateFormatVersion: '2010-09-09'
Transform: AWS::Serverless-2016-10-31

Resources:
  HeraldFunction:
    Type: AWS::Serverless::Function
    Properties:
      CodeUri: ./
      Handler: lambda_function.handler
      Runtime: python3.11
      Timeout: 300
      MemorySize: 3008
      Environment:
        Variables:
          HERALD_LOG_LEVEL: INFO
```

### 2. Google Cloud Platform

#### Compute Engine

```bash
# Create instance
gcloud compute instances create herald-instance \
    --zone=us-central1-a \
    --machine-type=n2-standard-8 \
    --image-family=ubuntu-2204-lts \
    --image-project=ubuntu-os-cloud \
    --boot-disk-size=100GB

# Follow production deployment steps
```

#### Cloud Run

```dockerfile
# Dockerfile for Cloud Run
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8080
ENV PORT=8080

CMD ["python", "-m", "api.server"]
```

Deploy:

```bash
# Build and push to Container Registry
gcloud builds submit --tag gcr.io/PROJECT_ID/herald

# Deploy to Cloud Run
gcloud run deploy herald \
    --image gcr.io/PROJECT_ID/herald \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated \
    --memory 4Gi \
    --cpu 2
```

### 3. Azure Deployment

#### Azure Container Instances

```bash
# Build and push to Azure Container Registry
az acr build --registry your-registry --image herald:latest .

# Deploy to Container Instances
az container create \
    --resource-group your-rg \
    --name herald-container \
    --image your-registry.azurecr.io/herald:latest \
    --ports 8000 \
    --memory 4 \
    --cpu 2
```

## Performance Tuning

### 1. System Optimization

```bash
# CPU optimization
echo 'performance' | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Memory optimization
echo 'vm.swappiness=10' | sudo tee -a /etc/sysctl.conf
sudo sysctl -p

# Disk optimization (if using SSD)
echo 'noatime' | sudo tee -a /etc/fstab
```

### 2. Application Tuning

```yaml
# config/performance_config.yaml
hardware:
  cpu_optimization: true
  avx512_support: true
  intel_mkl: true
  memory_mapping: true
  simd_vectorization: true
  bf16_precision: true

memory:
  active_size: 8192
  compressed_size: 32768
  chunk_size: 1024
  chunk_overlap: 128

compression:
  ratio: 8.5
  quantization: "int8"
  sparse_matrices: true
  lz4_compression: true
```

### 3. Load Balancing

For high-traffic deployments, use multiple instances:

```yaml
# docker-compose.yml with load balancer
version: '3.8'

services:
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - herald-api-1
      - herald-api-2

  herald-api-1:
    build: .
    environment:
      - HERALD_PORT=8001
    expose:
      - "8001"

  herald-api-2:
    build: .
    environment:
      - HERALD_PORT=8002
    expose:
      - "8002"
```

## Monitoring and Logging

### 1. Application Monitoring

```python
# utils/monitoring.py
import time
import psutil
import logging
from prometheus_client import Counter, Histogram, Gauge

# Metrics
request_counter = Counter('herald_requests_total', 'Total requests')
request_duration = Histogram('herald_request_duration_seconds', 'Request duration')
memory_usage = Gauge('herald_memory_bytes', 'Memory usage in bytes')

def monitor_request(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        request_counter.inc()
        
        try:
            result = func(*args, **kwargs)
            request_duration.observe(time.time() - start_time)
            return result
        except Exception as e:
            logging.error(f"Request failed: {e}")
            raise
    
    return wrapper
```

### 2. Logging Configuration

```python
# config/logging_config.py
import logging
import logging.handlers
import os

def setup_logging():
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.handlers.RotatingFileHandler(
                'logs/herald.log',
                maxBytes=10*1024*1024,  # 10MB
                backupCount=5
            ),
            logging.StreamHandler()
        ]
    )
```

### 3. Health Checks

```python
# api/health.py
import psutil
import time
from core.engine import NeuroEngine

def health_check():
    """Comprehensive health check."""
    checks = {
        'system': check_system_health(),
        'memory': check_memory_health(),
        'model': check_model_health(),
        'performance': check_performance_health()
    }
    
    overall_healthy = all(checks.values())
    
    return {
        'healthy': overall_healthy,
        'timestamp': time.time(),
        'checks': checks
    }

def check_system_health():
    """Check system resources."""
    cpu_percent = psutil.cpu_percent()
    memory_percent = psutil.virtual_memory().percent
    
    return cpu_percent < 90 and memory_percent < 90

def check_memory_health():
    """Check memory usage."""
    memory = psutil.virtual_memory()
    return memory.percent < 85

def check_model_health():
    """Check model status."""
    # Implementation depends on your model loading strategy
    return True

def check_performance_health():
    """Check performance metrics."""
    # Implementation depends on your performance monitoring
    return True
```

## Security Considerations

### 1. Network Security

```bash
# Configure firewall
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow ssh
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw enable
```

### 2. Application Security

```python
# api/security.py
from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import secrets

security = HTTPBearer()

def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify API key."""
    if credentials.credentials != os.getenv('HERALD_API_KEY'):
        raise HTTPException(status_code=401, detail="Invalid API key")
    return credentials.credentials

def rate_limit(request):
    """Implement rate limiting."""
    # Implementation depends on your rate limiting strategy
    pass
```

### 3. Data Security

```python
# utils/security.py
import hashlib
import hmac
import os

def verify_model_signature(model_path, signature):
    """Verify model file integrity."""
    with open(model_path, 'rb') as f:
        data = f.read()
    
    expected_signature = hmac.new(
        os.getenv('HERALD_SECRET_KEY').encode(),
        data,
        hashlib.sha256
    ).hexdigest()
    
    return hmac.compare_digest(signature, expected_signature)
```

## Troubleshooting

### 1. Common Issues

#### High Memory Usage

```bash
# Check memory usage
free -h
ps aux --sort=-%mem | head -10

# Optimize memory
python -c "
from core.engine import NeuroEngine
engine = NeuroEngine()
engine.optimize_memory()
"
```

#### Slow Performance

```bash
# Check CPU usage
top
htop

# Check for bottlenecks
python -m cProfile -o profile.stats your_script.py
python -c "import pstats; pstats.Stats('profile.stats').sort_stats('cumulative').print_stats(10)"
```

#### Model Loading Issues

```bash
# Check model file integrity
python -c "
from core.loader import ModelLoader
loader = ModelLoader()
print(loader.validate_model('model.herald'))
"

# Check file permissions
ls -la model.herald
```

### 2. Log Analysis

```bash
# Monitor logs in real-time
tail -f logs/herald.log

# Search for errors
grep -i error logs/herald.log

# Analyze performance
grep "inference_time" logs/herald.log | awk '{sum+=$NF; count++} END {print "Average:", sum/count}'
```

### 3. Performance Debugging

```python
# Enable debug mode
import logging
logging.getLogger().setLevel(logging.DEBUG)

# Profile specific functions
import cProfile
import pstats

def profile_function(func, *args, **kwargs):
    profiler = cProfile.Profile()
    profiler.enable()
    result = func(*args, **kwargs)
    profiler.disable()
    
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(10)
    
    return result
```

### 4. Recovery Procedures

#### Service Recovery

```bash
# Restart service
sudo systemctl restart herald

# Check service status
sudo systemctl status herald

# View recent logs
sudo journalctl -u herald -f
```

#### Data Recovery

```bash
# Backup configuration
cp -r config/ config_backup_$(date +%Y%m%d_%H%M%S)/

# Restore from backup
cp -r config_backup_20231201_143022/* config/
```

## Maintenance

### 1. Regular Maintenance

```bash
# Update system packages
sudo apt-get update && sudo apt-get upgrade -y

# Clean up old logs
find logs/ -name "*.log" -mtime +30 -delete

# Update application
cd /opt/herald/app
git pull origin main
/opt/herald/venv/bin/pip install -r requirements.txt
sudo systemctl restart herald
```

### 2. Backup Procedures

```bash
# Backup configuration and models
tar -czf herald_backup_$(date +%Y%m%d).tar.gz \
    config/ models/ logs/

# Automated backup script
#!/bin/bash
BACKUP_DIR="/backup/herald"
DATE=$(date +%Y%m%d_%H%M%S)
tar -czf "$BACKUP_DIR/herald_$DATE.tar.gz" \
    /opt/herald/config /opt/herald/models /opt/herald/logs

# Keep only last 7 days of backups
find "$BACKUP_DIR" -name "herald_*.tar.gz" -mtime +7 -delete
```

### 3. Monitoring Setup

```bash
# Install monitoring tools
sudo apt-get install -y prometheus node-exporter grafana

# Configure Prometheus
sudo tee /etc/prometheus/prometheus.yml > /dev/null <<EOF
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'herald'
    static_configs:
      - targets: ['localhost:8000']
EOF

# Start monitoring services
sudo systemctl enable prometheus node-exporter grafana-server
sudo systemctl start prometheus node-exporter grafana-server
```

This comprehensive deployment guide covers all aspects of deploying HERALD in various environments, from local development to production systems, with detailed instructions for performance tuning, monitoring, security, and maintenance. 