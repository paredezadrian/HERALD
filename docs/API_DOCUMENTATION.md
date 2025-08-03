# HERALD API Documentation

## Overview

HERALD (Hybrid Efficient Reasoning Architecture for Local Deployment) is a CPU-optimized, long-context, reasoning-capable AI architecture. This document provides comprehensive API documentation for all components.

## Table of Contents

1. [Core Engine](#core-engine)
2. [Tokenizer](#tokenizer)
3. [Memory Management](#memory-management)
4. [Reasoning Modules](#reasoning-modules)
5. [API Server](#api-server)
6. [CLI Interface](#cli-interface)
7. [Configuration](#configuration)

## Core Engine

### NeuroEngine

The main inference engine that orchestrates all components.

#### Initialization

```python
from core.engine import NeuroEngine, ModelConfig, InferenceConfig

# Default configuration
engine = NeuroEngine()

# Custom configuration
model_config = ModelConfig(
    num_transformer_layers=12,
    hidden_dim=768,
    num_heads=12,
    active_memory_size=8192,
    compressed_memory_size=32768
)

inference_config = InferenceConfig(
    max_new_tokens=100,
    temperature=0.7,
    top_p=0.9,
    top_k=50
)

engine = NeuroEngine(model_config, inference_config)
```

#### Model Loading

```python
# Load a .herald model file
success = engine.load_model("path/to/model.herald")
if success:
    print("Model loaded successfully")
else:
    print("Model loading failed")
```

#### Text Generation

```python
# Generate text from a prompt
output = engine.generate(
    prompt="Generate a response to this prompt",
    max_new_tokens=100,
    temperature=0.7,
    top_p=0.9,
    top_k=50,
    repetition_penalty=1.1
)
print(output)
```

#### Performance Monitoring

```python
# Get performance statistics
stats = engine.get_performance_stats()
print(f"Average inference time: {stats['avg_inference_time']:.3f}s")
print(f"Cache hit rate: {stats['cache_hit_rate']:.2f}")
print(f"Memory usage: {stats['memory_usage']:.2f}GB")
```

#### Memory Management

```python
# Clear attention cache
engine.clear_cache()

# Optimize memory usage
engine.optimize_memory()
```

### ModelConfig

Configuration for the model architecture.

```python
from core.engine import ModelConfig

config = ModelConfig(
    # Model architecture
    model_name="HERALD-v1.0",
    model_version="1.0.0",
    architecture="hybrid_transformer_mamba",
    
    # Transformer configuration
    num_transformer_layers=12,
    hidden_dim=768,
    num_heads=12,
    head_dim=64,
    dropout=0.1,
    activation="gelu",
    
    # Mamba configuration
    num_mamba_blocks=6,
    state_dim=1024,
    selective_scan=True,
    linear_complexity=True,
    
    # Mixture of Experts
    num_experts=8,
    routing_accuracy=0.85,
    load_balancing=True,
    
    # Memory configuration
    active_memory_size=8192,
    compressed_memory_size=32768,
    chunk_size=1024,
    chunk_overlap=128,
    
    # Performance targets
    max_context_length=1000000,
    peak_ram_usage=11.8,
    token_generation_speed=0.8,
    model_load_time=0.7,
    
    # Compression settings
    compression_ratio=8.5,
    quantization="int8",
    sparse_matrices=True,
    lz4_compression=True,
    
    # Hardware optimization
    cpu_optimization=True,
    avx512_support=True,
    intel_mkl=True,
    memory_mapping=True,
    simd_vectorization=True,
    bf16_precision=True
)
```

### InferenceConfig

Configuration for inference parameters.

```python
from core.engine import InferenceConfig

config = InferenceConfig(
    # Generation parameters
    max_new_tokens=100,
    temperature=0.7,
    top_p=0.9,
    top_k=50,
    repetition_penalty=1.1,
    do_sample=True,
    
    # Token IDs
    pad_token_id=0,
    eos_token_id=2,
    bos_token_id=1,
    
    # Memory management
    enable_attention_caching=True,
    enable_compression=True,
    memory_mapping=True,
    
    # Performance settings
    batch_size=1,
    use_cache=True,
    return_attention_weights=False
)
```

## Tokenizer

### ASCTokenizer

Advanced Symbolic Compression tokenizer with multi-tier processing.

#### Initialization

```python
from core.tokenizer import ASCTokenizer

# Basic configuration
config = {
    'vocabulary': {'<pad>': 0, '<bos>': 1, '<eos>': 2},
    'vocab_size': 50000,
    'compression_target': 3.2,
    'byte_level': True,
    'symbolic_tokens': True,
    'wordpiece': True
}

tokenizer = ASCTokenizer(config)
```

#### Tokenization

```python
# Tokenize text
result = tokenizer.tokenize("Hello, world!")
print(f"Tokens: {len(result.tokens)}")
print(f"Compression ratio: {result.compression_ratio:.2f}")

# Access token information
for token in result.tokens:
    print(f"Text: {token.text}, ID: {token.token_id}, Tier: {token.tier}")
```

#### Detokenization

```python
# Detokenize tokens back to text
tokens = [Token(text="", token_id=1, tier=1, frequency=0),
          Token(text="", token_id=4, tier=1, frequency=0)]
text = tokenizer.detokenize(tokens)
print(text)
```

#### Vocabulary Management

```python
# Get vocabulary statistics
stats = tokenizer.get_vocabulary_stats()
print(f"Vocabulary size: {stats['size']}")
print(f"Average frequency: {stats['avg_frequency']:.2f}")

# Add new tokens
tokenizer.add_tokens(["new", "tokens"])
```

## Memory Management

### MultiTierMemoryManager

Multi-tier memory system for efficient context management.

#### Initialization

```python
from core.memory import MultiTierMemoryManager, MemoryConfig

config = MemoryConfig(
    tier1_capacity=8192,      # Active working memory
    tier2_capacity=32768,     # Compressed summaries
    tier3_capacity=1000000,   # Archived knowledge
    chunk_size=1024,
    chunk_overlap=128,
    compression_ratio=4.0
)

memory_manager = MultiTierMemoryManager(config)
```

#### Context Management

```python
# Add context to memory
tokens = np.array([1, 2, 3, 4, 5], dtype=np.int32)
context_id = memory_manager.add_context(tokens)

# Retrieve context
context = memory_manager.get_context(context_id)
print(f"Context length: {len(context)}")

# Update context
memory_manager.update_context(context_id, new_tokens)
```

#### Memory Optimization

```python
# Get memory statistics
stats = memory_manager.get_memory_stats()
print(f"Tier 1 usage: {stats['tier1_usage']:.2f}%")
print(f"Tier 2 usage: {stats['tier2_usage']:.2f}%")
print(f"Tier 3 usage: {stats['tier3_usage']:.2f}%")

# Optimize memory usage
memory_manager.optimize_memory()

# Clear memory
memory_manager.clear_memory()
```

#### Attention Caching

```python
# Cache attention weights
attention_weights = np.random.rand(10, 10)
memory_manager.cache_attention(context_id, attention_weights)

# Retrieve cached attention
cached = memory_manager.get_cached_attention(context_id)
if cached is not None:
    print("Cached attention found")
```

## Reasoning Modules

### Logic Engine

Boolean satisfiability and first-order logic reasoning.

```python
from reasoning.logic import LogicEngine

engine = LogicEngine()

# Boolean satisfiability
clauses = [
    [1, 2],      # x1 OR x2
    [-1, 3],     # NOT x1 OR x3
    [-2, -3]     # NOT x2 OR NOT x3
]

result = engine.solve_sat(clauses, num_variables=3)
if result['satisfiable']:
    print(f"Solution: {result['assignment']}")
else:
    print("Unsatisfiable")

# First-order logic
formula = "forall x (P(x) -> Q(x))"
result = engine.evaluate_fol(formula, interpretation={})
print(f"Valid: {result['valid']}")
```

### Causal Reasoning

Causal inference and dependency analysis.

```python
from reasoning.causal import CausalEngine

engine = CausalEngine()

# Build causal graph
graph = engine.build_dependency_graph([
    ("A", "B"),
    ("B", "C"),
    ("A", "C")
])

# Intervention analysis
intervention = engine.analyze_intervention(graph, "B", "A")
print(f"Effect: {intervention['effect']}")

# Confounding detection
confounders = engine.detect_confounders(graph, "A", "C")
print(f"Confounders: {confounders}")
```

### Temporal Reasoning

Temporal logic and event sequence modeling.

```python
from reasoning.temporal import TemporalEngine

engine = TemporalEngine()

# Event sequence modeling
events = [
    {"time": 0, "event": "start"},
    {"time": 10, "event": "process"},
    {"time": 20, "event": "end"}
]

model = engine.model_event_sequence(events)
print(f"Duration: {model['duration']}")

# Temporal relationship reasoning
relationship = engine.analyze_temporal_relationship(
    "event1", "event2", "before"
)
print(f"Relationship: {relationship['valid']}")
```

### Router

Mixture-of-Experts routing system.

```python
from reasoning.router import MoERouter

router = MoERouter(num_experts=8, routing_accuracy=0.85)

# Route input to experts
input_tensor = np.random.rand(768)
expert_outputs = router.route(input_tensor, context_id="test")

# Get routing statistics
stats = router.get_routing_stats()
print(f"Load balancing: {stats['load_balancing_score']:.3f}")
print(f"Routing accuracy: {stats['routing_accuracy']:.3f}")
```

## API Server

### FastAPI Server

RESTful API interface for HERALD.

#### Starting the Server

```bash
# Start the server
python -m api.server

# Or with custom configuration
python -m api.server --host 0.0.0.0 --port 8000
```

#### API Endpoints

**POST /generate**
Generate text from a prompt.

```bash
curl -X POST "http://localhost:8000/generate" \
     -H "Content-Type: application/json" \
     -d '{
       "prompt": "Generate a response",
       "max_new_tokens": 100,
       "temperature": 0.7
     }'
```

Response:
```json
{
  "generated_text": "Generated response...",
  "tokens_generated": 100,
  "generation_time": 0.5
}
```

**POST /load-model**
Load a model file.

```bash
curl -X POST "http://localhost:8000/load-model" \
     -H "Content-Type: application/json" \
     -d '{
       "model_path": "/path/to/model.herald"
     }'
```

**GET /health**
Health check endpoint.

```bash
curl "http://localhost:8000/health"
```

Response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "memory_usage": 2.5
}
```

**GET /stats**
Get performance statistics.

```bash
curl "http://localhost:8000/stats"
```

Response:
```json
{
  "avg_inference_time": 0.3,
  "cache_hit_rate": 0.85,
  "memory_usage": 2.5,
  "tokens_generated": 1000
}
```

## CLI Interface

### Command Line Interface

HERALD provides a comprehensive CLI for model management and inference.

#### Basic Usage

```bash
# Load a model
herald load-model /path/to/model.herald

# Generate text
herald generate "Your prompt here" --max-tokens 100

# Interactive mode
herald chat

# Batch processing
herald batch-process input.txt output.txt
```

#### Command Reference

**Load Model**
```bash
herald load-model <model_path> [options]
  --config <config_file>    Configuration file
  --verbose                 Verbose output
```

**Generate Text**
```bash
herald generate <prompt> [options]
  --max-tokens <n>          Maximum tokens to generate
  --temperature <t>          Sampling temperature
  --top-p <p>               Nucleus sampling parameter
  --top-k <k>               Top-k sampling
  --repetition-penalty <r>  Repetition penalty
```

**Chat Mode**
```bash
herald chat [options]
  --model <model_path>      Model to use
  --system-prompt <text>    System prompt
  --history <file>          Conversation history file
```

**Batch Processing**
```bash
herald batch-process <input_file> <output_file> [options]
  --batch-size <n>          Batch size
  --max-tokens <n>          Maximum tokens per generation
  --format <format>         Output format (json, text)
```

**Performance Testing**
```bash
herald benchmark [options]
  --model <model_path>      Model to benchmark
  --prompts <file>          Prompt file
  --runs <n>                Number of runs
  --output <file>           Output file
```

## Configuration

### Configuration Files

HERALD uses YAML configuration files for various settings.

#### Model Configuration

`config/model_config.yaml`:
```yaml
model:
  name: "HERALD-v1.0"
  version: "1.0.0"
  architecture: "hybrid_transformer_mamba"
  
transformer:
  num_layers: 12
  hidden_dim: 768
  num_heads: 12
  head_dim: 64
  dropout: 0.1
  activation: "gelu"
  
mamba:
  num_blocks: 6
  state_dim: 1024
  selective_scan: true
  linear_complexity: true
  
experts:
  num_experts: 8
  routing_accuracy: 0.85
  load_balancing: true
  
memory:
  active_size: 8192
  compressed_size: 32768
  chunk_size: 1024
  chunk_overlap: 128
  
performance:
  max_context_length: 1000000
  peak_ram_usage: 11.8
  token_generation_speed: 0.8
  model_load_time: 0.7
  
compression:
  ratio: 8.5
  quantization: "int8"
  sparse_matrices: true
  lz4_compression: true
  
hardware:
  cpu_optimization: true
  avx512_support: true
  intel_mkl: true
  memory_mapping: true
  simd_vectorization: true
  bf16_precision: true
```

#### Runtime Configuration

`config/runtime_config.yaml`:
```yaml
inference:
  max_new_tokens: 100
  temperature: 0.7
  top_p: 0.9
  top_k: 50
  repetition_penalty: 1.1
  do_sample: true
  
tokens:
  pad_token_id: 0
  eos_token_id: 2
  bos_token_id: 1
  
memory:
  enable_attention_caching: true
  enable_compression: true
  memory_mapping: true
  
performance:
  batch_size: 1
  use_cache: true
  return_attention_weights: false
```

#### Deployment Configuration

`config/deployment_config.yaml`:
```yaml
server:
  host: "0.0.0.0"
  port: 8000
  workers: 4
  max_connections: 100
  
api:
  rate_limit: 100
  timeout: 30
  max_request_size: "10MB"
  
logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: "herald.log"
  
security:
  enable_cors: true
  allowed_origins: ["*"]
  api_key_required: false
```

### Environment Variables

HERALD supports configuration via environment variables:

```bash
# Model configuration
export HERALD_MODEL_PATH="/path/to/model.herald"
export HERALD_CONFIG_PATH="/path/to/config.yaml"

# Server configuration
export HERALD_HOST="0.0.0.0"
export HERALD_PORT="8000"

# Performance configuration
export HERALD_MAX_MEMORY="12GB"
export HERALD_NUM_THREADS="8"

# Logging configuration
export HERALD_LOG_LEVEL="INFO"
export HERALD_LOG_FILE="herald.log"
```

## Error Handling

### Common Exceptions

**ModelInitializationError**
Raised when model loading fails.

```python
from core.engine import ModelInitializationError

try:
    engine.load_model("invalid.herald")
except ModelInitializationError as e:
    print(f"Model loading failed: {e}")
```

**InferenceError**
Raised when text generation fails.

```python
from core.engine import InferenceError

try:
    output = engine.generate("prompt")
except InferenceError as e:
    print(f"Generation failed: {e}")
```

**MemoryError**
Raised when memory operations fail.

```python
from core.memory import MemoryError

try:
    memory_manager.add_context(tokens)
except MemoryError as e:
    print(f"Memory operation failed: {e}")
```

### Error Codes

| Code | Description |
|------|-------------|
| 1001 | Model file not found |
| 1002 | Invalid model format |
| 1003 | Model integrity check failed |
| 2001 | Tokenizer not loaded |
| 2002 | Memory manager not loaded |
| 3001 | Generation failed |
| 3002 | Model not loaded |
| 4001 | Memory allocation failed |
| 4002 | Context not found |

## Performance Optimization

### Memory Optimization

```python
# Enable memory mapping
config = ModelConfig(memory_mapping=True)

# Use compression
config = ModelConfig(lz4_compression=True)

# Optimize memory usage
engine.optimize_memory()
```

### CPU Optimization

```python
# Enable Intel MKL
config = ModelConfig(intel_mkl=True)

# Enable AVX-512
config = ModelConfig(avx512_support=True)

# Enable SIMD vectorization
config = ModelConfig(simd_vectorization=True)
```

### Quantization

```python
# Use int8 quantization
config = ModelConfig(quantization="int8")

# Use bf16 precision
config = ModelConfig(bf16_precision=True)
```

## Examples

### Complete Text Generation Example

```python
from core.engine import NeuroEngine, ModelConfig, InferenceConfig

# Initialize engine
model_config = ModelConfig(
    num_transformer_layers=12,
    hidden_dim=768,
    active_memory_size=8192
)

inference_config = InferenceConfig(
    max_new_tokens=100,
    temperature=0.7
)

engine = NeuroEngine(model_config, inference_config)

# Load model
if engine.load_model("model.herald"):
    # Generate text
    output = engine.generate(
        prompt="Explain quantum computing in simple terms.",
        max_new_tokens=200,
        temperature=0.8
    )
    print(output)
    
    # Get performance stats
    stats = engine.get_performance_stats()
    print(f"Generation time: {stats['avg_inference_time']:.3f}s")
else:
    print("Failed to load model")
```

### API Server Example

```python
from api.server import create_app
import uvicorn

app = create_app()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### CLI Example

```bash
# Load model and start chat
herald load-model model.herald
herald chat --system-prompt "You are a helpful assistant."

# Batch process a file
herald batch-process prompts.txt outputs.json --max-tokens 100

# Run benchmarks
herald benchmark --model model.herald --prompts test_prompts.txt
```

## Troubleshooting

### Common Issues

**Model Loading Slow**
- Check if Intel MKL is available
- Reduce model size or use quantization
- Enable memory mapping

**High Memory Usage**
- Reduce active memory size
- Enable compression
- Use memory mapping

**Poor Performance**
- Enable CPU optimizations
- Use quantization
- Optimize batch size

**API Errors**
- Check model is loaded
- Verify input format
- Check memory availability

### Debug Mode

Enable debug logging:

```python
import logging
logging.getLogger().setLevel(logging.DEBUG)
```

Or via environment variable:

```bash
export HERALD_LOG_LEVEL="DEBUG"
```

### Performance Monitoring

Monitor performance metrics:

```python
# Get detailed stats
stats = engine.get_performance_stats()
print(f"Memory usage: {stats['memory_usage']:.2f}GB")
print(f"Cache hit rate: {stats['cache_hit_rate']:.2f}")
print(f"Average inference time: {stats['avg_inference_time']:.3f}s")
```

## Version History

### v1.0.0
- Initial release
- Core engine implementation
- Basic API and CLI
- Multi-tier memory system
- Reasoning modules

### Future Versions
- Multi-language support
- Plugin architecture
- Distributed processing
- Advanced security features 