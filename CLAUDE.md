# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Testing
```bash
# Run all tests
python -m pytest

# Run specific test categories
python -m pytest tests/unit/                    # Unit tests
python -m pytest tests/integration/             # Integration tests
python -m pytest tests/test_logic.py           # Logic engine tests

# Run integration tests with detailed output
python tests/integration/run_integration_tests.py

# Run tests with coverage
python -m pytest --cov=.
```

### Development Server
```bash
# Start development server
python cli.py serve --host localhost --port 8000

# Load and test a model
python cli.py load-model path/to/model.herald

# Generate text
python cli.py generate "Your prompt here" --max-tokens 100

# Run benchmarks
python cli.py benchmark
```

### Code Quality
```bash
# Run linting (if configured)
flake8 .

# Run type checking (if configured)
mypy .

# Format code (if configured)
black .
```

## Architecture Overview

HERALD is a CPU-optimized AI architecture designed for local deployment with three main layers:

### Core Components (`core/`)
- **NeuroEngine (`engine.py`)**: Main inference orchestrator with hybrid transformer-mamba architecture
- **ASC Tokenizer (`tokenizer.py`)**: Adaptive symbolic compression tokenizer with 3.2:1 compression ratio
- **Memory Manager (`memory.py`)**: Multi-tier memory system (active/compressed/archived)

### Neural Layers (`layers/`)
- **Transformer Layers (`fast_transformer.py`)**: CPU-optimized 12-layer transformer with int8 quantization
- **Mamba Layers (`mamba_layer.py`)**: State space models for linear complexity long sequences
- **Attention Mechanisms (`attention.py`)**: Chunked attention for 1M+ token contexts
- **Quantization (`quantization.py`)**: Precision optimization (bf16/int8)

### Reasoning Engine (`reasoning/`)
- **Logic Engine (`logic.py`)**: Boolean SAT solver, FOL engine, rule chain inference
- **Causal Reasoning**: Dependency graphs and intervention analysis
- **Temporal Processing**: Event sequences and temporal logic

## Key Implementation Details

### Logic Engine Status
The logic engine currently has:
- ✅ 25/32 tests passing
- ✅ Core SAT solving, forward chaining, basic inference working
- ❌ 7 failing tests for advanced features (backward chaining, cycle detection, FOL proofs)

### Memory Architecture
- **Tier 1**: 8,192 tokens active working memory
- **Tier 2**: 32,768 tokens compressed (4:1 ratio)  
- **Tier 3**: Unlimited archived knowledge via reference indexing
- **Chunking**: 1,024-token chunks with 128-token overlap

### Performance Targets
- Context: 1M tokens maximum
- Speed: 0.8 seconds per 100 tokens
- RAM: 11.8GB peak usage
- Model load: 0.7 seconds from .herald files

## Configuration Files

### Model Configuration (`config/model_config.yaml`)
Contains architecture parameters, performance targets, compression settings, and hardware optimization flags.

### Runtime Configuration (`config/runtime_config.yaml`)
Runtime settings for deployment and inference parameters.

### Dependencies (`requirements.txt`)
Core dependencies include numpy, numba, fastapi, lz4, and pytest for testing.

## Testing Strategy

### Test Structure
- `tests/unit/`: Component-level tests for core modules
- `tests/integration/`: End-to-end functionality tests  
- `tests/test_logic.py`: Comprehensive logic engine testing
- `tests/benchmarks/`: Performance validation tests

### Integration Testing
Use `tests/integration/run_integration_tests.py` for comprehensive system validation with performance reporting.

## Development Patterns

### Module Integration
All core modules follow the pattern: tokenizer → memory manager → neural engine → reasoning modules

### Error Handling
Components use structured error handling with specific exception types for different failure modes.

### Performance Optimization
- CPU-first design using Intel MKL and AVX-512 instructions
- Memory mapping for large weight matrices
- SIMD vectorization for matrix operations
- Gradient checkpointing disabled (inference-only)

## File Formats

### .herald Files
Compressed knowledge format with magic number "LITE", containing quantized weights, vocabulary data, and symbolic rules with 8.5:1 compression ratio.