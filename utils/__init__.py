"""
HERALD Utils Module
Utility functions and helper components

This module contains utility components:
- Compression: Data compression and decompression utilities
- Metrics: Performance monitoring and metrics collection
- Validation: Input validation and error handling
"""

__version__ = "1.0.0"
__author__ = "HERALD Development Team"

# Utility components
try:
    from .compression import (
        LZ4Compressor,
        QuantizationUtils,
        SparseMatrix,
        CompressionManager,
        compress_model_weights,
        decompress_model_weights,
        get_compression_stats
    )
except ImportError:
    # Compression module not yet implemented
    pass

# TODO: Implement other utility modules
# from .metrics import MetricsCollector
# from .validation import InputValidator

__all__ = [
    "LZ4Compressor",
    "QuantizationUtils", 
    "SparseMatrix",
    "CompressionManager",
    "compress_model_weights",
    "decompress_model_weights",
    "get_compression_stats"
] 