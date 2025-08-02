"""
HERALD Core Module
Hybrid Efficient Reasoning Architecture for Local Deployment

This module contains the core components of the HERALD architecture:
- ASC Tokenizer: Adaptive Symbolic Compression tokenizer
- Memory Manager: Multi-tier context memory system (TODO)
- NeuroEngine: Core inference engine (TODO)
- Model Loader: .herald file format loader (TODO)
"""

__version__ = "1.0.0"
__author__ = "HERALD Development Team"

# Core components
from .tokenizer import ASCTokenizer

__all__ = [
    "ASCTokenizer"
] 