"""
HERALD Layers Module
Neural network layers and attention mechanisms

This module contains the neural network components:
- Fast Transformer: CPU-optimized transformer layers
- Mamba Layer: State space model layers
- Attention: Dual-chunk attention mechanisms
- Quantization: Precision optimization utilities
"""

__version__ = "1.0.0"
__author__ = "HERALD Development Team"

# Layer components
from .fast_transformer import FastTransformer
from .mamba_layer import MambaLayer
from .attention import DualChunkAttention
from .quantization import QuantizationLayer

__all__ = [
    "FastTransformer",
    "MambaLayer",
    "DualChunkAttention", 
    "QuantizationLayer"
] 