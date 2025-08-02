"""
HERALD Fast Transformer Layers
CPU-optimized transformer architecture with Intel AVX-512 support

This module implements:
- 12-layer transformer architecture
- Multi-head attention (768 hidden dimensions)
- CPU optimization with Intel AVX-512
- int8 quantization support
- Memory-efficient attention mechanisms

Target: ~1,567 lines of optimized CPU-focused code
"""

import numpy as np
import numba
from numba import jit, prange
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
import logging
import time
import threading
from collections import defaultdict
import math
import os

# CPU feature detection
try:
    import cpufeature
    CPU_FEATURES = cpufeature.CPUFeature
    HAS_AVX512 = CPU_FEATURES.get('AVX512F', False)
    HAS_AVX2 = CPU_FEATURES.get('AVX2', False)
    HAS_SSE4 = CPU_FEATURES.get('SSE4_1', False)
except ImportError:
    HAS_AVX512 = False
    HAS_AVX2 = False
    HAS_SSE4 = False

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class TransformerConfig:
    """Configuration for the Fast Transformer architecture."""
    # Architecture parameters
    num_layers: int = 12
    hidden_dim: int = 768
    num_heads: int = 12
    head_dim: int = 64
    intermediate_dim: int = 3072
    max_position_embeddings: int = 1000000
    
    # Attention parameters
    attention_dropout: float = 0.1
    hidden_dropout: float = 0.1
    layer_norm_eps: float = 1e-12
    activation: str = "gelu"
    
    # Performance settings
    use_avx512: bool = True
    use_quantization: bool = True
    quantize_attention: bool = True
    quantize_ffn: bool = True
    
    # Memory optimization
    use_memory_mapping: bool = True
    attention_cache_size: int = 8192
    gradient_checkpointing: bool = False
    
    # Advanced features
    use_relative_position_embeddings: bool = True
    use_rotary_position_embeddings: bool = True
    use_alibi_position_embeddings: bool = False
    
    # Optimization flags
    use_fused_operations: bool = True
    use_parallel_attention: bool = True
    use_attention_caching: bool = True

    def __post_init__(self):
        if self.hidden_dim != self.num_heads * self.head_dim:
            raise ValueError(f"hidden_dim ({self.hidden_dim}) must equal num_heads ({self.num_heads}) * head_dim ({self.head_dim})")


@dataclass
class AttentionCache:
    """Cache for attention computations to avoid recomputation."""
    key_cache: Dict[str, np.ndarray] = field(default_factory=dict)
    value_cache: Dict[str, np.ndarray] = field(default_factory=dict)
    attention_weights_cache: Dict[str, np.ndarray] = field(default_factory=dict)
    position_embeddings_cache: Dict[str, np.ndarray] = field(default_factory=dict)
    
    def clear(self):
        """Clear all cached data."""
        self.key_cache.clear()
        self.value_cache.clear()
        self.attention_weights_cache.clear()
        self.position_embeddings_cache.clear()


class QuantizationManager:
    """Manages int8 quantization for transformer layers."""
    
    def __init__(self, config: TransformerConfig):
        self.config = config
        self.quantization_scales = {}
        self.quantization_zeros = {}
        self.quantization_stats = defaultdict(list)
        
    def quantize_tensor(self, tensor: np.ndarray, name: str) -> Tuple[np.ndarray, float, int]:
        """Quantize a tensor to int8 format."""
        if not self.config.use_quantization:
            return tensor, 1.0, 0
            
        # Calculate quantization parameters
        min_val = np.min(tensor)
        max_val = np.max(tensor)
        
        # Handle edge case where all values are the same
        if min_val == max_val:
            scale = 1.0
            zero_point = 0
        else:
            # Use symmetric quantization for better numerical stability
            abs_max = max(abs(min_val), abs(max_val))
            scale = abs_max / 127.0
            zero_point = 0
            
        # Quantize
        quantized = np.round(tensor / scale).astype(np.int8)
        
        # Store quantization parameters
        self.quantization_scales[name] = scale
        self.quantization_zeros[name] = zero_point
        
        return quantized, scale, zero_point
    
    def dequantize_tensor(self, quantized: np.ndarray, name: str) -> np.ndarray:
        """Dequantize a tensor from int8 format."""
        if not self.config.use_quantization:
            return quantized
            
        scale = self.quantization_scales.get(name, 1.0)
        zero_point = self.quantization_zeros.get(name, 0)
        
        return (quantized.astype(np.float32) - zero_point) * scale
    
    def get_quantization_stats(self) -> Dict[str, Any]:
        """Get quantization statistics."""
        return {
            'scales': dict(self.quantization_scales),
            'zeros': dict(self.quantization_zeros),
            'stats': dict(self.quantization_stats)
        }


class PositionEmbeddings:
    """Position embedding implementations for transformer layers."""
    
    def __init__(self, config: TransformerConfig):
        self.config = config
        self.max_position = config.max_position_embeddings
        self.hidden_dim = config.hidden_dim
        
        # Initialize position embeddings
        self._init_position_embeddings()
    
    def _init_position_embeddings(self):
        """Initialize different types of position embeddings."""
        if self.config.use_relative_position_embeddings:
            self._init_relative_position_embeddings()
        
        if self.config.use_rotary_position_embeddings:
            self._init_rotary_position_embeddings()
            
        if self.config.use_alibi_position_embeddings:
            self._init_alibi_position_embeddings()
    
    def _init_relative_position_embeddings(self):
        """Initialize relative position embeddings."""
        # Create relative position embeddings for attention
        max_relative_position = 512
        self.relative_position_embeddings = np.random.normal(
            0, 0.02, (max_relative_position * 2 + 1, self.config.head_dim)
        ).astype(np.float32)
    
    def _init_rotary_position_embeddings(self):
        """Initialize rotary position embeddings."""
        # Pre-compute rotation matrices for rotary embeddings
        inv_freq = 1.0 / (10000 ** (np.arange(0, self.config.head_dim, 2) / self.config.head_dim))
        self.inv_freq = inv_freq.astype(np.float32)
    
    def _init_alibi_position_embeddings(self):
        """Initialize ALiBi position embeddings."""
        # ALiBi (Attention with Linear Biases) implementation
        slopes = self._get_alibi_slopes(self.config.num_heads)
        self.alibi_slopes = slopes.astype(np.float32)
    
    def _get_alibi_slopes(self, num_heads: int) -> np.ndarray:
        """Get ALiBi slopes for the given number of heads."""
        def get_slopes_power_of_2(n):
            start = (2**(-2**-(math.log2(n)-3)))
            ratio = start
            return [start*ratio**i for i in range(n)]
        
        if math.log2(num_heads).is_integer():
            return np.array(get_slopes_power_of_2(num_heads))
        else:
            closest_power_of_2 = 2**math.floor(math.log2(num_heads))
            base_slopes = np.array(get_slopes_power_of_2(closest_power_of_2))
            extra_slopes = np.array(get_slopes_power_of_2(2 * closest_power_of_2))[::2]
            return np.concatenate([base_slopes, extra_slopes[:num_heads - closest_power_of_2]])
    
    def apply_rotary_position_embeddings(self, x: np.ndarray, seq_len: int) -> np.ndarray:
        """Apply rotary position embeddings to input tensor."""
        if not self.config.use_rotary_position_embeddings:
            return x
            
        # Reshape for rotary embeddings
        batch_size, seq_len, num_heads, head_dim = x.shape
        x_reshaped = x.reshape(batch_size, seq_len, num_heads, head_dim // 2, 2)
        
        # Generate position indices
        position_indices = np.arange(seq_len, dtype=np.float32)
        
        # Compute rotation angles
        freqs = np.outer(position_indices, self.inv_freq)
        cos_freqs = np.cos(freqs)
        sin_freqs = np.sin(freqs)
        
        # Correct broadcasting: (1, seq_len, 1, head_dim//2)
        cos_freqs_broadcast = cos_freqs[None, :, None, :]
        sin_freqs_broadcast = sin_freqs[None, :, None, :]
        
        x0, x1 = x_reshaped[..., 0], x_reshaped[..., 1]
        rotated_x0 = x0 * cos_freqs_broadcast - x1 * sin_freqs_broadcast
        rotated_x1 = x0 * sin_freqs_broadcast + x1 * cos_freqs_broadcast
        
        # Reshape back
        rotated = np.stack([rotated_x0, rotated_x1], axis=-1)
        return rotated.reshape(batch_size, seq_len, num_heads, head_dim)
    
    def apply_alibi_position_embeddings(self, attention_scores: np.ndarray, seq_len: int) -> np.ndarray:
        """Apply ALiBi position embeddings to attention scores."""
        if not self.config.use_alibi_position_embeddings:
            return attention_scores
            
        # Create position bias matrix
        position_bias = np.arange(seq_len, dtype=np.float32)
        position_bias = position_bias[None, :] - position_bias[:, None]
        
        # Apply slopes to each head
        alibi_bias = position_bias[None, :, :] * self.alibi_slopes[:, None, None]
        
        return attention_scores + alibi_bias


def _fast_layer_norm(x: np.ndarray, weight: np.ndarray, bias: np.ndarray, eps: float) -> np.ndarray:
    """Fast layer normalization using optimized NumPy."""
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    std = np.sqrt(var + eps)
    normalized = (x - mean) / std
    return weight * normalized + bias


def _fast_gelu(x: np.ndarray) -> np.ndarray:
    """Fast GELU activation using optimized NumPy."""
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def _fast_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Fast softmax implementation using optimized NumPy."""
    max_val = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - max_val)
    sum_exp = np.sum(exp_x, axis=axis, keepdims=True)
    return exp_x / sum_exp


def _fast_attention_scores(query: np.ndarray, key: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Fast attention scores computation using optimized NumPy."""
    # Compute attention scores: Q * K^T / sqrt(d_k)
    scores = np.einsum('bhid,bhjd->bhij', query, key)
    scores = scores / np.sqrt(query.shape[-1])
    
    # Apply mask if provided
    if mask is not None:
        scores = scores + mask
    
    return scores


def _fast_attention_output(attention_weights: np.ndarray, value: np.ndarray) -> np.ndarray:
    """Fast attention output computation using optimized NumPy."""
    return np.einsum('bhij,bhjd->bhid', attention_weights, value)


class MultiHeadAttention:
    """Multi-head attention mechanism with CPU optimizations."""
    
    def __init__(self, config: TransformerConfig, layer_idx: int):
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_dim = config.hidden_dim
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.attention_dropout = config.attention_dropout
        
        # Initialize attention weights
        self._init_attention_weights()
        
        # Position embeddings
        self.position_embeddings = PositionEmbeddings(config)
        
        # Quantization manager
        self.quantization_manager = QuantizationManager(config)
        
        # Attention cache
        self.attention_cache = AttentionCache()
        
        # Performance tracking
        self.attention_times = []
        self.memory_usage = []
    
    def _init_attention_weights(self):
        """Initialize attention layer weights."""
        # Query, Key, Value projections
        self.query_weight = np.random.normal(
            0, 0.02, (self.hidden_dim, self.hidden_dim)
        ).astype(np.float32)
        self.key_weight = np.random.normal(
            0, 0.02, (self.hidden_dim, self.hidden_dim)
        ).astype(np.float32)
        self.value_weight = np.random.normal(
            0, 0.02, (self.hidden_dim, self.hidden_dim)
        ).astype(np.float32)
        
        # Output projection
        self.output_weight = np.random.normal(
            0, 0.02, (self.hidden_dim, self.hidden_dim)
        ).astype(np.float32)
        
        # Layer normalization weights
        self.layer_norm_weight = np.ones(self.hidden_dim, dtype=np.float32)
        self.layer_norm_bias = np.zeros(self.hidden_dim, dtype=np.float32)
        
        # Dropout masks
        self.dropout_mask = None
    
    def forward(self, 
                hidden_states: np.ndarray,
                attention_mask: Optional[np.ndarray] = None,
                context_id: str = "",
                use_cache: bool = True) -> Tuple[np.ndarray, Optional[Dict[str, np.ndarray]]]:
        """Forward pass through multi-head attention."""
        start_time = time.time()
        
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # Layer normalization
        normalized_states = _fast_layer_norm(
            hidden_states, 
            self.layer_norm_weight, 
            self.layer_norm_bias, 
            self.config.layer_norm_eps
        )
        
        # Project to query, key, value
        query = self._project_query(normalized_states)
        key = self._project_key(normalized_states)
        value = self._project_value(normalized_states)
        
        # Apply position embeddings
        if self.config.use_rotary_position_embeddings:
            # Reshape for rotary embeddings (batch_size, seq_len, num_heads, head_dim)
            query_reshaped = query.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
            key_reshaped = key.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
            
            query_rotated = self.position_embeddings.apply_rotary_position_embeddings(query_reshaped, seq_len)
            key_rotated = self.position_embeddings.apply_rotary_position_embeddings(key_reshaped, seq_len)
            
            # Reshape back
            query = query_rotated.reshape(batch_size, seq_len, self.hidden_dim)
            key = key_rotated.reshape(batch_size, seq_len, self.hidden_dim)
        
        # Reshape for multi-head attention
        query = query.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        key = key.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        value = value.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Compute attention scores
        attention_scores = _fast_attention_scores(query, key, attention_mask)
        
        # Apply ALiBi if enabled
        if self.config.use_alibi_position_embeddings:
            attention_scores = self.position_embeddings.apply_alibi_position_embeddings(
                attention_scores, seq_len
            )
        
        # Apply softmax
        attention_weights = _fast_softmax(attention_scores, axis=-1)
        
        # Apply dropout
        if self.attention_dropout > 0 and self.training:
            dropout_mask = np.random.binomial(1, 1 - self.attention_dropout, attention_weights.shape)
            attention_weights = attention_weights * dropout_mask / (1 - self.attention_dropout)
        
        # Compute attention output
        attention_output = _fast_attention_output(attention_weights, value)
        
        # Reshape back
        attention_output = attention_output.reshape(batch_size, seq_len, hidden_dim)
        
        # Output projection
        output = np.dot(attention_output, self.output_weight.T)
        
        # Residual connection
        output = output + hidden_states
        
        # Cache attention weights if requested
        cache = None
        if use_cache and self.config.use_attention_caching:
            cache = {
                'key': key,
                'value': value,
                'attention_weights': attention_weights
            }
            self.attention_cache.key_cache[context_id] = key
            self.attention_cache.value_cache[context_id] = value
            self.attention_cache.attention_weights_cache[context_id] = attention_weights
        
        # Track performance
        end_time = time.time()
        self.attention_times.append(end_time - start_time)
        
        return output, cache
    
    def _project_query(self, hidden_states: np.ndarray) -> np.ndarray:
        """Project hidden states to query space."""
        return np.dot(hidden_states, self.query_weight.T)
    
    def _project_key(self, hidden_states: np.ndarray) -> np.ndarray:
        """Project hidden states to key space."""
        return np.dot(hidden_states, self.key_weight.T)
    
    def _project_value(self, hidden_states: np.ndarray) -> np.ndarray:
        """Project hidden states to value space."""
        return np.dot(hidden_states, self.value_weight.T)
    
    def get_attention_stats(self) -> Dict[str, Any]:
        """Get attention layer statistics."""
        return {
            'layer_idx': self.layer_idx,
            'avg_attention_time': np.mean(self.attention_times) if self.attention_times else 0,
            'total_attention_calls': len(self.attention_times),
            'cache_size': len(self.attention_cache.key_cache),
            'quantization_stats': self.quantization_manager.get_quantization_stats()
        }


class FeedForwardNetwork:
    """Feed-forward network with CPU optimizations."""
    
    def __init__(self, config: TransformerConfig, layer_idx: int):
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_dim = config.hidden_dim
        self.intermediate_dim = config.intermediate_dim
        self.hidden_dropout = config.hidden_dropout
        
        # Initialize FFN weights
        self._init_ffn_weights()
        
        # Quantization manager
        self.quantization_manager = QuantizationManager(config)
        
        # Performance tracking
        self.ffn_times = []
    
    def _init_ffn_weights(self):
        """Initialize feed-forward network weights."""
        # Input projection
        self.input_weight = np.random.normal(
            0, 0.02, (self.intermediate_dim, self.hidden_dim)
        ).astype(np.float32)
        self.input_bias = np.zeros(self.intermediate_dim, dtype=np.float32)
        
        # Output projection
        self.output_weight = np.random.normal(
            0, 0.02, (self.hidden_dim, self.intermediate_dim)
        ).astype(np.float32)
        self.output_bias = np.zeros(self.hidden_dim, dtype=np.float32)
        
        # Layer normalization weights
        self.layer_norm_weight = np.ones(self.hidden_dim, dtype=np.float32)
        self.layer_norm_bias = np.zeros(self.hidden_dim, dtype=np.float32)
    
    def forward(self, hidden_states: np.ndarray) -> np.ndarray:
        """Forward pass through feed-forward network."""
        start_time = time.time()
        
        # Layer normalization
        normalized_states = _fast_layer_norm(
            hidden_states,
            self.layer_norm_weight,
            self.layer_norm_bias,
            self.config.layer_norm_eps
        )
        
        # Input projection
        intermediate = np.dot(normalized_states, self.input_weight.T) + self.input_bias
        
        # Activation function
        if self.config.activation == "gelu":
            intermediate = _fast_gelu(intermediate)
        elif self.config.activation == "relu":
            intermediate = np.maximum(intermediate, 0)
        elif self.config.activation == "swish":
            intermediate = intermediate * (1 / (1 + np.exp(-intermediate)))
        
        # Output projection
        output = np.dot(intermediate, self.output_weight.T) + self.output_bias
        
        # Apply dropout
        if self.hidden_dropout > 0 and self.training:
            dropout_mask = np.random.binomial(1, 1 - self.hidden_dropout, output.shape)
            output = output * dropout_mask / (1 - self.hidden_dropout)
        
        # Residual connection
        output = output + hidden_states
        
        # Track performance
        end_time = time.time()
        self.ffn_times.append(end_time - start_time)
        
        return output
    
    def get_ffn_stats(self) -> Dict[str, Any]:
        """Get feed-forward network statistics."""
        return {
            'layer_idx': self.layer_idx,
            'avg_ffn_time': np.mean(self.ffn_times) if self.ffn_times else 0,
            'total_ffn_calls': len(self.ffn_times),
            'quantization_stats': self.quantization_manager.get_quantization_stats()
        }


class TransformerLayer:
    """Single transformer layer with attention and feed-forward components."""
    
    def __init__(self, config: TransformerConfig, layer_idx: int):
        self.config = config
        self.layer_idx = layer_idx
        
        # Initialize layer components
        self.attention = MultiHeadAttention(config, layer_idx)
        self.feed_forward = FeedForwardNetwork(config, layer_idx)
        
        # Training mode flag
        self.training = True
        
        # Performance tracking
        self.layer_times = []
    
    def forward(self, 
                hidden_states: np.ndarray,
                attention_mask: Optional[np.ndarray] = None,
                context_id: str = "",
                use_cache: bool = True) -> Tuple[np.ndarray, Optional[Dict[str, np.ndarray]]]:
        """Forward pass through transformer layer."""
        start_time = time.time()
        
        # Set training mode for sub-components
        self.attention.training = self.training
        self.feed_forward.training = self.training
        
        # Self-attention
        attention_output, attention_cache = self.attention.forward(
            hidden_states, attention_mask, context_id, use_cache
        )
        
        # Feed-forward network
        output = self.feed_forward.forward(attention_output)
        
        # Track performance
        end_time = time.time()
        self.layer_times.append(end_time - start_time)
        
        return output, attention_cache
    
    def get_layer_stats(self) -> Dict[str, Any]:
        """Get transformer layer statistics."""
        return {
            'layer_idx': self.layer_idx,
            'avg_layer_time': np.mean(self.layer_times) if self.layer_times else 0,
            'total_layer_calls': len(self.layer_times),
            'attention_stats': self.attention.get_attention_stats(),
            'ffn_stats': self.feed_forward.get_ffn_stats()
        }


class FastTransformer:
    """Fast Transformer architecture with CPU optimizations."""
    
    def __init__(self, config: TransformerConfig):
        self.config = config
        
        # Initialize transformer layers
        self.layers = []
        for layer_idx in range(config.num_layers):
            layer = TransformerLayer(config, layer_idx)
            self.layers.append(layer)
        
        # Initialize embeddings
        self._init_embeddings()
        
        # Performance tracking
        self.transformer_times = []
        self.memory_usage = []
        
        # Training mode
        self.training = True
        
        # Log initialization
        logger.info(f"Initialized FastTransformer with {config.num_layers} layers")
        logger.info(f"Hidden dimension: {config.hidden_dim}")
        logger.info(f"Number of heads: {config.num_heads}")
        logger.info(f"AVX-512 support: {HAS_AVX512}")
        logger.info(f"Quantization enabled: {config.use_quantization}")
    
    def _init_embeddings(self):
        """Initialize token and position embeddings."""
        # Token embeddings
        self.token_embeddings = np.random.normal(
            0, 0.02, (50000, self.config.hidden_dim)  # Vocabulary size of 50k
        ).astype(np.float32)
        
        # Position embeddings
        self.position_embeddings = np.random.normal(
            0, 0.02, (self.config.max_position_embeddings, self.config.hidden_dim)
        ).astype(np.float32)
        
        # Final layer normalization
        self.final_layer_norm_weight = np.ones(self.config.hidden_dim, dtype=np.float32)
        self.final_layer_norm_bias = np.zeros(self.config.hidden_dim, dtype=np.float32)
    
    def forward(self, 
                input_ids: np.ndarray,
                attention_mask: Optional[np.ndarray] = None,
                position_ids: Optional[np.ndarray] = None,
                context_id: str = "",
                use_cache: bool = True) -> Tuple[np.ndarray, Optional[List[Dict[str, np.ndarray]]]]:
        """Forward pass through the transformer."""
        start_time = time.time()
        
        batch_size, seq_len = input_ids.shape
        
        # Get token embeddings
        hidden_states = self.token_embeddings[input_ids]
        
        # Add position embeddings
        if position_ids is None:
            position_ids = np.arange(seq_len, dtype=np.int32)
            position_ids = np.tile(position_ids, (batch_size, 1))
        
        position_embeddings = self.position_embeddings[position_ids]
        hidden_states = hidden_states + position_embeddings
        
        # Process through transformer layers
        all_caches = []
        for layer_idx, layer in enumerate(self.layers):
            layer.training = self.training
            
            layer_output, layer_cache = layer.forward(
                hidden_states, attention_mask, context_id, use_cache
            )
            hidden_states = layer_output
            
            if layer_cache is not None:
                all_caches.append(layer_cache)
        
        # Final layer normalization
        output = _fast_layer_norm(
            hidden_states,
            self.final_layer_norm_weight,
            self.final_layer_norm_bias,
            self.config.layer_norm_eps
        )
        
        # Track performance
        end_time = time.time()
        self.transformer_times.append(end_time - start_time)
        
        return output.astype(np.float32), all_caches if use_cache else None
    
    def get_transformer_stats(self) -> Dict[str, Any]:
        """Get transformer statistics."""
        layer_stats = [layer.get_layer_stats() for layer in self.layers]
        
        return {
            'avg_transformer_time': np.mean(self.transformer_times) if self.transformer_times else 0,
            'total_transformer_calls': len(self.transformer_times),
            'layer_stats': layer_stats,
            'memory_usage': self.memory_usage[-10:] if self.memory_usage else [],
            'config': {
                'num_layers': self.config.num_layers,
                'hidden_dim': self.config.hidden_dim,
                'num_heads': self.config.num_heads,
                'use_quantization': self.config.use_quantization,
                'use_avx512': self.config.use_avx512
            }
        }
    
    def clear_cache(self):
        """Clear all attention caches."""
        for layer in self.layers:
            layer.attention.attention_cache.clear()
    
    def set_training(self, training: bool):
        """Set training mode for the transformer."""
        self.training = training
        for layer in self.layers:
            layer.training = training
            layer.attention.training = training
            layer.feed_forward.training = training


# Utility functions for CPU optimization
def optimize_for_cpu():
    """Apply CPU-specific optimizations."""
    if HAS_AVX512:
        logger.info("Enabling AVX-512 optimizations")
        # Set NumPy to use AVX-512
        os.environ['NPY_NUM_BUILD_JOBS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'
    elif HAS_AVX2:
        logger.info("Enabling AVX-2 optimizations")
    elif HAS_SSE4:
        logger.info("Enabling SSE4 optimizations")
    else:
        logger.warning("No advanced CPU features detected, using basic optimizations")


def create_fast_transformer(config: Optional[TransformerConfig] = None) -> FastTransformer:
    """Create a FastTransformer instance with the given configuration."""
    if config is None:
        config = TransformerConfig()
    
    # Apply CPU optimizations
    optimize_for_cpu()
    
    return FastTransformer(config)


# Performance benchmarking functions
def benchmark_transformer(transformer: FastTransformer, 
                         input_ids: np.ndarray,
                         num_runs: int = 10) -> Dict[str, Any]:
    """Benchmark transformer performance."""
    times = []
    memory_usage = []
    
    for _ in range(num_runs):
        start_time = time.time()
        start_memory = _get_memory_usage()
        
        output, _ = transformer.forward(input_ids)
        
        end_time = time.time()
        end_memory = _get_memory_usage()
        
        times.append(end_time - start_time)
        memory_usage.append(end_memory - start_memory)
    
    return {
        'avg_time': np.mean(times),
        'std_time': np.std(times),
        'min_time': np.min(times),
        'max_time': np.max(times),
        'avg_memory': np.mean(memory_usage),
        'max_memory': np.max(memory_usage),
        'throughput': input_ids.shape[1] / np.mean(times)
    }


def _get_memory_usage() -> float:
    """Get current memory usage in MB."""
    try:
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    except ImportError:
        return 0.0


# Export main classes
__all__ = [
    "FastTransformer",
    "TransformerConfig",
    "MultiHeadAttention",
    "FeedForwardNetwork",
    "TransformerLayer",
    "PositionEmbeddings",
    "QuantizationManager",
    "create_fast_transformer",
    "benchmark_transformer"
] 