"""
HERALD Attention Mechanisms
Dual-chunk attention system with intra-chunk and inter-chunk attention

This module implements:
- Dual-chunk attention system
- Intra-chunk and inter-chunk attention
- Attention weight preservation
- CPU-optimized attention computations

Target: ~987 lines of optimized CPU-focused code
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

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class AttentionConfig:
    """Configuration for the attention mechanisms."""
    # Architecture parameters
    hidden_dim: int = 768
    num_heads: int = 12
    head_dim: int = 64
    max_position_embeddings: int = 1000000
    
    # Chunk parameters
    chunk_size: int = 1024
    chunk_overlap: int = 128
    max_chunks: int = 100
    
    # Attention parameters
    attention_dropout: float = 0.1
    layer_norm_eps: float = 1e-12
    
    # Performance settings
    use_avx512: bool = True
    use_quantization: bool = True
    quantize_attention: bool = True
    
    # Memory optimization
    use_memory_mapping: bool = True
    attention_cache_size: int = 8192
    gradient_checkpointing: bool = False
    
    # Advanced features
    use_dual_chunk: bool = True
    use_intra_chunk: bool = True
    use_inter_chunk: bool = True
    use_attention_caching: bool = True
    
    # Optimization flags
    use_fused_operations: bool = True
    use_parallel_attention: bool = True
    use_attention_compression: bool = True


@dataclass
class AttentionCache:
    """Cache for attention computations."""
    intra_chunk_cache: Dict[str, np.ndarray] = field(default_factory=dict)
    inter_chunk_cache: Dict[str, np.ndarray] = field(default_factory=dict)
    attention_weights_cache: Dict[str, np.ndarray] = field(default_factory=dict)
    chunk_cache: Dict[str, np.ndarray] = field(default_factory=dict)
    
    def clear(self):
        """Clear all cached data."""
        self.intra_chunk_cache.clear()
        self.inter_chunk_cache.clear()
        self.attention_weights_cache.clear()
        self.chunk_cache.clear()


class ChunkProcessor:
    """Processes input sequences into chunks for dual-chunk attention."""
    
    def __init__(self, config: AttentionConfig):
        self.config = config
        self.chunk_size = config.chunk_size
        self.chunk_overlap = config.chunk_overlap
        
        # Performance tracking
        self.chunk_times = []
    
    def create_chunks(self, input_tensor: np.ndarray) -> List[np.ndarray]:
        """Create overlapping chunks from input tensor."""
        start_time = time.time()
        
        batch_size, seq_len, hidden_dim = input_tensor.shape
        chunks = []
        
        # Calculate chunk boundaries
        chunk_boundaries = self._calculate_chunk_boundaries(seq_len)
        
        # Create chunks
        for start_idx, end_idx in chunk_boundaries:
            chunk = input_tensor[:, start_idx:end_idx, :]
            chunks.append(chunk)
        
        # Track performance
        end_time = time.time()
        self.chunk_times.append(end_time - start_time)
        
        return chunks
    
    def _calculate_chunk_boundaries(self, seq_len: int) -> List[Tuple[int, int]]:
        """Calculate chunk boundaries with overlap."""
        boundaries = []
        effective_chunk_size = self.chunk_size - self.chunk_overlap
        
        start_idx = 0
        while start_idx < seq_len:
            end_idx = min(start_idx + self.chunk_size, seq_len)
            boundaries.append((start_idx, end_idx))
            start_idx += effective_chunk_size
        
        return boundaries
    
    def merge_chunks(self, chunks: List[np.ndarray], original_seq_len: int) -> np.ndarray:
        """Merge processed chunks back into a single tensor."""
        batch_size, _, hidden_dim = chunks[0].shape
        
        # Initialize output tensor
        output = np.zeros((batch_size, original_seq_len, hidden_dim), dtype=np.float32)
        
        # Calculate chunk boundaries
        chunk_boundaries = self._calculate_chunk_boundaries(original_seq_len)
        
        # Merge chunks with overlap handling
        for i, (start_idx, end_idx) in enumerate(chunk_boundaries):
            chunk = chunks[i]
            chunk_len = end_idx - start_idx
            
            if i == 0:
                # First chunk: use all tokens
                output[:, start_idx:end_idx, :] = chunk[:, :chunk_len, :]
            else:
                # Subsequent chunks: handle overlap
                overlap_start = self.chunk_overlap
                overlap_end = min(self.chunk_overlap + chunk_len, chunk.shape[1])
                
                # Use weighted average for overlap region
                if overlap_start < overlap_end:
                    # Simple average for overlap (can be improved with learned weights)
                    overlap_region = (output[:, start_idx:start_idx + self.chunk_overlap, :] + 
                                   chunk[:, :self.chunk_overlap, :]) / 2
                    output[:, start_idx:start_idx + self.chunk_overlap, :] = overlap_region
                
                # Copy non-overlapping region
                if overlap_end < chunk.shape[1]:
                    output[:, start_idx + self.chunk_overlap:end_idx, :] = chunk[:, overlap_start:overlap_end, :]
        
        return output
    
    def get_chunk_stats(self) -> Dict[str, Any]:
        """Get chunk processing statistics."""
        return {
            'avg_chunk_time': np.mean(self.chunk_times) if self.chunk_times else 0,
            'total_chunk_calls': len(self.chunk_times),
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap
        }


class IntraChunkAttention:
    """Intra-chunk attention mechanism for processing within chunks."""
    
    def __init__(self, config: AttentionConfig):
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.attention_dropout = config.attention_dropout
        
        # Initialize attention weights
        self._init_attention_weights()
        
        # Training mode flag
        self.training = True
        
        # Performance tracking
        self.intra_times = []
    
    def _init_attention_weights(self):
        """Initialize intra-chunk attention weights."""
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
    
    def forward(self, 
                chunk: np.ndarray,
                attention_mask: Optional[np.ndarray] = None,
                context_id: str = "",
                use_cache: bool = True) -> Tuple[np.ndarray, Optional[Dict[str, np.ndarray]]]:
        """Forward pass through intra-chunk attention."""
        start_time = time.time()
        
        batch_size, seq_len, hidden_dim = chunk.shape
        
        # Layer normalization
        normalized_chunk = self._layer_norm(chunk)
        
        # Project to query, key, value
        query = self._project_query(normalized_chunk)
        key = self._project_key(normalized_chunk)
        value = self._project_value(normalized_chunk)
        
        # Reshape for multi-head attention
        query = query.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        key = key.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        value = value.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Compute attention scores
        attention_scores = self._compute_attention_scores(query, key, attention_mask)
        
        # Apply softmax
        attention_weights = self._apply_softmax(attention_scores)
        
        # Apply dropout
        if self.attention_dropout > 0 and self.training:
            attention_weights = self._apply_dropout(attention_weights)
        
        # Compute attention output
        attention_output = self._compute_attention_output(attention_weights, value)
        
        # Reshape back
        attention_output = attention_output.reshape(batch_size, seq_len, hidden_dim)
        
        # Output projection
        output = np.dot(attention_output, self.output_weight.T)
        
        # Residual connection
        output = output + chunk
        
        # Cache if requested
        cache = None
        if use_cache and self.config.use_attention_caching:
            cache = {
                'query': query,
                'key': key,
                'value': value,
                'attention_weights': attention_weights
            }
        
        # Track performance
        end_time = time.time()
        self.intra_times.append(end_time - start_time)
        
        return output, cache
    
    def _layer_norm(self, input_tensor: np.ndarray) -> np.ndarray:
        """Apply layer normalization."""
        mean = np.mean(input_tensor, axis=-1, keepdims=True)
        var = np.var(input_tensor, axis=-1, keepdims=True)
        std = np.sqrt(var + self.config.layer_norm_eps)
        normalized = (input_tensor - mean) / std
        return self.layer_norm_weight * normalized + self.layer_norm_bias
    
    def _project_query(self, input_tensor: np.ndarray) -> np.ndarray:
        """Project input to query space."""
        return np.dot(input_tensor, self.query_weight.T)
    
    def _project_key(self, input_tensor: np.ndarray) -> np.ndarray:
        """Project input to key space."""
        return np.dot(input_tensor, self.key_weight.T)
    
    def _project_value(self, input_tensor: np.ndarray) -> np.ndarray:
        """Project input to value space."""
        return np.dot(input_tensor, self.value_weight.T)
    
    def _compute_attention_scores(self, 
                                 query: np.ndarray, 
                                 key: np.ndarray, 
                                 mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute attention scores."""
        # Compute attention scores: Q * K^T / sqrt(d_k)
        scores = np.einsum('bhid,bhjd->bhij', query, key)
        scores = scores / np.sqrt(self.head_dim)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores + mask
        
        return scores
    
    def _apply_softmax(self, scores: np.ndarray) -> np.ndarray:
        """Apply softmax to attention scores."""
        max_scores = np.max(scores, axis=-1, keepdims=True)
        exp_scores = np.exp(scores - max_scores)
        sum_exp = np.sum(exp_scores, axis=-1, keepdims=True)
        return exp_scores / sum_exp
    
    def _apply_dropout(self, attention_weights: np.ndarray) -> np.ndarray:
        """Apply dropout to attention weights."""
        dropout_mask = np.random.binomial(1, 1 - self.attention_dropout, attention_weights.shape)
        return attention_weights * dropout_mask / (1 - self.attention_dropout)
    
    def _compute_attention_output(self, attention_weights: np.ndarray, value: np.ndarray) -> np.ndarray:
        """Compute attention output."""
        return np.einsum('bhij,bhjd->bhid', attention_weights, value)
    
    def get_intra_stats(self) -> Dict[str, Any]:
        """Get intra-chunk attention statistics."""
        return {
            'avg_intra_time': np.mean(self.intra_times) if self.intra_times else 0,
            'total_intra_calls': len(self.intra_times),
            'hidden_dim': self.hidden_dim,
            'num_heads': self.num_heads
        }


class InterChunkAttention:
    """Inter-chunk attention mechanism for processing between chunks."""
    
    def __init__(self, config: AttentionConfig):
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.attention_dropout = config.attention_dropout
        
        # Initialize attention weights
        self._init_attention_weights()
        
        # Performance tracking
        self.inter_times = []
        
        # Training mode
        self.training = True
    
    def _init_attention_weights(self):
        """Initialize inter-chunk attention weights."""
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
    
    def forward(self, 
                chunks: List[np.ndarray],
                attention_mask: Optional[np.ndarray] = None,
                context_id: str = "",
                use_cache: bool = True) -> Tuple[List[np.ndarray], Optional[Dict[str, np.ndarray]]]:
        """Forward pass through inter-chunk attention."""
        start_time = time.time()
        
        if len(chunks) <= 1:
            # No inter-chunk attention needed for single chunk
            return chunks, None
        
        # Create chunk representations (e.g., mean pooling)
        chunk_reprs = self._create_chunk_representations(chunks)
        
        # Apply attention between chunk representations
        attended_reprs = self._apply_chunk_attention(chunk_reprs, attention_mask)
        
        # Distribute attention information back to chunks
        updated_chunks = self._distribute_attention_to_chunks(chunks, attended_reprs)
        
        # Cache if requested
        cache = None
        if use_cache and self.config.use_attention_caching:
            cache = {
                'chunk_reprs': chunk_reprs,
                'attended_reprs': attended_reprs
            }
        
        # Track performance
        end_time = time.time()
        self.inter_times.append(end_time - start_time)
        
        return updated_chunks, cache
    
    def _create_chunk_representations(self, chunks: List[np.ndarray]) -> np.ndarray:
        """Create representations for each chunk."""
        # Use mean pooling as chunk representation
        chunk_reprs = []
        for chunk in chunks:
            # Mean pooling over sequence dimension
            repr_ = np.mean(chunk, axis=1)  # (batch_size, hidden_dim)
            chunk_reprs.append(repr_)
        
        return np.stack(chunk_reprs, axis=1)  # (batch_size, num_chunks, hidden_dim)
    
    def _apply_chunk_attention(self, 
                              chunk_reprs: np.ndarray, 
                              attention_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Apply attention between chunk representations."""
        batch_size, num_chunks, hidden_dim = chunk_reprs.shape
        
        # Layer normalization
        normalized_reprs = self._layer_norm(chunk_reprs)
        
        # Project to query, key, value
        query = self._project_query(normalized_reprs)
        key = self._project_key(normalized_reprs)
        value = self._project_value(normalized_reprs)
        
        # Reshape for multi-head attention
        query = query.reshape(batch_size, num_chunks, self.num_heads, self.head_dim)
        key = key.reshape(batch_size, num_chunks, self.num_heads, self.head_dim)
        value = value.reshape(batch_size, num_chunks, self.num_heads, self.head_dim)
        
        # Compute attention scores
        attention_scores = self._compute_attention_scores(query, key, attention_mask)
        
        # Apply softmax
        attention_weights = self._apply_softmax(attention_scores)
        
        # Apply dropout
        if self.attention_dropout > 0 and self.training:
            attention_weights = self._apply_dropout(attention_weights)
        
        # Compute attention output
        attention_output = self._compute_attention_output(attention_weights, value)
        
        # Reshape back
        attention_output = attention_output.reshape(batch_size, num_chunks, hidden_dim)
        
        # Output projection
        output = np.dot(attention_output, self.output_weight.T)
        
        # Residual connection
        output = output + chunk_reprs
        
        return output
    
    def _distribute_attention_to_chunks(self, 
                                      chunks: List[np.ndarray], 
                                      attended_reprs: np.ndarray) -> List[np.ndarray]:
        """Distribute attention information back to individual chunks."""
        updated_chunks = []
        
        for i, chunk in enumerate(chunks):
            # Get corresponding attended representation
            attended_repr = attended_reprs[:, i, :]  # (batch_size, hidden_dim)
            
            # Add attended representation to chunk (simple addition)
            # This can be improved with more sophisticated fusion methods
            updated_chunk = chunk + attended_repr[:, None, :]  # Broadcast to sequence dimension
            updated_chunks.append(updated_chunk)
        
        return updated_chunks
    
    def _layer_norm(self, input_tensor: np.ndarray) -> np.ndarray:
        """Apply layer normalization."""
        mean = np.mean(input_tensor, axis=-1, keepdims=True)
        var = np.var(input_tensor, axis=-1, keepdims=True)
        std = np.sqrt(var + self.config.layer_norm_eps)
        normalized = (input_tensor - mean) / std
        return self.layer_norm_weight * normalized + self.layer_norm_bias
    
    def _project_query(self, input_tensor: np.ndarray) -> np.ndarray:
        """Project input to query space."""
        return np.dot(input_tensor, self.query_weight.T)
    
    def _project_key(self, input_tensor: np.ndarray) -> np.ndarray:
        """Project input to key space."""
        return np.dot(input_tensor, self.key_weight.T)
    
    def _project_value(self, input_tensor: np.ndarray) -> np.ndarray:
        """Project input to value space."""
        return np.dot(input_tensor, self.value_weight.T)
    
    def _compute_attention_scores(self, 
                                 query: np.ndarray, 
                                 key: np.ndarray, 
                                 mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute attention scores."""
        # Compute attention scores: Q * K^T / sqrt(d_k)
        scores = np.einsum('bhid,bhjd->bhij', query, key)
        scores = scores / np.sqrt(self.head_dim)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores + mask
        
        return scores
    
    def _apply_softmax(self, scores: np.ndarray) -> np.ndarray:
        """Apply softmax to attention scores."""
        max_scores = np.max(scores, axis=-1, keepdims=True)
        exp_scores = np.exp(scores - max_scores)
        sum_exp = np.sum(exp_scores, axis=-1, keepdims=True)
        return exp_scores / sum_exp
    
    def _apply_dropout(self, attention_weights: np.ndarray) -> np.ndarray:
        """Apply dropout to attention weights."""
        dropout_mask = np.random.binomial(1, 1 - self.attention_dropout, attention_weights.shape)
        return attention_weights * dropout_mask / (1 - self.attention_dropout)
    
    def _compute_attention_output(self, attention_weights: np.ndarray, value: np.ndarray) -> np.ndarray:
        """Compute attention output."""
        return np.einsum('bhij,bhjd->bhid', attention_weights, value)
    
    def get_inter_stats(self) -> Dict[str, Any]:
        """Get inter-chunk attention statistics."""
        return {
            'avg_inter_time': np.mean(self.inter_times) if self.inter_times else 0,
            'total_inter_calls': len(self.inter_times),
            'hidden_dim': self.hidden_dim,
            'num_heads': self.num_heads
        }


class DualChunkAttention:
    """Dual-chunk attention system with intra-chunk and inter-chunk attention."""
    
    def __init__(self, config: AttentionConfig):
        self.config = config
        
        # Initialize components
        self.chunk_processor = ChunkProcessor(config)
        self.intra_attention = IntraChunkAttention(config)
        self.inter_attention = InterChunkAttention(config)
        
        # Attention cache
        self.attention_cache = AttentionCache()
        
        # Performance tracking
        self.dual_times = []
        self.memory_usage = []
        
        # Training mode
        self.training = True
        
        # Log initialization
        logger.info(f"Initialized DualChunkAttention")
        logger.info(f"Chunk size: {config.chunk_size}")
        logger.info(f"Chunk overlap: {config.chunk_overlap}")
        logger.info(f"Hidden dimension: {config.hidden_dim}")
        logger.info(f"Number of heads: {config.num_heads}")
    
    def forward(self, 
                input_tensor: np.ndarray,
                attention_mask: Optional[np.ndarray] = None,
                context_id: str = "",
                use_cache: bool = True) -> Tuple[np.ndarray, Optional[Dict[str, np.ndarray]]]:
        """Forward pass through dual-chunk attention."""
        start_time = time.time()
        
        batch_size, seq_len, hidden_dim = input_tensor.shape
        
        # Create chunks
        chunks = self.chunk_processor.create_chunks(input_tensor)
        
        # Apply intra-chunk attention to each chunk
        intra_processed_chunks = []
        intra_caches = []
        
        for i, chunk in enumerate(chunks):
            chunk_context_id = f"{context_id}_chunk_{i}"
            
            intra_output, intra_cache = self.intra_attention.forward(
                chunk, attention_mask, chunk_context_id, use_cache
            )
            intra_processed_chunks.append(intra_output)
            
            if intra_cache is not None:
                intra_caches.append(intra_cache)
        
        # Apply inter-chunk attention
        inter_processed_chunks, inter_cache = self.inter_attention.forward(
            intra_processed_chunks, attention_mask, context_id, use_cache
        )
        
        # Merge chunks back
        output = self.chunk_processor.merge_chunks(inter_processed_chunks, seq_len)
        
        # Combine caches
        cache = None
        if use_cache:
            cache = {
                'intra_caches': intra_caches,
                'inter_cache': inter_cache,
                'chunks': chunks
            }
        
        # Track performance
        end_time = time.time()
        self.dual_times.append(end_time - start_time)
        
        return output, cache
    
    def get_dual_stats(self) -> Dict[str, Any]:
        """Get dual-chunk attention statistics."""
        return {
            'avg_dual_time': np.mean(self.dual_times) if self.dual_times else 0,
            'total_dual_calls': len(self.dual_times),
            'chunk_stats': self.chunk_processor.get_chunk_stats(),
            'intra_stats': self.intra_attention.get_intra_stats(),
            'inter_stats': self.inter_attention.get_inter_stats(),
            'memory_usage': self.memory_usage[-10:] if self.memory_usage else [],
            'config': {
                'chunk_size': self.config.chunk_size,
                'chunk_overlap': self.config.chunk_overlap,
                'hidden_dim': self.config.hidden_dim,
                'num_heads': self.config.num_heads,
                'use_dual_chunk': self.config.use_dual_chunk
            }
        }
    
    def clear_cache(self):
        """Clear all attention caches."""
        self.attention_cache.clear()
    
    def set_training(self, training: bool):
        """Set training mode for the attention system."""
        self.training = training
        self.intra_attention.training = training
        self.inter_attention.training = training


# Utility functions for CPU optimization
def optimize_attention_for_cpu():
    """Apply CPU-specific optimizations for attention."""
    # Set NumPy to use optimized BLAS
    os.environ['NPY_NUM_BUILD_JOBS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    
    # Enable parallel processing
    numba.set_num_threads(os.cpu_count())


def create_dual_chunk_attention(config: Optional[AttentionConfig] = None) -> DualChunkAttention:
    """Create a DualChunkAttention instance with the given configuration."""
    if config is None:
        config = AttentionConfig()
    
    # Apply CPU optimizations
    optimize_attention_for_cpu()
    
    return DualChunkAttention(config)


# Performance benchmarking functions
def benchmark_attention(attention_layer: DualChunkAttention, 
                       input_tensor: np.ndarray,
                       num_runs: int = 10) -> Dict[str, Any]:
    """Benchmark attention layer performance."""
    times = []
    memory_usage = []
    
    for _ in range(num_runs):
        start_time = time.time()
        start_memory = _get_memory_usage()
        
        output, _ = attention_layer.forward(input_tensor)
        
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
        'throughput': input_tensor.shape[1] / np.mean(times)
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
    "DualChunkAttention",
    "AttentionConfig",
    "ChunkProcessor",
    "IntraChunkAttention",
    "InterChunkAttention",
    "AttentionCache",
    "create_dual_chunk_attention",
    "benchmark_attention"
] 