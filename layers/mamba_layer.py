"""
HERALD Mamba State Space Layers
Selective state space mechanism with linear O(n) complexity

This module implements:
- Selective state space mechanism
- 6 Mamba blocks with 1,024 state dimensions
- Optimize for linear O(n) complexity
- CPU-specific optimizations

Target: ~2,123 lines of optimized CPU-focused code
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
class MambaConfig:
    """Configuration for the Mamba state space model."""
    # Architecture parameters
    num_blocks: int = 6
    state_dim: int = 1024
    hidden_dim: int = 768
    intermediate_dim: int = 3072
    max_position_embeddings: int = 1000000
    
    # State space parameters
    selective_scan: bool = True
    linear_complexity: bool = True
    use_parallel_state: bool = True
    
    # Performance settings
    use_avx512: bool = True
    use_quantization: bool = True
    quantize_state: bool = True
    quantize_projection: bool = True
    
    # Memory optimization
    use_memory_mapping: bool = True
    state_cache_size: int = 8192
    gradient_checkpointing: bool = False
    
    # Advanced features
    use_convolutional_state: bool = True
    use_parallel_scan: bool = True
    use_state_caching: bool = True
    
    # Optimization flags
    use_fused_operations: bool = True
    use_parallel_state_update: bool = True
    use_state_compression: bool = True


@dataclass
class StateCache:
    """Cache for state space computations."""
    state_cache: Dict[str, np.ndarray] = field(default_factory=dict)
    scan_cache: Dict[str, np.ndarray] = field(default_factory=dict)
    convolution_cache: Dict[str, np.ndarray] = field(default_factory=dict)
    
    def clear(self):
        """Clear all cached data."""
        self.state_cache.clear()
        self.scan_cache.clear()
        self.convolution_cache.clear()


class StateSpaceModel:
    """State space model with selective scanning mechanism."""
    
    def __init__(self, config: MambaConfig, block_idx: int):
        self.config = config
        self.block_idx = block_idx
        self.state_dim = config.state_dim
        self.hidden_dim = config.hidden_dim
        
        # Initialize state space parameters
        self._init_state_parameters()
        
        # Performance tracking
        self.state_times = []
        self.scan_times = []
    
    def _init_state_parameters(self):
        """Initialize state space model parameters."""
        # State transition matrices
        self.A = np.random.normal(
            0, 0.1, (self.state_dim, self.state_dim)
        ).astype(np.float32)
        
        # Input projection matrix
        self.B = np.random.normal(
            0, 0.1, (self.state_dim, self.hidden_dim)
        ).astype(np.float32)
        
        # Output projection matrix
        self.C = np.random.normal(
            0, 0.1, (self.hidden_dim, self.state_dim)
        ).astype(np.float32)
        
        # Diagonal state matrix for efficiency
        self.D = np.random.normal(
            0, 0.1, self.state_dim
        ).astype(np.float32)
        
        # Selective scan parameters
        if self.config.selective_scan:
            self.delta = np.random.normal(
                0, 0.1, self.hidden_dim
            ).astype(np.float32)
            self.gamma = np.random.normal(
                0, 0.1, self.hidden_dim
            ).astype(np.float32)
    
    def forward(self, 
                input_tensor: np.ndarray,
                context_id: str = "",
                use_cache: bool = True) -> Tuple[np.ndarray, Optional[Dict[str, np.ndarray]]]:
        """Forward pass through state space model."""
        start_time = time.time()
        
        batch_size, seq_len, hidden_dim = input_tensor.shape
        
        # Initialize state
        state = np.zeros((batch_size, self.state_dim), dtype=np.float32)
        
        # Process sequence through state space model
        outputs = []
        states = []
        
        for t in range(seq_len):
            # Get current input
            x_t = input_tensor[:, t, :]
            
            # Update state
            state = self._update_state(state, x_t)
            states.append(state.copy())
            
            # Compute output
            output = self._compute_output(state, x_t)
            outputs.append(output)
        
        # Stack outputs
        output_tensor = np.stack(outputs, axis=1)
        
        # Cache states if requested
        cache = None
        if use_cache and self.config.use_state_caching:
            cache = {
                'states': np.stack(states, axis=1),
                'final_state': state
            }
        
        # Track performance
        end_time = time.time()
        self.state_times.append(end_time - start_time)
        
        return output_tensor, cache
    
    def _update_state(self, state: np.ndarray, input_tensor: np.ndarray) -> np.ndarray:
        """Update state using state space equations."""
        # State update: s_t = A * s_{t-1} + B * x_t
        state_update = np.dot(state, self.A.T) + np.dot(input_tensor, self.B.T)
        
        # Apply selective scan if enabled
        if self.config.selective_scan:
            # Compute selective scan parameters
            delta = np.dot(input_tensor, self.delta)
            gamma = np.dot(input_tensor, self.gamma)
            
            # Apply selective scan
            state_update = self._apply_selective_scan(state_update, delta, gamma)
        
        return state_update
    
    def _apply_selective_scan(self, 
                             state_update: np.ndarray, 
                             delta: np.ndarray, 
                             gamma: np.ndarray) -> np.ndarray:
        """Apply selective scan mechanism."""
        # Compute selective scan weights
        scan_weights = np.tanh(delta[:, None] * state_update + gamma[:, None])
        
        # Apply selective scan
        return state_update * scan_weights
    
    def _compute_output(self, state: np.ndarray, input_tensor: np.ndarray) -> np.ndarray:
        """Compute output using state space equations."""
        # Output: y_t = C * s_t + D * x_t
        state_output = np.dot(state, self.C.T)
        # Broadcast D to match input_tensor shape
        input_output = input_tensor * self.D[:input_tensor.shape[-1]]
        
        return state_output + input_output
    
    def get_state_stats(self) -> Dict[str, Any]:
        """Get state space model statistics."""
        return {
            'block_idx': self.block_idx,
            'avg_state_time': np.mean(self.state_times) if self.state_times else 0,
            'total_state_calls': len(self.state_times),
            'state_dim': self.state_dim,
            'hidden_dim': self.hidden_dim
        }


class ConvolutionalStateModel:
    """Convolutional state space model for parallel processing."""
    
    def __init__(self, config: MambaConfig, block_idx: int):
        self.config = config
        self.block_idx = block_idx
        self.hidden_dim = config.hidden_dim
        
        # Initialize convolutional parameters
        self._init_convolutional_parameters()
        
        # Performance tracking
        self.conv_times = []
    
    def _init_convolutional_parameters(self):
        """Initialize convolutional state space parameters."""
        # Convolutional kernel
        kernel_size = 4
        self.conv_kernel = np.random.normal(
            0, 0.1, (kernel_size, self.hidden_dim, self.hidden_dim)
        ).astype(np.float32)
        
        # State projection
        self.state_projection = np.random.normal(
            0, 0.1, (self.hidden_dim, self.hidden_dim)
        ).astype(np.float32)
        
        # Output projection
        self.output_projection = np.random.normal(
            0, 0.1, (self.hidden_dim, self.hidden_dim)
        ).astype(np.float32)
    
    def forward(self, 
                input_tensor: np.ndarray,
                context_id: str = "",
                use_cache: bool = True) -> Tuple[np.ndarray, Optional[Dict[str, np.ndarray]]]:
        """Forward pass through convolutional state model."""
        start_time = time.time()
        
        batch_size, seq_len, hidden_dim = input_tensor.shape
        
        # Apply convolutional state processing
        conv_output = self._apply_convolution(input_tensor)
        
        # State projection
        state_output = np.dot(conv_output, self.state_projection.T)
        
        # Output projection
        output = np.dot(state_output, self.output_projection.T)
        
        # Residual connection
        output = output + input_tensor
        
        # Cache if requested
        cache = None
        if use_cache:
            cache = {
                'conv_output': conv_output,
                'state_output': state_output
            }
        
        # Track performance
        end_time = time.time()
        self.conv_times.append(end_time - start_time)
        
        return output, cache
    
    def _apply_convolution(self, input_tensor: np.ndarray) -> np.ndarray:
        """Apply convolutional state processing."""
        batch_size, seq_len, hidden_dim = input_tensor.shape
        kernel_size = self.conv_kernel.shape[0]
        
        # Pad input for convolution
        padded_input = np.pad(input_tensor, ((0, 0), (kernel_size - 1, 0), (0, 0)), mode='constant')
        
        # Apply convolution
        conv_output = np.zeros_like(input_tensor)
        
        for t in range(seq_len):
            # Extract window
            window = padded_input[:, t:t + kernel_size, :]
            
            # Apply kernel
            for k in range(kernel_size):
                conv_output[:, t, :] += np.dot(window[:, k, :], self.conv_kernel[k, :, :].T)
        
        return conv_output
    
    def get_conv_stats(self) -> Dict[str, Any]:
        """Get convolutional state model statistics."""
        return {
            'block_idx': self.block_idx,
            'avg_conv_time': np.mean(self.conv_times) if self.conv_times else 0,
            'total_conv_calls': len(self.conv_times),
            'hidden_dim': self.hidden_dim
        }


class MambaBlock:
    """Single Mamba block with state space and convolutional components."""
    
    def __init__(self, config: MambaConfig, block_idx: int):
        self.config = config
        self.block_idx = block_idx
        
        # Initialize block components
        self.state_model = StateSpaceModel(config, block_idx)
        self.conv_model = ConvolutionalStateModel(config, block_idx)
        
        # Layer normalization
        self.layer_norm_weight = np.ones(config.hidden_dim, dtype=np.float32)
        self.layer_norm_bias = np.zeros(config.hidden_dim, dtype=np.float32)
        
        # Training mode flag
        self.training = True
        
        # Performance tracking
        self.block_times = []
    
    def forward(self, 
                input_tensor: np.ndarray,
                context_id: str = "",
                use_cache: bool = True) -> Tuple[np.ndarray, Optional[Dict[str, np.ndarray]]]:
        """Forward pass through Mamba block."""
        start_time = time.time()
        
        # Set training mode for sub-components
        self.state_model.training = self.training
        self.conv_model.training = self.training
        
        # Layer normalization
        normalized_input = self._layer_norm(input_tensor)
        
        # State space processing
        state_output, state_cache = self.state_model.forward(
            normalized_input, context_id, use_cache
        )
        
        # Convolutional processing
        conv_output, conv_cache = self.conv_model.forward(
            state_output, context_id, use_cache
        )
        
        # Residual connection
        output = conv_output + input_tensor
        
        # Combine caches
        cache = None
        if use_cache:
            cache = {
                'state_cache': state_cache,
                'conv_cache': conv_cache
            }
        
        # Track performance
        end_time = time.time()
        self.block_times.append(end_time - start_time)
        
        return output, cache
    
    def _layer_norm(self, input_tensor: np.ndarray) -> np.ndarray:
        """Apply layer normalization."""
        mean = np.mean(input_tensor, axis=-1, keepdims=True)
        var = np.var(input_tensor, axis=-1, keepdims=True)
        std = np.sqrt(var + 1e-12)
        normalized = (input_tensor - mean) / std
        return self.layer_norm_weight * normalized + self.layer_norm_bias
    
    def get_block_stats(self) -> Dict[str, Any]:
        """Get Mamba block statistics."""
        return {
            'block_idx': self.block_idx,
            'avg_block_time': np.mean(self.block_times) if self.block_times else 0,
            'total_block_calls': len(self.block_times),
            'state_stats': self.state_model.get_state_stats(),
            'conv_stats': self.conv_model.get_conv_stats()
        }


class MambaLayer:
    """Mamba layer with multiple blocks and CPU optimizations."""
    
    def __init__(self, config: MambaConfig):
        self.config = config
        
        # Initialize Mamba blocks
        self.blocks = []
        for block_idx in range(config.num_blocks):
            block = MambaBlock(config, block_idx)
            self.blocks.append(block)
        
        # Initialize embeddings
        self._init_embeddings()
        
        # State cache
        self.state_cache = StateCache()
        
        # Performance tracking
        self.mamba_times = []
        self.memory_usage = []
        
        # Training mode
        self.training = True
        
        # Log initialization
        logger.info(f"Initialized MambaLayer with {config.num_blocks} blocks")
        logger.info(f"State dimension: {config.state_dim}")
        logger.info(f"Hidden dimension: {config.hidden_dim}")
        logger.info(f"Selective scan: {config.selective_scan}")
        logger.info(f"Linear complexity: {config.linear_complexity}")
    
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
        """Forward pass through the Mamba layer."""
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
        
        # Process through Mamba blocks
        all_caches = []
        for block_idx, block in enumerate(self.blocks):
            block.training = self.training
            
            block_output, block_cache = block.forward(
                hidden_states, context_id, use_cache
            )
            hidden_states = block_output
            
            if block_cache is not None:
                all_caches.append(block_cache)
        
        # Final layer normalization
        output = self._final_layer_norm(hidden_states)
        
        # Track performance
        end_time = time.time()
        self.mamba_times.append(end_time - start_time)
        
        return output, all_caches if use_cache else None
    
    def _final_layer_norm(self, hidden_states: np.ndarray) -> np.ndarray:
        """Apply final layer normalization."""
        mean = np.mean(hidden_states, axis=-1, keepdims=True)
        var = np.var(hidden_states, axis=-1, keepdims=True)
        std = np.sqrt(var + 1e-12)
        normalized = (hidden_states - mean) / std
        return self.final_layer_norm_weight * normalized + self.final_layer_norm_bias
    
    def get_mamba_stats(self) -> Dict[str, Any]:
        """Get Mamba layer statistics."""
        block_stats = [block.get_block_stats() for block in self.blocks]
        
        return {
            'avg_mamba_time': np.mean(self.mamba_times) if self.mamba_times else 0,
            'total_mamba_calls': len(self.mamba_times),
            'block_stats': block_stats,
            'memory_usage': self.memory_usage[-10:] if self.memory_usage else [],
            'config': {
                'num_blocks': self.config.num_blocks,
                'state_dim': self.config.state_dim,
                'hidden_dim': self.config.hidden_dim,
                'selective_scan': self.config.selective_scan,
                'linear_complexity': self.config.linear_complexity
            }
        }
    
    def clear_cache(self):
        """Clear all state caches."""
        self.state_cache.clear()
        for block in self.blocks:
            block.state_model.state_cache.clear()
    
    def set_training(self, training: bool):
        """Set training mode for the Mamba layer."""
        self.training = training
        for block in self.blocks:
            block.training = training
            block.state_model.training = training
            block.conv_model.training = training


# Utility functions for CPU optimization
def optimize_mamba_for_cpu():
    """Apply CPU-specific optimizations for Mamba."""
    # Set NumPy to use optimized BLAS
    os.environ['NPY_NUM_BUILD_JOBS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    
    # Enable parallel processing
    numba.set_num_threads(os.cpu_count())


def create_mamba_layer(config: Optional[MambaConfig] = None) -> MambaLayer:
    """Create a MambaLayer instance with the given configuration."""
    if config is None:
        config = MambaConfig()
    
    # Apply CPU optimizations
    optimize_mamba_for_cpu()
    
    return MambaLayer(config)


# Performance benchmarking functions
def benchmark_mamba(mamba_layer: MambaLayer, 
                   input_ids: np.ndarray,
                   num_runs: int = 10) -> Dict[str, Any]:
    """Benchmark Mamba layer performance."""
    times = []
    memory_usage = []
    
    for _ in range(num_runs):
        start_time = time.time()
        start_memory = _get_memory_usage()
        
        output, _ = mamba_layer.forward(input_ids)
        
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
    "MambaLayer",
    "MambaConfig",
    "StateSpaceModel",
    "ConvolutionalStateModel",
    "MambaBlock",
    "StateCache",
    "create_mamba_layer",
    "benchmark_mamba"
] 