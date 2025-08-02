"""
Unit tests for HERALD Transformer Layers
Tests the FastTransformer, MambaLayer, DualChunkAttention, and QuantizationLayer
"""

import pytest
import numpy as np
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from layers.fast_transformer import FastTransformer, TransformerConfig, create_fast_transformer
from layers.mamba_layer import MambaLayer, MambaConfig, create_mamba_layer
from layers.attention import DualChunkAttention, AttentionConfig, create_dual_chunk_attention
from layers.quantization import QuantizationLayer, QuantizationConfig, create_quantization_layer


class TestFastTransformer:
    """Test cases for FastTransformer."""
    
    def test_transformer_initialization(self):
        """Test that FastTransformer initializes correctly."""
        config = TransformerConfig(
            num_layers=2,  # Use fewer layers for testing
            hidden_dim=128,
            num_heads=4,
            head_dim=32,
            activation="gelu"
        )
        transformer = create_fast_transformer(config)
        
        assert transformer is not None
        assert len(transformer.layers) == 2
        assert transformer.config.hidden_dim == 128
        assert transformer.config.num_heads == 4
    
    def test_transformer_forward_pass(self):
        """Test that FastTransformer forward pass works."""
        config = TransformerConfig(
            num_layers=2,
            hidden_dim=128,
            num_heads=4,
            head_dim=32,
            activation="gelu"
        )
        transformer = create_fast_transformer(config)
        
        # Create test input
        batch_size = 2
        seq_len = 10
        input_ids = np.random.randint(0, 1000, (batch_size, seq_len))
        
        # Forward pass
        output, cache = transformer.forward(input_ids)
        
        assert output is not None
        assert output.shape == (batch_size, seq_len, config.hidden_dim)
        assert output.dtype == np.float32
    
    def test_transformer_attention_mechanism(self):
        """Test that attention mechanism works correctly."""
        config = TransformerConfig(
            num_layers=1,
            hidden_dim=128,
            num_heads=4,
            head_dim=32,
            activation="gelu"
        )
        transformer = create_fast_transformer(config)
        
        # Test input
        input_ids = np.array([[1, 2, 3, 4, 5]])
        
        # Forward pass
        output, cache = transformer.forward(input_ids)
        
        # Check that output has correct shape
        assert output.shape == (1, 5, 128)
        
        # Check that output is not all zeros
        assert not np.allclose(output, 0)
    
    def test_transformer_performance_stats(self):
        """Test that performance statistics are collected."""
        config = TransformerConfig(num_layers=1, hidden_dim=128, num_heads=4, head_dim=32, activation="gelu")
        transformer = create_fast_transformer(config)
        
        # Run a few forward passes
        input_ids = np.random.randint(0, 100, (1, 5))
        for _ in range(3):
            transformer.forward(input_ids)
        
        # Get stats
        stats = transformer.get_transformer_stats()
        
        assert 'avg_transformer_time' in stats
        assert 'total_transformer_calls' in stats
        assert stats['total_transformer_calls'] == 3


class TestMambaLayer:
    """Test cases for MambaLayer."""
    
    def test_mamba_initialization(self):
        """Test that MambaLayer initializes correctly."""
        config = MambaConfig(
            num_blocks=2,
            state_dim=256,
            hidden_dim=128
        )
        mamba = create_mamba_layer(config)
        
        assert mamba is not None
        assert len(mamba.blocks) == 2
        assert mamba.config.state_dim == 256
        assert mamba.config.hidden_dim == 128
    
    def test_mamba_forward_pass(self):
        """Test that MambaLayer forward pass works."""
        config = MambaConfig(
            num_blocks=2,
            state_dim=256,
            hidden_dim=128
        )
        mamba = create_mamba_layer(config)
        
        # Create test input
        batch_size = 2
        seq_len = 10
        input_ids = np.random.randint(0, 1000, (batch_size, seq_len))
        
        # Forward pass
        output, cache = mamba.forward(input_ids)
        
        assert output is not None
        assert output.shape == (batch_size, seq_len, config.hidden_dim)
        assert output.dtype == np.float32
    
    def test_mamba_state_space_model(self):
        """Test that state space model works correctly."""
        config = MambaConfig(
            num_blocks=1,
            state_dim=128,
            hidden_dim=64
        )
        mamba = create_mamba_layer(config)
        
        # Test input
        input_ids = np.array([[1, 2, 3, 4, 5]])
        
        # Forward pass
        output, cache = mamba.forward(input_ids)
        
        # Check that output has correct shape
        assert output.shape == (1, 5, 64)
        
        # Check that output is not all zeros
        assert not np.allclose(output, 0)


class TestDualChunkAttention:
    """Test cases for DualChunkAttention."""
    
    def test_attention_initialization(self):
        """Test that DualChunkAttention initializes correctly."""
        config = AttentionConfig(
            hidden_dim=128,
            num_heads=4,
            head_dim=32,
            chunk_size=4,
            chunk_overlap=1
        )
        attention = create_dual_chunk_attention(config)
        
        assert attention is not None
        assert attention.config.hidden_dim == 128
        assert attention.config.num_heads == 4
    
    def test_attention_forward_pass(self):
        """Test that DualChunkAttention forward pass works."""
        config = AttentionConfig(
            hidden_dim=128,
            num_heads=4,
            head_dim=32,
            chunk_size=4,
            chunk_overlap=1
        )
        attention = create_dual_chunk_attention(config)
        
        # Create test input
        batch_size = 2
        seq_len = 8
        hidden_dim = 128
        input_tensor = np.random.normal(0, 1, (batch_size, seq_len, hidden_dim))
        
        # Forward pass
        output, cache = attention.forward(input_tensor)
        
        assert output is not None
        assert output.shape == (batch_size, seq_len, hidden_dim)
        assert output.dtype == np.float32
    
    def test_chunk_processing(self):
        """Test that chunk processing works correctly."""
        config = AttentionConfig(
            hidden_dim=64,
            num_heads=2,
            head_dim=32,
            chunk_size=3,
            chunk_overlap=1
        )
        attention = create_dual_chunk_attention(config)
        
        # Test input
        input_tensor = np.random.normal(0, 1, (1, 6, 64))
        
        # Forward pass
        output, cache = attention.forward(input_tensor)
        
        # Check that output has correct shape
        assert output.shape == (1, 6, 64)
        
        # Check that output is not all zeros
        assert not np.allclose(output, 0)


class TestQuantizationLayer:
    """Test cases for QuantizationLayer."""
    
    def test_quantization_initialization(self):
        """Test that QuantizationLayer initializes correctly."""
        config = QuantizationConfig(
            input_precision="int8",
            weight_precision="int8",
            activation_precision="int8"
        )
        quantization = create_quantization_layer(config)
        
        assert quantization is not None
        assert quantization.config.input_precision == "int8"
        assert quantization.config.weight_precision == "int8"
    
    def test_quantization_forward_pass(self):
        """Test that QuantizationLayer forward pass works."""
        config = QuantizationConfig(
            input_precision="int8",
            weight_precision="int8",
            activation_precision="int8"
        )
        quantization = create_quantization_layer(config)
        
        # Create test input
        input_tensor = np.random.normal(0, 1, (2, 3, 4))
        
        # Forward pass
        output, cache = quantization.forward(input_tensor, operation="quantize")
        
        assert output is not None
        assert output.shape == input_tensor.shape
        assert output.dtype == np.int8
    
    def test_quantization_and_dequantization(self):
        """Test that quantization and dequantization work correctly."""
        config = QuantizationConfig(
            input_precision="int8",
            weight_precision="int8",
            activation_precision="int8"
        )
        quantization = create_quantization_layer(config)
        
        # Create test input
        input_tensor = np.random.normal(0, 1, (2, 3, 4))
        
        # Quantize
        quantized, _ = quantization.forward(input_tensor, operation="quantize")
        
        # Dequantize
        dequantized, _ = quantization.forward(quantized, operation="dequantize")
        
        # Check that dequantized output has correct shape and type
        assert dequantized.shape == input_tensor.shape
        assert dequantized.dtype == np.float32
    
    def test_mixed_precision(self):
        """Test that mixed precision operations work."""
        config = QuantizationConfig(
            input_precision="float32",
            weight_precision="int8",
            activation_precision="int8",
            use_mixed_precision=True
        )
        quantization = create_quantization_layer(config)
        
        # Create test input
        input_tensor = np.random.normal(0, 1, (2, 3, 4))
        
        # Mixed precision forward pass
        output, cache = quantization.forward(input_tensor, operation="mixed_precision")
        
        assert output is not None
        assert output.shape == input_tensor.shape
        assert output.dtype == np.float32


class TestIntegration:
    """Integration tests for all layers working together."""
    
    def test_layers_integration(self):
        """Test that all layers can work together."""
        # Create configurations
        transformer_config = TransformerConfig(num_layers=1, hidden_dim=64, num_heads=2, head_dim=32)
        mamba_config = MambaConfig(num_blocks=1, hidden_dim=64)
        attention_config = AttentionConfig(hidden_dim=64, num_heads=2, head_dim=32)
        quantization_config = QuantizationConfig(input_precision="int8")
        
        # Create layers
        transformer = create_fast_transformer(transformer_config)
        mamba = create_mamba_layer(mamba_config)
        attention = create_dual_chunk_attention(attention_config)
        quantization = create_quantization_layer(quantization_config)
        
        # Test input
        input_ids = np.random.randint(0, 100, (1, 5))
        
        # Process through transformer
        transformer_output, _ = transformer.forward(input_ids)
        
        # Process through mamba
        mamba_output, _ = mamba.forward(input_ids)
        
        # Process through attention
        attention_output, _ = attention.forward(transformer_output)
        
        # Process through quantization
        quantized_output, _ = quantization.forward(attention_output, operation="quantize")
        
        # Check that all outputs have correct shapes
        assert transformer_output.shape == (1, 5, 64)
        assert mamba_output.shape == (1, 5, 64)
        assert attention_output.shape == (1, 5, 64)
        assert quantized_output.shape == (1, 5, 64)
        
        # Check that outputs are not all zeros
        assert not np.allclose(transformer_output, 0)
        assert not np.allclose(mamba_output, 0)
        assert not np.allclose(attention_output, 0)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"]) 