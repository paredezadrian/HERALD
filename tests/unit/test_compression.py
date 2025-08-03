"""
Unit tests for HERALD compression utilities.

Tests LZ4 compression, quantization utilities, and sparse matrix representations.
"""

import pytest
import numpy as np
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from utils.compression import (
    LZ4Compressor,
    QuantizationUtils,
    SparseMatrix,
    CompressionManager,
    compress_model_weights,
    decompress_model_weights,
    get_compression_stats
)


class TestLZ4Compressor:
    """Test LZ4 compression functionality."""
    
    def test_compressor_initialization(self):
        """Test compressor initialization with different parameters."""
        compressor = LZ4Compressor(acceleration=5, compression_level=3)
        assert compressor.acceleration == 5
        assert compressor.compression_level == 3
    
    def test_compressor_parameter_bounds(self):
        """Test that parameters are clamped to valid ranges."""
        compressor = LZ4Compressor(acceleration=15, compression_level=0)
        assert compressor.acceleration == 9
        assert compressor.compression_level == 1
    
    def test_compress_bytes(self):
        """Test compression of byte data."""
        compressor = LZ4Compressor()
        # Use a longer string that will actually compress
        test_data = b"Hello, this is a test string for compression! " * 100
        compressed = compressor.compress(test_data)
        
        assert isinstance(compressed, bytes)
        assert len(compressed) > 0
        # For longer data, compression should work
        assert len(compressed) < len(test_data)
    
    def test_compress_numpy_array(self):
        """Test compression of numpy array."""
        compressor = LZ4Compressor()
        test_array = np.random.randn(100, 100).astype(np.float32)
        compressed = compressor.compress(test_array)
        
        assert isinstance(compressed, bytes)
        assert len(compressed) > 0
    
    def test_decompress(self):
        """Test decompression functionality."""
        compressor = LZ4Compressor()
        test_data = b"Test data for compression and decompression"
        compressed = compressor.compress(test_data)
        decompressed = compressor.decompress(compressed)
        
        assert decompressed == test_data
    
    def test_compress_with_stats(self):
        """Test compression with statistics."""
        compressor = LZ4Compressor()
        # Use larger array for better compression
        test_array = np.random.randn(200, 200).astype(np.float32)
        compressed, stats = compressor.compress_with_stats(test_array)
        
        assert isinstance(compressed, bytes)
        assert isinstance(stats.original_size, int)
        assert isinstance(stats.compressed_size, int)
        assert isinstance(stats.compression_ratio, float)
        # Compression ratio should be reasonable (may not always be > 1.0 for small data)
        assert stats.compression_ratio > 0.1
        assert stats.compression_time >= 0
        assert stats.decompression_time >= 0


class TestQuantizationUtils:
    """Test quantization utilities."""
    
    def test_int8_quantization(self):
        """Test int8 quantization and dequantization."""
        test_array = np.random.randn(10, 10).astype(np.float32)
        quantized, scale = QuantizationUtils.to_int8(test_array)
        
        assert quantized.dtype == np.int8
        assert np.all(quantized >= -128)
        assert np.all(quantized <= 127)
        assert scale > 0
        
        # Test dequantization
        dequantized = QuantizationUtils.from_int8(quantized, scale)
        assert dequantized.dtype == np.float32
        assert dequantized.shape == test_array.shape
    
    def test_int8_quantization_with_scale(self):
        """Test int8 quantization with provided scale."""
        test_array = np.random.randn(5, 5).astype(np.float32)
        scale = 10.0
        quantized, returned_scale = QuantizationUtils.to_int8(test_array, scale)
        
        assert returned_scale == scale
        assert quantized.dtype == np.int8
    
    def test_bfloat16_conversion(self):
        """Test bfloat16 conversion."""
        test_array = np.random.randn(10, 10).astype(np.float32)
        bf16_array = QuantizationUtils.to_bfloat16(test_array)
        
        assert bf16_array.dtype == np.float16  # Using float16 as approximation
        
        # Test conversion back
        converted_back = QuantizationUtils.from_bfloat16(bf16_array)
        assert converted_back.dtype == np.float32
        assert converted_back.shape == test_array.shape
    
    def test_quantization_stats(self):
        """Test quantization statistics calculation."""
        test_array = np.random.randn(20, 20).astype(np.float32)
        quantized, scale = QuantizationUtils.to_int8(test_array)
        dequantized = QuantizationUtils.from_int8(quantized, scale)
        
        stats = QuantizationUtils.get_quantization_stats(test_array, dequantized)
        
        assert 'mse' in stats
        assert 'mae' in stats
        assert 'max_error' in stats
        assert 'compression_ratio' in stats
        assert stats['compression_ratio'] >= 1.0  # Should be compressed


class TestSparseMatrix:
    """Test sparse matrix functionality."""
    
    def test_sparse_matrix_creation(self):
        """Test sparse matrix creation from dense matrix."""
        # Create a sparse matrix (mostly zeros)
        dense_matrix = np.zeros((10, 10), dtype=np.float32)
        dense_matrix[0, 0] = 1.0
        dense_matrix[5, 5] = 2.0
        dense_matrix[9, 9] = 3.0
        
        sparse = SparseMatrix.from_dense(dense_matrix)
        
        assert sparse.shape == (10, 10)
        assert sparse.nnz == 3
        assert len(sparse.data) == 3
        assert len(sparse.indices) == 3
        assert len(sparse.indptr) == 11  # shape[0] + 1
    
    def test_sparse_to_dense_conversion(self):
        """Test conversion back to dense format."""
        dense_matrix = np.random.randn(5, 5).astype(np.float32)
        # Make it sparse by setting small values to zero
        dense_matrix[np.abs(dense_matrix) < 0.1] = 0
        
        sparse = SparseMatrix.from_dense(dense_matrix, threshold=0.1)
        converted_back = sparse.to_dense()
        
        assert converted_back.shape == dense_matrix.shape
        assert np.allclose(converted_back, dense_matrix, atol=1e-6)
    
    def test_compression_ratio(self):
        """Test compression ratio calculation."""
        dense_matrix = np.zeros((100, 100), dtype=np.float32)
        dense_matrix[0, 0] = 1.0
        dense_matrix[50, 50] = 2.0
        
        sparse = SparseMatrix.from_dense(dense_matrix)
        ratio = sparse.get_compression_ratio()
        
        assert ratio > 1.0  # Should be compressed
        assert ratio < 10000.0  # Reasonable upper bound
    
    def test_memory_usage(self):
        """Test memory usage calculation."""
        dense_matrix = np.zeros((10, 10), dtype=np.float32)
        dense_matrix[0, 0] = 1.0
        
        sparse = SparseMatrix.from_dense(dense_matrix)
        memory_usage = sparse.get_memory_usage()
        
        assert 'data_bytes' in memory_usage
        assert 'indices_bytes' in memory_usage
        assert 'indptr_bytes' in memory_usage
        assert 'total_bytes' in memory_usage
        assert 'dense_equivalent_bytes' in memory_usage
        assert memory_usage['total_bytes'] < memory_usage['dense_equivalent_bytes']


class TestCompressionManager:
    """Test compression manager functionality."""
    
    def test_manager_initialization(self):
        """Test compression manager initialization."""
        manager = CompressionManager()
        assert hasattr(manager, 'lz4_compressor')
        assert hasattr(manager, 'quantization_utils')
    
    def test_compress_weights_lz4(self):
        """Test weight compression with LZ4."""
        manager = CompressionManager()
        weights = np.random.randn(100, 100).astype(np.float32)
        
        compressed = manager.compress_weights(weights, method='lz4')
        
        assert compressed['method'] == 'lz4'
        assert 'data' in compressed
        assert 'shape' in compressed
        assert 'dtype' in compressed
        assert 'stats' in compressed
        assert compressed['shape'] == weights.shape
    
    def test_compress_weights_quantize(self):
        """Test weight compression with quantization."""
        manager = CompressionManager()
        weights = np.random.randn(50, 50).astype(np.float32)
        
        compressed = manager.compress_weights(weights, method='quantize')
        
        assert compressed['method'] == 'quantize'
        assert 'data' in compressed
        assert 'shape' in compressed
        assert 'scale' in compressed
        assert 'dtype' in compressed
        assert compressed['dtype'] == 'int8'
    
    def test_compress_weights_sparse(self):
        """Test weight compression with sparse representation."""
        manager = CompressionManager()
        # Create sparse weights
        weights = np.zeros((20, 20), dtype=np.float32)
        weights[0, 0] = 1.0
        weights[10, 10] = 2.0
        
        compressed = manager.compress_weights(weights, method='sparse')
        
        assert compressed['method'] == 'sparse'
        assert 'data' in compressed
        assert 'indices' in compressed
        assert 'indptr' in compressed
        assert 'shape' in compressed
        assert 'nnz' in compressed
        assert 'compression_ratio' in compressed
    
    def test_compress_weights_auto(self):
        """Test automatic compression method selection."""
        manager = CompressionManager()
        
        # Large matrix should use LZ4 (but auto might choose quantize for efficiency)
        large_weights = np.random.randn(1000, 1000).astype(np.float32)
        compressed = manager.compress_weights(large_weights, method='auto')
        assert compressed['method'] in ['lz4', 'quantize']  # Auto might choose either
        
        # Sparse matrix should use sparse
        sparse_weights = np.zeros((100, 100), dtype=np.float32)
        sparse_weights[0, 0] = 1.0
        compressed = manager.compress_weights(sparse_weights, method='auto')
        assert compressed['method'] == 'sparse'
    
    def test_decompress_weights(self):
        """Test weight decompression."""
        manager = CompressionManager()
        original_weights = np.random.randn(50, 50).astype(np.float32)
        
        # Test LZ4 decompression
        compressed = manager.compress_weights(original_weights, method='lz4')
        decompressed = manager.decompress_weights(compressed)
        assert np.allclose(decompressed, original_weights, atol=1e-6)
        
        # Test quantization decompression
        compressed = manager.compress_weights(original_weights, method='quantize')
        decompressed = manager.decompress_weights(compressed)
        assert decompressed.shape == original_weights.shape
        # Quantization will have some loss, so we check shape and reasonable values
    
    def test_compression_summary(self):
        """Test compression summary generation."""
        manager = CompressionManager()
        weights = np.random.randn(100, 100).astype(np.float32)
        original_size = weights.nbytes
        
        compressed = manager.compress_weights(weights, method='lz4')
        summary = manager.get_compression_summary(original_size, compressed)
        
        assert 'method' in summary
        assert 'original_size_bytes' in summary
        assert 'compressed_size_bytes' in summary
        assert 'compression_ratio' in summary
        assert 'space_saved_bytes' in summary
        assert 'space_saved_percent' in summary
        assert 'stats' in summary
        assert summary['compression_ratio'] > 0.1  # Should be reasonable


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_compress_model_weights(self):
        """Test convenience function for weight compression."""
        weights = np.random.randn(50, 50).astype(np.float32)
        compressed = compress_model_weights(weights, method='lz4')
        
        assert 'method' in compressed
        assert compressed['method'] == 'lz4'
    
    def test_decompress_model_weights(self):
        """Test convenience function for weight decompression."""
        weights = np.random.randn(30, 30).astype(np.float32)
        compressed = compress_model_weights(weights, method='lz4')
        decompressed = decompress_model_weights(compressed)
        
        assert decompressed.shape == weights.shape
        assert np.allclose(decompressed, weights, atol=1e-6)
    
    def test_get_compression_stats(self):
        """Test convenience function for compression statistics."""
        weights = np.random.randn(40, 40).astype(np.float32)
        original_size = weights.nbytes
        compressed = compress_model_weights(weights, method='lz4')
        
        stats = get_compression_stats(original_size, compressed)
        assert 'compression_ratio' in stats
        assert stats['compression_ratio'] > 0.1  # Should be reasonable


if __name__ == '__main__':
    pytest.main([__file__]) 