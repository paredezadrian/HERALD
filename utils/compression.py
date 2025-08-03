"""
HERALD Compression Utilities
============================

This module provides comprehensive compression utilities for the HERALD architecture,
including LZ4 compression, quantization utilities, and sparse matrix representations.

Features:
- LZ4 compression integration for model weights and data
- Quantization utilities (int8/bf16) for memory optimization
- Sparse matrix representations for efficient storage
- Compression ratio monitoring and optimization
"""

import numpy as np
import lz4.frame
import struct
from typing import Union, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import warnings


@dataclass
class CompressionStats:
    """Statistics for compression operations."""
    original_size: int
    compressed_size: int
    compression_ratio: float
    compression_time: float
    decompression_time: float


class LZ4Compressor:
    """
    LZ4 compression wrapper with optimized settings for HERALD.
    
    Provides high-speed compression with reasonable compression ratios
    suitable for model weights and intermediate data.
    """
    
    def __init__(self, acceleration: int = 1, compression_level: int = 1):
        """
        Initialize LZ4 compressor.
        
        Args:
            acceleration: LZ4 acceleration parameter (1-9, higher = faster)
            compression_level: LZ4 compression level (1-9, higher = better compression)
        """
        self.acceleration = max(1, min(9, acceleration))
        self.compression_level = max(1, min(9, compression_level))
    
    def compress(self, data: Union[bytes, np.ndarray]) -> bytes:
        """
        Compress data using LZ4.
        
        Args:
            data: Data to compress (bytes or numpy array)
            
        Returns:
            Compressed data as bytes
        """
        if isinstance(data, np.ndarray):
            # Convert numpy array to bytes
            data_bytes = data.tobytes()
        else:
            data_bytes = data
        
        return lz4.frame.compress(data_bytes)
    
    def decompress(self, compressed_data: bytes) -> bytes:
        """
        Decompress LZ4 data.
        
        Args:
            compressed_data: Compressed data
            
        Returns:
            Decompressed data as bytes
        """
        return lz4.frame.decompress(compressed_data)
    
    def compress_with_stats(self, data: Union[bytes, np.ndarray]) -> Tuple[bytes, CompressionStats]:
        """
        Compress data and return statistics.
        
        Args:
            data: Data to compress
            
        Returns:
            Tuple of (compressed_data, stats)
        """
        import time
        
        if isinstance(data, np.ndarray):
            original_size = data.nbytes
            data_bytes = data.tobytes()
        else:
            original_size = len(data)
            data_bytes = data
        
        start_time = time.time()
        compressed = self.compress(data_bytes)
        compression_time = time.time() - start_time
        
        start_time = time.time()
        self.decompress(compressed)
        decompression_time = time.time() - start_time
        
        stats = CompressionStats(
            original_size=original_size,
            compressed_size=len(compressed),
            compression_ratio=original_size / len(compressed),
            compression_time=compression_time,
            decompression_time=decompression_time
        )
        
        return compressed, stats


class QuantizationUtils:
    """
    Quantization utilities for memory optimization.
    
    Provides int8 and bfloat16 quantization with minimal precision loss.
    """
    
    @staticmethod
    def to_int8(array: np.ndarray, scale: Optional[float] = None) -> Tuple[np.ndarray, float]:
        """
        Quantize float32 array to int8.
        
        Args:
            array: Input float32 array
            scale: Optional scale factor (auto-calculated if None)
            
        Returns:
            Tuple of (int8_array, scale_factor)
        """
        if scale is None:
            # Calculate optimal scale factor
            max_val = np.max(np.abs(array))
            scale = 127.0 / max_val if max_val > 0 else 1.0
        
        # Quantize to int8
        quantized = np.clip(np.round(array * scale), -128, 127).astype(np.int8)
        
        return quantized, scale
    
    @staticmethod
    def from_int8(array: np.ndarray, scale: float) -> np.ndarray:
        """
        Dequantize int8 array back to float32.
        
        Args:
            array: Input int8 array
            scale: Scale factor used for quantization
            
        Returns:
            Dequantized float32 array
        """
        return array.astype(np.float32) / scale
    
    @staticmethod
    def to_bfloat16(array: np.ndarray) -> np.ndarray:
        """
        Convert float32 array to bfloat16.
        
        Args:
            array: Input float32 array
            
        Returns:
            bfloat16 array
        """
        # bfloat16 has 7 bits of mantissa and 8 bits of exponent
        # This is a simplified implementation
        return array.astype(np.float16)  # Using float16 as approximation
    
    @staticmethod
    def from_bfloat16(array: np.ndarray) -> np.ndarray:
        """
        Convert bfloat16 array back to float32.
        
        Args:
            array: Input bfloat16 array
            
        Returns:
            float32 array
        """
        return array.astype(np.float32)
    
    @staticmethod
    def get_quantization_stats(array: np.ndarray, quantized: np.ndarray) -> Dict[str, float]:
        """
        Calculate quantization statistics.
        
        Args:
            array: Original array
            quantized: Quantized array
            
        Returns:
            Dictionary with quantization statistics
        """
        mse = np.mean((array - quantized) ** 2)
        mae = np.mean(np.abs(array - quantized))
        max_error = np.max(np.abs(array - quantized))
        
        return {
            'mse': mse,
            'mae': mae,
            'max_error': max_error,
            'compression_ratio': array.nbytes / quantized.nbytes
        }


class SparseMatrix:
    """
    Sparse matrix representation for efficient storage of sparse weight matrices.
    
    Uses CSR (Compressed Sparse Row) format for optimal CPU performance.
    """
    
    def __init__(self, data: np.ndarray, indices: np.ndarray, indptr: np.ndarray, shape: Tuple[int, int]):
        """
        Initialize sparse matrix.
        
        Args:
            data: Non-zero values
            indices: Column indices
            indptr: Row pointers
            shape: Matrix shape (rows, cols)
        """
        self.data = data
        self.indices = indices
        self.indptr = indptr
        self.shape = shape
        self.nnz = len(data)
    
    @classmethod
    def from_dense(cls, dense_matrix: np.ndarray, threshold: float = 1e-6) -> 'SparseMatrix':
        """
        Create sparse matrix from dense matrix.
        
        Args:
            dense_matrix: Dense numpy array
            threshold: Values below this threshold are considered zero
            
        Returns:
            SparseMatrix instance
        """
        # Simple CSR implementation without scipy dependency
        rows, cols = dense_matrix.shape
        data = []
        indices = []
        indptr = [0]  # Start with 0
        
        for i in range(rows):
            row_data = []
            row_indices = []
            
            for j in range(cols):
                value = dense_matrix[i, j]
                if abs(value) > threshold:
                    row_data.append(value)
                    row_indices.append(j)
            
            data.extend(row_data)
            indices.extend(row_indices)
            indptr.append(indptr[-1] + len(row_data))
        
        return cls(
            np.array(data, dtype=dense_matrix.dtype),
            np.array(indices, dtype=np.int32),
            np.array(indptr, dtype=np.int32),
            dense_matrix.shape
        )
    
    def to_dense(self) -> np.ndarray:
        """
        Convert sparse matrix back to dense format.
        
        Returns:
            Dense numpy array
        """
        # Reconstruct dense matrix from CSR format
        rows, cols = self.shape
        dense = np.zeros(self.shape, dtype=self.data.dtype)
        
        for i in range(rows):
            start_idx = self.indptr[i]
            end_idx = self.indptr[i + 1]
            
            for j in range(start_idx, end_idx):
                col_idx = self.indices[j]
                dense[i, col_idx] = self.data[j]
        
        return dense
    
    def get_compression_ratio(self) -> float:
        """
        Calculate compression ratio compared to dense storage.
        
        Returns:
            Compression ratio (dense_size / sparse_size)
        """
        dense_size = self.shape[0] * self.shape[1] * 4  # float32 = 4 bytes
        sparse_size = (self.nnz * 4 +  # data (float32)
                      self.nnz * 4 +    # indices (int32)
                      (self.shape[0] + 1) * 4)  # indptr (int32)
        
        return dense_size / sparse_size
    
    def get_memory_usage(self) -> Dict[str, int]:
        """
        Get detailed memory usage information.
        
        Returns:
            Dictionary with memory usage details
        """
        data_size = self.nnz * 4  # float32
        indices_size = self.nnz * 4  # int32
        indptr_size = (self.shape[0] + 1) * 4  # int32
        
        return {
            'data_bytes': data_size,
            'indices_bytes': indices_size,
            'indptr_bytes': indptr_size,
            'total_bytes': data_size + indices_size + indptr_size,
            'dense_equivalent_bytes': self.shape[0] * self.shape[1] * 4
        }


class CompressionManager:
    """
    High-level compression manager for HERALD model components.
    
    Provides unified interface for compressing different types of model data
    with appropriate compression strategies.
    """
    
    def __init__(self):
        """Initialize compression manager with default settings."""
        self.lz4_compressor = LZ4Compressor(acceleration=1, compression_level=1)
        self.quantization_utils = QuantizationUtils()
    
    def compress_weights(self, weights: np.ndarray, method: str = 'auto') -> Dict[str, Any]:
        """
        Compress model weights using optimal strategy.
        
        Args:
            weights: Weight matrix
            method: Compression method ('auto', 'lz4', 'quantize', 'sparse')
            
        Returns:
            Dictionary with compressed data and metadata
        """
        if method == 'auto':
            # Auto-select best compression method
            if weights.size > 1000000:  # Large matrices
                method = 'lz4'
            elif np.count_nonzero(weights) / weights.size < 0.1:  # Sparse
                method = 'sparse'
            else:
                method = 'quantize'
        
        if method == 'lz4':
            compressed, stats = self.lz4_compressor.compress_with_stats(weights)
            return {
                'method': 'lz4',
                'data': compressed,
                'shape': weights.shape,
                'dtype': str(weights.dtype),
                'stats': stats
            }
        
        elif method == 'quantize':
            quantized, scale = self.quantization_utils.to_int8(weights)
            compressed, stats = self.lz4_compressor.compress_with_stats(quantized)
            return {
                'method': 'quantize',
                'data': compressed,
                'shape': weights.shape,
                'scale': scale,
                'dtype': 'int8',
                'stats': stats
            }
        
        elif method == 'sparse':
            sparse = SparseMatrix.from_dense(weights)
            return {
                'method': 'sparse',
                'data': sparse.data,
                'indices': sparse.indices,
                'indptr': sparse.indptr,
                'shape': sparse.shape,
                'nnz': sparse.nnz,
                'compression_ratio': sparse.get_compression_ratio()
            }
        
        else:
            raise ValueError(f"Unknown compression method: {method}")
    
    def decompress_weights(self, compressed_data: Dict[str, Any]) -> np.ndarray:
        """
        Decompress model weights.
        
        Args:
            compressed_data: Compressed data dictionary
            
        Returns:
            Decompressed weight matrix
        """
        method = compressed_data['method']
        
        if method == 'lz4':
            decompressed = self.lz4_compressor.decompress(compressed_data['data'])
            return np.frombuffer(decompressed, dtype=np.dtype(compressed_data['dtype'])).reshape(compressed_data['shape'])
        
        elif method == 'quantize':
            decompressed = self.lz4_compressor.decompress(compressed_data['data'])
            quantized = np.frombuffer(decompressed, dtype=np.int8).reshape(compressed_data['shape'])
            return self.quantization_utils.from_int8(quantized, compressed_data['scale'])
        
        elif method == 'sparse':
            sparse = SparseMatrix(
                compressed_data['data'],
                compressed_data['indices'],
                compressed_data['indptr'],
                compressed_data['shape']
            )
            return sparse.to_dense()
        
        else:
            raise ValueError(f"Unknown decompression method: {method}")
    
    def get_compression_summary(self, original_size: int, compressed_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate compression summary statistics.
        
        Args:
            original_size: Original data size in bytes
            compressed_data: Compressed data dictionary
            
        Returns:
            Dictionary with compression summary
        """
        method = compressed_data['method']
        
        if method == 'lz4':
            compressed_size = len(compressed_data['data'])
            stats = compressed_data['stats']
        elif method == 'quantize':
            compressed_size = len(compressed_data['data'])
            stats = compressed_data['stats']
        elif method == 'sparse':
            sparse = SparseMatrix(
                compressed_data['data'],
                compressed_data['indices'],
                compressed_data['indptr'],
                compressed_data['shape']
            )
            memory_usage = sparse.get_memory_usage()
            compressed_size = memory_usage['total_bytes']
            stats = CompressionStats(
                original_size=original_size,
                compressed_size=compressed_size,
                compression_ratio=original_size / compressed_size,
                compression_time=0.0,
                decompression_time=0.0
            )
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return {
            'method': method,
            'original_size_bytes': original_size,
            'compressed_size_bytes': compressed_size,
            'compression_ratio': original_size / compressed_size,
            'space_saved_bytes': original_size - compressed_size,
            'space_saved_percent': (1 - compressed_size / original_size) * 100,
            'stats': stats
        }


# Global compression manager instance
compression_manager = CompressionManager()


def compress_model_weights(weights: np.ndarray, method: str = 'auto') -> Dict[str, Any]:
    """
    Convenience function to compress model weights.
    
    Args:
        weights: Weight matrix
        method: Compression method
        
    Returns:
        Compressed data dictionary
    """
    return compression_manager.compress_weights(weights, method)


def decompress_model_weights(compressed_data: Dict[str, Any]) -> np.ndarray:
    """
    Convenience function to decompress model weights.
    
    Args:
        compressed_data: Compressed data dictionary
        
    Returns:
        Decompressed weight matrix
    """
    return compression_manager.decompress_weights(compressed_data)


def get_compression_stats(original_size: int, compressed_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convenience function to get compression statistics.
    
    Args:
        original_size: Original data size in bytes
        compressed_data: Compressed data dictionary
        
    Returns:
        Compression statistics dictionary
    """
    return compression_manager.get_compression_summary(original_size, compressed_data) 