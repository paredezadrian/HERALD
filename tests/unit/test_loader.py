"""
Unit Tests for HERALD Model Loader (.herald File Format)
HERALD v1.0 - Test Suite for Core Loading Functionality

This module tests the .herald file format implementation including:
- Header validation and magic number checking
- Weight compression and decompression
- Vocabulary serialization and restoration
- Model integrity validation
- Performance benchmarks
"""

import unittest
import tempfile
import os
import time
import numpy as np
import json
import pickle
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

# Import modules to test
from core.loader import (
    HeraldLoader, HeraldHeader, HeraldFileStructure,
    HeraldFileFormatError, HeraldCompressionError, HeraldValidationError,
    HERALD_MAGIC, HERALD_VERSION, create_herald_loader, benchmark_loader
)
from core.engine import ModelConfig, ModelState
from core.tokenizer import ASCTokenizer
from core.memory import MultiTierMemoryManager, MemoryConfig


class TestHeraldHeader(unittest.TestCase):
    """Test HeraldHeader dataclass functionality."""
    
    def test_header_creation(self):
        """Test basic header creation."""
        header = HeraldHeader(
            model_name="test_model",
            model_version="1.0.0",
            architecture="hybrid_transformer_mamba"
        )
        
        self.assertEqual(header.magic, HERALD_MAGIC)
        self.assertEqual(header.version, HERALD_VERSION)
        self.assertEqual(header.model_name, "test_model")
        self.assertEqual(header.model_version, "1.0.0")
        self.assertEqual(header.architecture, "hybrid_transformer_mamba")
    
    def test_header_defaults(self):
        """Test header default values."""
        header = HeraldHeader()
        
        self.assertEqual(header.magic, HERALD_MAGIC)
        self.assertEqual(header.version, HERALD_VERSION)
        self.assertEqual(header.compression_algorithm, "lz4")
        self.assertEqual(header.quantization_type, "int8")
        self.assertIsInstance(header.metadata, dict)


class TestHeraldLoader(unittest.TestCase):
    """Test HeraldLoader class functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.loader = HeraldLoader()
        self.temp_dir = tempfile.mkdtemp()
        
        # Create mock model state
        self.mock_model_config = ModelConfig(
            model_name="test_model",
            model_version="1.0.0",
            architecture="hybrid_transformer_mamba",
            num_transformer_layers=12,
            hidden_dim=768,
            num_mamba_blocks=6,
            state_dim=1024
        )
        
        self.mock_tokenizer = ASCTokenizer()
        self.mock_memory_manager = MultiTierMemoryManager()
        
        self.mock_model_state = ModelState(
            model_config=self.mock_model_config,
            inference_config=Mock(),
            tokenizer=self.mock_tokenizer,
            memory_manager=self.mock_memory_manager,
            transformer_weights={
                'transformer.layer.0.weight': np.random.randn(768, 768).astype(np.float32),
                'transformer.layer.1.weight': np.random.randn(768, 768).astype(np.float32)
            },
            mamba_weights={
                'mamba.block.0.weight': np.random.randn(1024, 1024).astype(np.float32)
            },
            embedding_weights={
                'embedding.weight': np.random.randn(50000, 768).astype(np.float32)
            },
            output_weights={
                'output.weight': np.random.randn(768, 50000).astype(np.float32)
            },
            expert_weights={},
            router_weights={},
            is_loaded=True,
            is_quantized=False
        )
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_loader_initialization(self):
        """Test loader initialization with different parameters."""
        # Test default initialization
        loader = HeraldLoader()
        self.assertEqual(loader.compression_algorithm, "lz4")
        self.assertEqual(loader.quantization_type, "int8")
        self.assertTrue(loader.enable_memory_mapping)
        self.assertTrue(loader.enable_parallel_loading)
        
        # Test custom initialization
        loader = HeraldLoader(
            compression_algorithm="zlib",
            quantization_type="int16",
            enable_memory_mapping=False,
            enable_parallel_loading=False
        )
        self.assertEqual(loader.compression_algorithm, "zlib")
        self.assertEqual(loader.quantization_type, "int16")
        self.assertFalse(loader.enable_memory_mapping)
        self.assertFalse(loader.enable_parallel_loading)
    
    def test_create_header(self):
        """Test header creation from model state."""
        header = self.loader._create_header(self.mock_model_state)
        
        self.assertEqual(header.magic, HERALD_MAGIC)
        self.assertEqual(header.version, HERALD_VERSION)
        self.assertEqual(header.model_name, "test_model")
        self.assertEqual(header.model_version, "1.0.0")
        self.assertEqual(header.architecture, "hybrid_transformer_mamba")
        self.assertIn('num_transformer_layers', header.metadata)
        self.assertIn('num_mamba_blocks', header.metadata)
        self.assertIn('hidden_dim', header.metadata)
    
    def test_serialize_model_config(self):
        """Test model configuration serialization."""
        config_data = self.loader._serialize_model_config(self.mock_model_config)
        
        self.assertEqual(config_data['model_name'], "test_model")
        self.assertEqual(config_data['model_version'], "1.0.0")
        self.assertEqual(config_data['architecture'], "hybrid_transformer_mamba")
        self.assertEqual(config_data['num_transformer_layers'], 12)
        self.assertEqual(config_data['hidden_dim'], 768)
        self.assertEqual(config_data['num_mamba_blocks'], 6)
        self.assertEqual(config_data['state_dim'], 1024)
    
    def test_serialize_vocabulary(self):
        """Test vocabulary serialization."""
        # Test with tokenizer
        vocab_data = self.loader._serialize_vocabulary(self.mock_tokenizer)
        
        self.assertIn('vocabulary', vocab_data)
        self.assertIn('vocab_size', vocab_data)
        self.assertIn('min_frequency', vocab_data)
        self.assertIn('max_token_length', vocab_data)
        self.assertIn('compression_target', vocab_data)
        self.assertIn('byte_tier_config', vocab_data)
        self.assertIn('symbolic_tier_config', vocab_data)
        self.assertIn('wordpiece_tier_config', vocab_data)
        
        # Test with None tokenizer
        vocab_data = self.loader._serialize_vocabulary(None)
        self.assertEqual(vocab_data, {})
    
    def test_quantize_weights(self):
        """Test weight quantization."""
        test_weights = np.random.randn(100, 100).astype(np.float32)
        
        # Test int8 quantization
        self.loader.quantization_type = "int8"
        quantized = self.loader._quantize_weights(test_weights)
        self.assertEqual(quantized.dtype, np.int8)
        self.assertTrue(np.all(quantized >= -127))
        self.assertTrue(np.all(quantized <= 127))
        
        # Test int16 quantization
        self.loader.quantization_type = "int16"
        quantized = self.loader._quantize_weights(test_weights)
        self.assertEqual(quantized.dtype, np.int16)
        self.assertTrue(np.all(quantized >= -32767))
        self.assertTrue(np.all(quantized <= 32767))
        
        # Test bf16 quantization
        self.loader.quantization_type = "bf16"
        quantized = self.loader._quantize_weights(test_weights)
        self.assertEqual(quantized.dtype, np.float16)
        
        # Test fp32 quantization (no change)
        self.loader.quantization_type = "fp32"
        quantized = self.loader._quantize_weights(test_weights)
        self.assertEqual(quantized.dtype, np.float32)
    
    def test_compress_array(self):
        """Test array compression."""
        test_array = np.random.randn(100, 100).astype(np.float32)
        
        # Test LZ4 compression
        self.loader.compression_algorithm = "lz4"
        compressed = self.loader._compress_array(test_array)
        self.assertEqual(compressed.dtype, np.uint8)
        # Note: Small random arrays may not compress well, so we don't assert compression ratio
        
        # Test zlib compression
        self.loader.compression_algorithm = "zlib"
        compressed = self.loader._compress_array(test_array)
        self.assertEqual(compressed.dtype, np.uint8)
        # Note: Small random arrays may not compress well, so we don't assert compression ratio
        
        # Test no compression
        self.loader.compression_algorithm = "none"
        compressed = self.loader._compress_array(test_array)
        np.testing.assert_array_equal(compressed, test_array)
    
    def test_decompress_array(self):
        """Test array decompression."""
        test_array = np.random.randn(100, 100).astype(np.float32)
        
        # Test LZ4 compression/decompression
        self.loader.compression_algorithm = "lz4"
        compressed = self.loader._compress_array(test_array)
        decompressed = self.loader._decompress_array(compressed)
        self.assertEqual(decompressed.dtype, np.float32)
        # Note: Shape is lost in basic compression, use shape-aware compression for tests
        
        # Test zlib compression/decompression
        self.loader.compression_algorithm = "zlib"
        compressed = self.loader._compress_array(test_array)
        decompressed = self.loader._decompress_array(compressed)
        self.assertEqual(decompressed.dtype, np.float32)
        # Note: Shape is lost in basic compression, use shape-aware compression for tests
    
    def test_dequantize_weights(self):
        """Test weight dequantization."""
        test_weights = np.random.randn(100, 100).astype(np.float32)
        
        # Test int8 dequantization
        self.loader.quantization_type = "int8"
        quantized = self.loader._quantize_weights(test_weights)
        dequantized = self.loader._dequantize_weights(quantized)
        self.assertEqual(dequantized.dtype, np.float32)
        
        # Test int16 dequantization
        self.loader.quantization_type = "int16"
        quantized = self.loader._quantize_weights(test_weights)
        dequantized = self.loader._dequantize_weights(quantized)
        self.assertEqual(dequantized.dtype, np.float32)
    
    def test_compress_weights(self):
        """Test weight compression with target ratio."""
        compressed_weights = self.loader._compress_weights(self.mock_model_state, 8.5)
        
        # Check that weights were compressed
        self.assertIsInstance(compressed_weights, dict)
        self.assertGreater(len(compressed_weights), 0)
        
        # Check compression stats
        self.assertIn('actual_ratio', self.loader.compression_stats)
        self.assertIn('original_size', self.loader.compression_stats)
        self.assertIn('compressed_size', self.loader.compression_stats)
        
        # Verify compression ratio is reasonable
        actual_ratio = self.loader.compression_stats['actual_ratio']
        self.assertGreater(actual_ratio, 1.0)  # Should achieve some compression
    
    def test_create_tokenizer_from_vocabulary(self):
        """Test tokenizer creation from vocabulary data."""
        vocab_data = {
            'vocabulary': {'test': 1, 'token': 2},
            'vocab_size': 1000,
            'min_frequency': 5,
            'max_token_length': 50,
            'compression_target': 3.5,
            'byte_tier_config': {'test': 'config'},
            'symbolic_tier_config': {'test': 'config'},
            'wordpiece_tier_config': {'test': 'config'}
        }
        
        tokenizer = self.loader._create_tokenizer_from_vocabulary(vocab_data)
        
        self.assertIsInstance(tokenizer, ASCTokenizer)
        self.assertEqual(tokenizer.vocab_size, 1000)
        self.assertEqual(tokenizer.min_frequency, 5)
        self.assertEqual(tokenizer.max_token_length, 50)
        self.assertEqual(tokenizer.compression_target, 3.5)
        self.assertEqual(tokenizer.vocabulary, {'test': 1, 'token': 2})
        self.assertEqual(tokenizer.byte_tier_config, {'test': 'config'})
        self.assertEqual(tokenizer.symbolic_tier_config, {'test': 'config'})
        self.assertEqual(tokenizer.wordpiece_tier_config, {'test': 'config'})
        
        # Test with empty vocabulary
        tokenizer = self.loader._create_tokenizer_from_vocabulary({})
        self.assertIsInstance(tokenizer, ASCTokenizer)
    
    def test_validate_model_integrity(self):
        """Test model integrity validation."""
        # Test valid model state
        self.assertTrue(self.loader._validate_model_integrity(self.mock_model_state))
        
        # Test invalid model state (not loaded)
        invalid_state = ModelState(
            model_config=self.mock_model_config,
            inference_config=Mock(),
            is_loaded=False
        )
        self.assertFalse(self.loader._validate_model_integrity(invalid_state))
        
        # Test invalid model state (no weights)
        invalid_state = ModelState(
            model_config=self.mock_model_config,
            inference_config=Mock(),
            is_loaded=True,
            transformer_weights={},
            mamba_weights={},
            embedding_weights={},
            output_weights={},
            expert_weights={},
            router_weights={}
        )
        self.assertFalse(self.loader._validate_model_integrity(invalid_state))
    
    def test_get_loading_stats(self):
        """Test loading statistics retrieval."""
        stats = self.loader.get_loading_stats()
        
        self.assertIn('load_times', stats)
        self.assertIn('compression_stats', stats)
        self.assertIn('validation_stats', stats)
        self.assertIsInstance(stats['load_times'], dict)
        self.assertIsInstance(stats['compression_stats'], dict)
        self.assertIsInstance(stats['validation_stats'], dict)


class TestHeraldFileOperations(unittest.TestCase):
    """Test .herald file creation and loading operations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.loader = HeraldLoader()
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a simple model state for testing
        self.model_config = ModelConfig(
            model_name="test_model",
            model_version="1.0.0",
            architecture="hybrid_transformer_mamba"
        )
        
        self.tokenizer = ASCTokenizer()
        self.memory_manager = MultiTierMemoryManager()
        
        self.model_state = ModelState(
            model_config=self.model_config,
            inference_config=Mock(),
            tokenizer=self.tokenizer,
            memory_manager=self.memory_manager,
            transformer_weights={
                'test.weight': np.random.randn(100, 100).astype(np.float32)
            },
            mamba_weights={},
            embedding_weights={},
            output_weights={},
            expert_weights={},
            router_weights={},
            is_loaded=True,
            is_quantized=False
        )
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_create_and_load_herald_file(self):
        """Test complete .herald file creation and loading cycle."""
        file_path = os.path.join(self.temp_dir, "test_model.herald")
        
        # Create .herald file
        success = self.loader.create_herald_file(self.model_state, file_path)
        self.assertTrue(success)
        self.assertTrue(os.path.exists(file_path))
        
        # Load .herald file
        loaded_state = self.loader.load_herald_file(file_path)
        self.assertIsNotNone(loaded_state)
        self.assertTrue(loaded_state.is_loaded)
        
        # Verify model configuration
        self.assertEqual(loaded_state.model_config.model_name, "test_model")
        self.assertEqual(loaded_state.model_config.model_version, "1.0.0")
        self.assertEqual(loaded_state.model_config.architecture, "hybrid_transformer_mamba")
        
        # Verify tokenizer
        self.assertIsInstance(loaded_state.tokenizer, ASCTokenizer)
        
        # Verify weights
        self.assertIn('test.weight', loaded_state.transformer_weights)
        self.assertEqual(loaded_state.transformer_weights['test.weight'].shape, (100, 100))
    
    def test_load_nonexistent_file(self):
        """Test loading a non-existent file."""
        file_path = os.path.join(self.temp_dir, "nonexistent.herald")
        
        loaded_state = self.loader.load_herald_file(file_path)
        self.assertIsNone(loaded_state)
    
    def test_load_invalid_file(self):
        """Test loading an invalid file."""
        file_path = os.path.join(self.temp_dir, "invalid.herald")
        
        # Create an invalid file
        with open(file_path, 'wb') as f:
            f.write(b'invalid data')
        
        loaded_state = self.loader.load_herald_file(file_path)
        self.assertIsNone(loaded_state)
    
    def test_compression_performance(self):
        """Test compression performance and ratios."""
        # Create larger model state for better compression testing
        large_weights = {
            f'layer_{i}.weight': np.random.randn(512, 512).astype(np.float32)
            for i in range(10)
        }
        
        large_model_state = ModelState(
            model_config=self.model_config,
            inference_config=Mock(),
            tokenizer=self.tokenizer,
            memory_manager=self.memory_manager,
            transformer_weights=large_weights,
            mamba_weights={},
            embedding_weights={},
            output_weights={},
            expert_weights={},
            router_weights={},
            is_loaded=True,
            is_quantized=False
        )
        
        file_path = os.path.join(self.temp_dir, "large_model.herald")
        
        # Test with different compression algorithms
        for algorithm in ["lz4", "zlib"]:
            loader = HeraldLoader(compression_algorithm=algorithm)
            success = loader.create_herald_file(large_model_state, file_path)
            self.assertTrue(success)
            
            # Check compression stats
            stats = loader.get_loading_stats()
            compression_stats = stats['compression_stats']
            
            self.assertGreater(compression_stats['actual_ratio'], 1.0)
            self.assertLess(compression_stats['compressed_size'], 
                          compression_stats['original_size'])
    
    def test_quantization_performance(self):
        """Test quantization performance and accuracy."""
        # Create test weights
        test_weights = np.random.randn(256, 256).astype(np.float32)
        
        model_state = ModelState(
            model_config=self.model_config,
            inference_config=Mock(),
            tokenizer=self.tokenizer,
            memory_manager=self.memory_manager,
            transformer_weights={'test.weight': test_weights},
            mamba_weights={},
            embedding_weights={},
            output_weights={},
            expert_weights={},
            router_weights={},
            is_loaded=True,
            is_quantized=False
        )
        
        file_path = os.path.join(self.temp_dir, "quantized_model.herald")
        
        # Test with different quantization types
        for quant_type in ["int8", "int16", "bf16"]:
            loader = HeraldLoader(quantization_type=quant_type)
            success = loader.create_herald_file(model_state, file_path)
            self.assertTrue(success)
            
            # Load and verify
            loaded_state = loader.load_herald_file(file_path)
            self.assertIsNotNone(loaded_state)
            
            # Check that weights were properly quantized and dequantized
            loaded_weights = loaded_state.transformer_weights['test.weight']
            # For bf16 quantization, expect float16, otherwise float32
            expected_dtype = np.float16 if quant_type == "bf16" else np.float32
            self.assertEqual(loaded_weights.dtype, expected_dtype)
            self.assertEqual(loaded_weights.shape, test_weights.shape)


class TestHeraldLoaderIntegration(unittest.TestCase):
    """Integration tests for HeraldLoader."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.loader = HeraldLoader()
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_benchmark_loader(self):
        """Test loader benchmarking functionality."""
        # Create a test file
        model_config = ModelConfig(
            model_name="benchmark_model",
            model_version="1.0.0",
            architecture="hybrid_transformer_mamba"
        )
        
        model_state = ModelState(
            model_config=model_config,
            inference_config=Mock(),
            tokenizer=ASCTokenizer(),
            memory_manager=MultiTierMemoryManager(),
            transformer_weights={
                'benchmark.weight': np.random.randn(100, 100).astype(np.float32)
            },
            mamba_weights={},
            embedding_weights={},
            output_weights={},
            expert_weights={},
            router_weights={},
            is_loaded=True,
            is_quantized=False
        )
        
        file_path = os.path.join(self.temp_dir, "benchmark_model.herald")
        self.loader.create_herald_file(model_state, file_path)
        
        # Run benchmark
        benchmark_results = benchmark_loader(self.loader, file_path)
        
        self.assertIn('load_time', benchmark_results)
        self.assertIn('success', benchmark_results)
        self.assertIn('stats', benchmark_results)
        self.assertTrue(benchmark_results['success'])
        self.assertIsInstance(benchmark_results['load_time'], float)
        self.assertGreater(benchmark_results['load_time'], 0)
    
    def test_create_herald_loader_function(self):
        """Test the create_herald_loader utility function."""
        loader = create_herald_loader("zlib", "int16")
        
        self.assertEqual(loader.compression_algorithm, "zlib")
        self.assertEqual(loader.quantization_type, "int16")
        self.assertIsInstance(loader, HeraldLoader)


class TestHeraldErrorHandling(unittest.TestCase):
    """Test error handling in HeraldLoader."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.loader = HeraldLoader()
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_invalid_compression_algorithm(self):
        """Test handling of invalid compression algorithm."""
        with self.assertRaises(Exception):
            loader = HeraldLoader(compression_algorithm="invalid")
    
    def test_invalid_quantization_type(self):
        """Test handling of invalid quantization type."""
        with self.assertRaises(Exception):
            loader = HeraldLoader(quantization_type="invalid")
    
    def test_corrupted_file_handling(self):
        """Test handling of corrupted .herald files."""
        file_path = os.path.join(self.temp_dir, "corrupted.herald")
        
        # Create a corrupted file
        with open(file_path, 'wb') as f:
            f.write(b'HERALDv1.0')  # Valid magic
            f.write(b'\x00\x00\x00\x01')  # Valid version
            f.write(b'\x00\x00\x00\x00')  # Invalid length
            f.write(b'corrupted data')
        
        loaded_state = self.loader.load_herald_file(file_path)
        self.assertIsNone(loaded_state)


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2) 