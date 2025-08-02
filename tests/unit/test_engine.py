"""
Unit tests for NeuroEngine Core
Tests the basic NeuroEngine structure and functionality
"""

import unittest
import numpy as np
import tempfile
import os
import pickle
from unittest.mock import Mock, patch

# Import the engine module
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from core.engine import (
    NeuroEngine, 
    ModelConfig, 
    InferenceConfig, 
    ExpertRouter,
    create_neuro_engine,
    benchmark_engine,
    ModelInitializationError,
    InferenceError
)


class TestModelConfig(unittest.TestCase):
    """Test ModelConfig class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = ModelConfig()
        
        self.assertEqual(config.model_name, "HERALD-v1.0")
        self.assertEqual(config.model_version, "1.0.0")
        self.assertEqual(config.architecture, "hybrid_transformer_mamba")
        self.assertEqual(config.num_transformer_layers, 12)
        self.assertEqual(config.hidden_dim, 768)
        self.assertEqual(config.num_heads, 12)
        self.assertEqual(config.num_mamba_blocks, 6)
        self.assertEqual(config.num_experts, 8)
        self.assertEqual(config.routing_accuracy, 0.85)
        self.assertTrue(config.load_balancing)
        self.assertEqual(config.active_memory_size, 8192)
        self.assertEqual(config.compressed_memory_size, 32768)
        self.assertEqual(config.max_context_length, 1000000)
        self.assertEqual(config.peak_ram_usage, 11.8)
        self.assertTrue(config.cpu_optimization)
        self.assertTrue(config.avx512_support)
        self.assertTrue(config.bf16_precision)
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = ModelConfig(
            model_name="TEST-MODEL",
            num_transformer_layers=6,
            hidden_dim=512,
            num_experts=4
        )
        
        self.assertEqual(config.model_name, "TEST-MODEL")
        self.assertEqual(config.num_transformer_layers, 6)
        self.assertEqual(config.hidden_dim, 512)
        self.assertEqual(config.num_experts, 4)


class TestInferenceConfig(unittest.TestCase):
    """Test InferenceConfig class."""
    
    def test_default_config(self):
        """Test default inference configuration values."""
        config = InferenceConfig()
        
        self.assertEqual(config.max_new_tokens, 100)
        self.assertEqual(config.temperature, 0.7)
        self.assertEqual(config.top_p, 0.9)
        self.assertEqual(config.top_k, 50)
        self.assertEqual(config.repetition_penalty, 1.1)
        self.assertTrue(config.do_sample)
        self.assertEqual(config.pad_token_id, 0)
        self.assertEqual(config.eos_token_id, 2)
        self.assertEqual(config.bos_token_id, 1)
        self.assertTrue(config.enable_attention_caching)
        self.assertTrue(config.enable_compression)
        self.assertTrue(config.memory_mapping)
    
    def test_custom_config(self):
        """Test custom inference configuration values."""
        config = InferenceConfig(
            max_new_tokens=50,
            temperature=0.5,
            top_p=0.8,
            do_sample=False
        )
        
        self.assertEqual(config.max_new_tokens, 50)
        self.assertEqual(config.temperature, 0.5)
        self.assertEqual(config.top_p, 0.8)
        self.assertFalse(config.do_sample)


class TestExpertRouter(unittest.TestCase):
    """Test ExpertRouter class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.router = ExpertRouter(
            num_experts=4,
            routing_accuracy=0.8,
            load_balancing=True
        )
    
    def test_initialization(self):
        """Test router initialization."""
        self.assertEqual(self.router.num_experts, 4)
        self.assertEqual(self.router.routing_accuracy, 0.8)
        self.assertTrue(self.router.load_balancing)
        self.assertEqual(self.router.total_routes, 0)
        self.assertEqual(len(self.router.expert_usage), 4)
        self.assertTrue(np.all(self.router.expert_usage == 0))
    
    def test_routing_scores_calculation(self):
        """Test routing scores calculation."""
        input_tensor = np.random.rand(10, 768)
        routing_scores = self.router._calculate_routing_scores(input_tensor)
        
        self.assertEqual(len(routing_scores), 4)
        self.assertTrue(np.all(routing_scores >= 0))
        self.assertAlmostEqual(np.sum(routing_scores), 1.0, places=6)
    
    def test_load_balancing(self):
        """Test load balancing functionality."""
        # Simulate some expert usage
        self.router.expert_usage = np.array([10, 5, 15, 8])
        self.router.total_routes = 38
        
        routing_scores = np.array([0.25, 0.25, 0.25, 0.25])
        balanced_scores = self.router._apply_load_balancing(routing_scores)
        
        self.assertEqual(len(balanced_scores), 4)
        self.assertTrue(np.all(balanced_scores >= 0))
        self.assertAlmostEqual(np.sum(balanced_scores), 1.0, places=6)
    
    def test_expert_selection(self):
        """Test expert selection."""
        routing_scores = np.array([0.1, 0.3, 0.2, 0.4])
        selected_experts = self.router._select_experts(routing_scores)
        
        self.assertEqual(len(selected_experts), 2)  # Should select top 2
        self.assertIn(3, selected_experts)  # Highest score
        self.assertIn(1, selected_experts)  # Second highest
    
    def test_usage_stats_update(self):
        """Test usage statistics update."""
        expert_indices = [0, 2]
        initial_usage = self.router.expert_usage.copy()
        
        self.router._update_usage_stats(expert_indices)
        
        self.assertEqual(self.router.total_routes, 1)
        self.assertEqual(self.router.expert_usage[0], initial_usage[0] + 1)
        self.assertEqual(self.router.expert_usage[2], initial_usage[2] + 1)
        self.assertEqual(self.router.expert_usage[1], initial_usage[1])  # Unchanged
        self.assertEqual(self.router.expert_usage[3], initial_usage[3])  # Unchanged
    
    def test_routing_stats(self):
        """Test routing statistics."""
        # Simulate some usage
        self.router.expert_usage = np.array([5, 3, 7, 2])
        self.router.total_routes = 17
        
        stats = self.router.get_routing_stats()
        
        self.assertEqual(stats['total_routes'], 17)
        self.assertEqual(stats['expert_usage'], [5, 3, 7, 2])
        self.assertTrue(stats['load_balancing_enabled'])
        self.assertEqual(stats['routing_accuracy'], 0.8)
        
        # Check usage ratios
        expected_ratios = [5/17, 3/17, 7/17, 2/17]
        for i, ratio in enumerate(stats['usage_ratios']):
            self.assertAlmostEqual(ratio, expected_ratios[i], places=6)


class TestNeuroEngine(unittest.TestCase):
    """Test NeuroEngine class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model_config = ModelConfig(
            num_transformer_layers=2,  # Smaller for testing
            num_mamba_blocks=2,
            num_experts=4,
            active_memory_size=1024,
            compressed_memory_size=2048
        )
        self.inference_config = InferenceConfig(
            max_new_tokens=10,
            temperature=0.5
        )
        self.engine = NeuroEngine(self.model_config, self.inference_config)
    
    def test_initialization(self):
        """Test engine initialization."""
        self.assertIsNotNone(self.engine.model_config)
        self.assertIsNotNone(self.engine.inference_config)
        self.assertIsNotNone(self.engine.state)
        self.assertFalse(self.engine.state.is_loaded)
        self.assertIsNone(self.engine.tokenizer)
        self.assertIsNone(self.engine.memory_manager)
        self.assertIsNotNone(self.engine.expert_router)
    
    def test_hardware_optimization_setup(self):
        """Test hardware optimization setup."""
        # This should not raise any exceptions
        self.engine._setup_hardware_optimization()
    
    def test_model_format_validation(self):
        """Test model format validation."""
        # Valid model data
        valid_data = {
            'config': {},
            'weights': {},
            'tokenizer': {},
            'magic_number': 'LITE'
        }
        self.assertTrue(self.engine._validate_model_format(valid_data))
        
        # Invalid model data (missing required keys)
        invalid_data = {
            'config': {},
            'weights': {}
            # Missing 'tokenizer'
        }
        self.assertFalse(self.engine._validate_model_format(invalid_data))
        
        # Invalid magic number
        invalid_magic_data = {
            'config': {},
            'weights': {},
            'tokenizer': {},
            'magic_number': 'INVALID'
        }
        self.assertFalse(self.engine._validate_model_format(invalid_magic_data))
    
    def test_model_config_loading(self):
        """Test model configuration loading."""
        config_data = {
            'model_name': 'TEST-MODEL',
            'num_transformer_layers': 6,
            'hidden_dim': 512
        }
        
        self.engine._load_model_config(config_data)
        
        self.assertEqual(self.engine.model_config.model_name, 'TEST-MODEL')
        self.assertEqual(self.engine.model_config.num_transformer_layers, 6)
        self.assertEqual(self.engine.model_config.hidden_dim, 512)
    
    def test_tokenizer_loading(self):
        """Test tokenizer loading."""
        tokenizer_data = {
            'vocabulary': {'test': 1, 'token': 2},
            'config': {
                'vocab_size': 1000,
                'compression_target': 3.0
            }
        }
        
        self.engine._load_tokenizer(tokenizer_data)
        
        self.assertIsNotNone(self.engine.tokenizer)
        self.assertEqual(self.engine.tokenizer.vocabulary, {'test': 1, 'token': 2})
        self.assertEqual(self.engine.tokenizer.vocab_size, 1000)
        self.assertEqual(self.engine.tokenizer.compression_target, 3.0)
    
    def test_weight_loading(self):
        """Test model weight loading."""
        weights_data = {
            'embeddings': {'token_embeddings': np.random.rand(1000, 768)},
            'transformer': {'layer_0.weight': np.random.rand(768, 768)},
            'mamba': {'block_0.weight': np.random.rand(768, 768)},
            'output': {'output_projection': np.random.rand(1000, 768)},
            'experts': {'expert_0.weight': np.random.rand(768, 768)},
            'router': {'router.weight': np.random.rand(4, 768)}
        }
        
        self.engine._load_model_weights(weights_data)
        
        self.assertIn('token_embeddings', self.engine.state.embedding_weights)
        self.assertIn('layer_0.weight', self.engine.state.transformer_weights)
        self.assertIn('block_0.weight', self.engine.state.mamba_weights)
        self.assertIn('output_projection', self.engine.state.output_weights)
        self.assertIn('expert_0.weight', self.engine.state.expert_weights)
        self.assertIn('router.weight', self.engine.state.router_weights)
    
    def test_memory_manager_initialization(self):
        """Test memory manager initialization."""
        self.engine._initialize_memory_manager()
        
        self.assertIsNotNone(self.engine.memory_manager)
        self.assertEqual(self.engine.memory_manager.config.tier1_capacity, 1024)
        self.assertEqual(self.engine.memory_manager.config.tier2_capacity, 2048)
    
    def test_expert_router_setup(self):
        """Test expert router setup."""
        self.engine._setup_expert_router()
        
        self.assertIsNotNone(self.engine.expert_router)
        self.assertEqual(self.engine.expert_router.num_experts, 4)
        self.assertEqual(self.engine.expert_router.routing_accuracy, 0.85)
    
    def test_model_integrity_validation(self):
        """Test model integrity validation."""
        # Should fail when components are not loaded
        self.assertFalse(self.engine._validate_model_integrity())
        
        # Setup components
        self.engine.tokenizer = Mock()
        self.engine.memory_manager = Mock()
        self.engine.expert_router = Mock()
        self.engine.state.embedding_weights = {'token_embeddings': np.random.rand(1000, 768)}
        self.engine.state.transformer_weights = {'layer_0.weight': np.random.rand(768, 768)}
        self.engine.state.output_weights = {'output_projection': np.random.rand(1000, 768)}
        
        # Should pass with all components loaded
        self.assertTrue(self.engine._validate_model_integrity())
    
    def test_weight_shape_validation(self):
        """Test weight shape validation."""
        # Setup tokenizer with vocabulary
        self.engine.tokenizer = Mock()
        self.engine.tokenizer.vocabulary = {'test': 1, 'token': 2}
        
        # Valid weights
        self.engine.state.embedding_weights = {
            'token_embeddings': np.random.rand(2, 768)  # vocab_size=2, hidden_dim=768
        }
        self.engine.state.transformer_weights = {
            'transformer_layer_0.attention.query.weight': np.random.rand(768, 768)
        }
        
        self.assertTrue(self.engine._validate_weight_shapes())
        
        # Invalid embedding shape
        self.engine.state.embedding_weights = {
            'token_embeddings': np.random.rand(3, 768)  # Wrong vocab size
        }
        self.assertFalse(self.engine._validate_weight_shapes())
    
    def test_token_embedding_retrieval(self):
        """Test token embedding retrieval."""
        # Setup embeddings
        embedding_matrix = np.random.rand(1000, 768)
        self.engine.state.embedding_weights = {'token_embeddings': embedding_matrix}
        
        tokens = np.array([0, 1, 999, 1000])  # Include out-of-bounds token
        embeddings = self.engine._get_token_embeddings(tokens)
        
        self.assertEqual(embeddings.shape, (4, 768))
        # Check that out-of-bounds token was clipped
        self.assertTrue(np.array_equal(embeddings[3], embedding_matrix[999]))
    
    def test_sampling_functions(self):
        """Test token sampling functions."""
        logits = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        # Test repetition penalty
        previous_tokens = [1, 3]
        penalized_logits = self.engine._apply_repetition_penalty(logits, previous_tokens)
        # With repetition_penalty=1.1, penalized values should be larger (penalty > 1.0)
        self.assertGreater(penalized_logits[1], logits[1])
        self.assertGreater(penalized_logits[3], logits[3])
        
        # Test top-k filtering
        self.engine.inference_config.top_k = 3
        filtered_logits = self.engine._apply_top_k_filtering(logits)
        # Should only keep top 3 values
        non_inf_mask = filtered_logits != float('-inf')
        self.assertEqual(np.sum(non_inf_mask), 3)
        
        # Test top-p sampling
        self.engine.inference_config.top_p = 0.8
        nucleus_logits = self.engine._apply_top_p_sampling(logits)
        # Should filter based on cumulative probability
        non_inf_mask = nucleus_logits != float('-inf')
        self.assertGreater(np.sum(non_inf_mask), 0)
    
    def test_softmax(self):
        """Test softmax function."""
        logits = np.array([1.0, 2.0, 3.0])
        probabilities = self.engine._softmax(logits)
        
        self.assertEqual(len(probabilities), 3)
        self.assertTrue(np.all(probabilities >= 0))
        self.assertAlmostEqual(np.sum(probabilities), 1.0, places=6)
    
    def test_performance_stats(self):
        """Test performance statistics."""
        # Add some mock inference times
        self.engine.inference_times.extend([0.1, 0.2, 0.3])
        self.engine.attention_cache_hits = 10
        self.engine.attention_cache_misses = 5
        
        stats = self.engine.get_performance_stats()
        
        self.assertFalse(stats['model_loaded'])
        self.assertIn('avg_inference_time', stats)
        self.assertIn('cache_hit_rate', stats)
        self.assertEqual(stats['attention_cache_hits'], 10)
        self.assertEqual(stats['attention_cache_misses'], 5)
        self.assertEqual(stats['cache_hit_rate'], 10/15)
    
    def test_cache_clearing(self):
        """Test cache clearing."""
        # Add some cache data
        self.engine.state.attention_cache = {'test': np.random.rand(10, 10)}
        self.engine.attention_cache_hits = 10
        self.engine.attention_cache_misses = 5
        
        self.engine.clear_cache()
        
        self.assertEqual(len(self.engine.state.attention_cache), 0)
        self.assertEqual(self.engine.attention_cache_hits, 0)
        self.assertEqual(self.engine.attention_cache_misses, 0)


class TestNeuroEngineIntegration(unittest.TestCase):
    """Test NeuroEngine integration functionality."""
    
    def test_create_neuro_engine(self):
        """Test engine creation function."""
        engine = create_neuro_engine()
        
        self.assertIsInstance(engine, NeuroEngine)
        self.assertIsInstance(engine.model_config, ModelConfig)
        self.assertIsInstance(engine.inference_config, InferenceConfig)
    
    def test_benchmark_engine(self):
        """Test engine benchmarking."""
        engine = create_neuro_engine()
        
        # Mock the engine to be loaded
        engine.state.is_loaded = True
        engine.generate = Mock(return_value="Generated text")
        engine.get_performance_stats = Mock(return_value={'memory_stats': {}})
        
        test_prompts = ["Test prompt 1", "Test prompt 2"]
        
        results = benchmark_engine(engine, test_prompts, num_runs=2)
        
        self.assertEqual(results['total_prompts'], 2)
        self.assertEqual(results['runs_per_prompt'], 2)
        self.assertEqual(results['total_runs'], 4)
        self.assertIn('avg_generation_time', results)
        self.assertIn('total_tokens_generated', results)
    
    def test_benchmark_engine_not_loaded(self):
        """Test benchmarking with unloaded engine."""
        engine = create_neuro_engine()
        
        test_prompts = ["Test prompt"]
        
        with self.assertRaises(ValueError):
            benchmark_engine(engine, test_prompts)


class TestNeuroEngineErrorHandling(unittest.TestCase):
    """Test NeuroEngine error handling."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.engine = create_neuro_engine()
    
    def test_generate_without_loaded_model(self):
        """Test generation without loaded model."""
        with self.assertRaises(InferenceError):
            self.engine.generate("Test prompt")
    
    def test_load_nonexistent_model(self):
        """Test loading nonexistent model file."""
        result = self.engine.load_model("nonexistent_model.herald")
        self.assertFalse(result)
    
    def test_load_invalid_model_format(self):
        """Test loading invalid model format."""
        with tempfile.NamedTemporaryFile(suffix='.herald', delete=False) as f:
            # Write invalid data
            pickle.dump({'invalid': 'data'}, f)
            temp_path = f.name
        
        try:
            result = self.engine.load_model(temp_path)
            self.assertFalse(result)
        finally:
            os.unlink(temp_path)
    
    def test_token_embeddings_not_available(self):
        """Test error when token embeddings are not available."""
        self.engine.state.embedding_weights = {}
        
        with self.assertRaises(InferenceError):
            self.engine._get_token_embeddings(np.array([1, 2, 3]))
    
    def test_output_weights_not_available(self):
        """Test error when output weights are not available."""
        self.engine.state.output_weights = {}
        
        with self.assertRaises(InferenceError):
            self.engine._get_output_logits(np.random.rand(10, 768))


if __name__ == '__main__':
    unittest.main() 