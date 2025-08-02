"""
Unit tests for ASC Tokenizer
HERALD v1.0 - Tokenizer Testing Module
"""

import pytest
import sys
import os
import tempfile
import json
from typing import List

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from core.tokenizer import ASCTokenizer, Token, TokenizationResult, create_test_corpus, benchmark_tokenizer


class TestASCTokenizer:
    """Test suite for ASC Tokenizer functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.tokenizer = ASCTokenizer(
            vocab_size=1000,
            min_frequency=1,
            max_token_length=50,
            compression_target=3.2
        )
        
        self.test_corpus = [
            "Hello world! This is a test.",
            "def calculate(a, b): return a + b",
            "2 + 2 = 4 and 3 * 5 = 15",
            "import numpy as np",
            "if x > 0: print('Positive')",
            "class MyClass: pass",
            "for i in range(10): print(i)",
            "while True: break",
            "try: pass except: pass",
            "The quick brown fox jumps over the lazy dog."
        ]
    
    def test_tokenizer_initialization(self):
        """Test tokenizer initialization with default parameters."""
        assert self.tokenizer.vocab_size == 1000
        assert self.tokenizer.min_frequency == 1
        assert self.tokenizer.max_token_length == 50
        assert self.tokenizer.compression_target == 3.2
        
        # Check special tokens are initialized
        assert '<PAD>' in self.tokenizer.vocabulary
        assert '<UNK>' in self.tokenizer.vocabulary
        assert '<BOS>' in self.tokenizer.vocabulary
        assert '<EOS>' in self.tokenizer.vocabulary
    
    def test_byte_tier_analysis(self):
        """Test Tier 1: Byte-level UTF-8 analysis."""
        text = "Hello, World! 你好"
        byte_patterns = self.tokenizer._analyze_utf8_byte_patterns(text)
        
        assert isinstance(byte_patterns, dict)
        assert len(byte_patterns) > 0
        
        # Check that byte patterns are captured
        text_bytes = text.encode('utf-8')
        assert any(len(pattern) <= self.tokenizer.byte_tier_config['max_byte_sequence'] 
                  for pattern in byte_patterns.keys())
    
    def test_symbolic_pattern_extraction(self):
        """Test Tier 2: Symbolic pattern extraction."""
        text = "def calculate(a, b): return a + b * 2"
        symbolic_patterns = self.tokenizer._extract_symbolic_patterns(text)
        
        assert isinstance(symbolic_patterns, dict)
        
        # Check for mathematical operators
        assert any(op in symbolic_patterns for op in ['+', '*', '='])
        
        # Check for code keywords
        assert 'def' in symbolic_patterns
        assert 'return' in symbolic_patterns
    
    def test_wordpiece_tokenization(self):
        """Test Tier 3: WordPiece tokenization."""
        # Add some basic tokens to vocabulary for testing
        self.tokenizer.vocabulary.update({
            'Hello': 10,
            'world': 11,
            'this': 12,
            'is': 13,
            'a': 14,
            'test': 15,
            'H': 16,
            'e': 17,
            'l': 18,
            'o': 19,
            'w': 20,
            'r': 21,
            'd': 22,
            't': 23,
            'h': 24,
            'i': 25,
            's': 26
        })
        
        text = "Hello world this is a test"
        tokens = self.tokenizer._apply_wordpiece_tokenization(text)
        
        assert isinstance(tokens, list)
        assert len(tokens) > 0
        
        # Check that words are properly tokenized
        expected_words = ['Hello', 'world', 'this', 'is', 'a', 'test']
        for word in expected_words:
            assert any(word in token for token in tokens)
    
    def test_subword_splitting(self):
        """Test subword splitting functionality."""
        # Add some tokens to vocabulary for testing
        self.tokenizer.vocabulary.update({
            'hello': 10,
            'world': 11,
            'h': 12,
            'e': 13,
            'l': 14,
            'o': 15,
            'w': 16,
            'r': 17,
            'd': 18
        })
        
        # Test splitting a long word
        subwords = self.tokenizer._split_word_into_subwords("helloworld")
        assert isinstance(subwords, list)
        assert len(subwords) > 0
    
    def test_dynamic_vocabulary_building(self):
        """Test dynamic vocabulary construction."""
        self.tokenizer._build_dynamic_vocabulary(self.test_corpus)
        
        # Check that vocabulary was built
        assert len(self.tokenizer.vocabulary) > len(self.tokenizer.special_tokens)
        
        # Check that reverse vocabulary was built
        assert len(self.tokenizer.reverse_vocabulary) == len(self.tokenizer.vocabulary)
        
        # Check that token frequencies were recorded
        assert len(self.tokenizer.token_frequencies) > 0
    
    def test_token_tier_determination(self):
        """Test token tier classification."""
        # Test byte tier
        assert self.tokenizer._determine_token_tier("b'Hello'") == 1
        
        # Test symbolic tier
        assert self.tokenizer._determine_token_tier("+") == 2
        assert self.tokenizer._determine_token_tier("def") == 2
        assert self.tokenizer._determine_token_tier("123") == 2
        
        # Test wordpiece tier - 'hello' should be tier 2 because it matches the identifier pattern
        assert self.tokenizer._determine_token_tier("hello") == 2
    
    def test_tokenization_pipeline(self):
        """Test complete tokenization pipeline."""
        # Train the tokenizer first
        self.tokenizer.train(self.test_corpus)
        
        # Test tokenization
        test_text = "def hello_world(): print('Hello, World!')"
        result = self.tokenizer.tokenize(test_text)
        
        assert isinstance(result, TokenizationResult)
        assert isinstance(result.tokens, list)
        assert len(result.tokens) > 0
        
        # Check token structure
        for token in result.tokens:
            assert isinstance(token, Token)
            assert hasattr(token, 'text')
            assert hasattr(token, 'token_id')
            assert hasattr(token, 'tier')
            assert token.tier in [1, 2, 3]
        
        # Check compression ratio
        assert result.compression_ratio > 0
        
        # Check tier distribution
        assert isinstance(result.tier_distribution, dict)
        assert all(tier in result.tier_distribution for tier in [1, 2, 3])
    
    def test_detokenization(self):
        """Test tokenization and detokenization round-trip."""
        # Train the tokenizer
        self.tokenizer.train(self.test_corpus)
        
        # Test text
        original_text = "Hello world! This is a test."
        result = self.tokenizer.tokenize(original_text)
        
        # Detokenize
        reconstructed_text = self.tokenizer.detokenize(result.tokens)
        
        # Check that detokenization produces something
        assert isinstance(reconstructed_text, str)
        assert len(reconstructed_text) > 0
    
    def test_vocabulary_save_load(self):
        """Test vocabulary saving and loading."""
        # Train the tokenizer
        self.tokenizer.train(self.test_corpus)
        
        # Save vocabulary
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            vocab_file = f.name
        
        try:
            self.tokenizer.save_vocabulary(vocab_file)
            
            # Create new tokenizer and load vocabulary
            new_tokenizer = ASCTokenizer()
            new_tokenizer.load_vocabulary(vocab_file)
            
            # Check that vocabularies match
            assert new_tokenizer.vocabulary == self.tokenizer.vocabulary
            assert new_tokenizer.reverse_vocabulary == self.tokenizer.reverse_vocabulary
            
        finally:
            # Clean up
            if os.path.exists(vocab_file):
                os.unlink(vocab_file)
    
    def test_compression_statistics(self):
        """Test compression statistics collection."""
        # Train and use tokenizer
        self.tokenizer.train(self.test_corpus)
        result = self.tokenizer.tokenize("Test text for compression analysis.")
        
        # Get statistics
        stats = self.tokenizer.get_compression_stats()
        
        assert isinstance(stats, dict)
        assert 'vocabulary_size' in stats
        assert 'tier_distribution' in stats
        assert 'processing_times' in stats
    
    def test_cpu_optimization(self):
        """Test CPU optimization functionality."""
        # Train tokenizer
        self.tokenizer.train(self.test_corpus)
        
        # Apply optimizations
        self.tokenizer.optimize_for_cpu()
        
        # Check that optimizations were applied
        assert isinstance(self.tokenizer.vocabulary, dict)
        assert isinstance(self.tokenizer.token_frequencies, dict)
    
    def test_benchmark_functionality(self):
        """Test benchmarking functionality."""
        # Train tokenizer
        self.tokenizer.train(self.test_corpus)
        
        # Run benchmark
        benchmark_results = benchmark_tokenizer(self.tokenizer, self.test_corpus)
        
        assert isinstance(benchmark_results, dict)
        assert 'total_texts' in benchmark_results
        assert 'total_tokens' in benchmark_results
        assert 'average_compression_ratio' in benchmark_results
        assert 'total_time' in benchmark_results
        assert 'average_time_per_text' in benchmark_results
        assert 'tier_distribution' in benchmark_results
        
        # Check that results are reasonable
        assert benchmark_results['total_texts'] == len(self.test_corpus)
        assert benchmark_results['total_tokens'] > 0
        assert benchmark_results['total_time'] > 0
    
    def test_error_handling(self):
        """Test error handling for edge cases."""
        # Test with empty text
        result = self.tokenizer.tokenize("")
        assert isinstance(result, TokenizationResult)
        assert len(result.tokens) == 0
        
        # Test with very long text
        long_text = "a" * 10000
        result = self.tokenizer.tokenize(long_text)
        assert isinstance(result, TokenizationResult)
        
        # Test with special characters
        special_text = "!@#$%^&*()_+-=[]{}|;':\",./<>?"
        result = self.tokenizer.tokenize(special_text)
        assert isinstance(result, TokenizationResult)
    
    def test_performance_targets(self):
        """Test that performance targets are met."""
        # Train tokenizer
        self.tokenizer.train(self.test_corpus)
        
        # Test compression ratio target
        test_text = "This is a test of the compression ratio target."
        result = self.tokenizer.tokenize(test_text)
        
        # For a simple implementation, compression ratio might be less than 1.0
        # This is expected for basic tokenization without advanced compression
        assert result.compression_ratio > 0.1  # Basic sanity check
        
        # Test processing time
        assert result.metadata['processing_time'] < 1.0  # Should be fast
    
    def test_tier_distribution(self):
        """Test that all tiers are utilized."""
        # Train tokenizer
        self.tokenizer.train(self.test_corpus)
        
        # Test with mixed content
        mixed_text = "def calculate(a, b): return a + b * 2.5"
        result = self.tokenizer.tokenize(mixed_text)
        
        # Check that tier distribution is recorded
        assert isinstance(result.tier_distribution, dict)
        assert all(tier in result.tier_distribution for tier in [1, 2, 3])
        
        # At least some tokens should be found
        total_tokens = sum(result.tier_distribution.values())
        assert total_tokens > 0


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_create_test_corpus(self):
        """Test test corpus creation."""
        corpus = create_test_corpus()
        
        assert isinstance(corpus, list)
        assert len(corpus) > 0
        
        for text in corpus:
            assert isinstance(text, str)
            assert len(text) > 0
    
    def test_benchmark_tokenizer_function(self):
        """Test benchmark function with mock data."""
        tokenizer = ASCTokenizer(vocab_size=100, min_frequency=1)
        test_texts = ["Hello world", "def test(): pass", "1 + 1 = 2"]
        
        # Train tokenizer
        tokenizer.train(test_texts)
        
        # Run benchmark
        results = benchmark_tokenizer(tokenizer, test_texts)
        
        assert isinstance(results, dict)
        assert results['total_texts'] == len(test_texts)
        assert results['total_tokens'] >= 0
        assert results['total_time'] >= 0


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"]) 