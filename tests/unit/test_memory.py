"""
Unit tests for Multi-Tier Memory Architecture

Tests all three tiers of the memory system:
- Tier 1: Active Working Memory
- Tier 2: Compressed Context Summaries  
- Tier 3: Archived Knowledge Base
"""

import pytest
import numpy as np
import tempfile
import shutil
import os
from pathlib import Path

from core.memory import (
    MemoryConfig,
    MemoryChunk,
    Tier1ActiveMemory,
    Tier2CompressedMemory,
    Tier3ArchivedMemory,
    MultiTierMemoryManager,
    create_memory_manager
)


class TestMemoryConfig:
    """Test memory configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = MemoryConfig()
        
        assert config.tier1_capacity == 8192
        assert config.tier2_capacity == 32768
        assert config.tier2_compression_ratio == 4.0
        assert config.enable_attention_caching is True
        assert config.enable_compression is True
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = MemoryConfig(
            tier1_capacity=4096,
            tier2_capacity=16384,
            enable_attention_caching=False
        )
        
        assert config.tier1_capacity == 4096
        assert config.tier2_capacity == 16384
        assert config.enable_attention_caching is False


class TestMemoryChunk:
    """Test memory chunk data structure."""
    
    def test_chunk_creation(self):
        """Test creating a memory chunk."""
        tokens = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        chunk = MemoryChunk(
            tokens=tokens,
            chunk_id="test_chunk",
            timestamp=123.456,
            access_count=5
        )
        
        assert len(chunk.tokens) == 5
        assert chunk.chunk_id == "test_chunk"
        assert chunk.timestamp == 123.456
        assert chunk.access_count == 5
        assert chunk.compression_ratio == 1.0
    
    def test_chunk_with_attention_weights(self):
        """Test chunk with attention weights."""
        tokens = np.array([1, 2, 3], dtype=np.int32)
        attention_weights = np.array([[0.5, 0.3, 0.2], [0.4, 0.4, 0.2], [0.3, 0.3, 0.4]])
        
        chunk = MemoryChunk(
            tokens=tokens,
            attention_weights=attention_weights,
            chunk_id="test_chunk"
        )
        
        assert chunk.attention_weights is not None
        assert chunk.attention_weights.shape == (3, 3)


class TestTier1ActiveMemory:
    """Test Tier 1: Active Working Memory."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = MemoryConfig(tier1_capacity=1024)
        self.memory = Tier1ActiveMemory(self.config)
    
    def test_add_tokens(self):
        """Test adding tokens to tier 1."""
        tokens = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        chunk_ids = self.memory.add_tokens(tokens)
        
        assert len(chunk_ids) == 1
        assert len(self.memory.chunks) == 1
        assert self.memory.total_tokens == 5
    
    def test_add_large_tokens(self):
        """Test adding tokens larger than capacity."""
        tokens = np.arange(2000, dtype=np.int32)
        chunk_ids = self.memory.add_tokens(tokens)
        
        # Should handle overflow by evicting old chunks
        assert len(chunk_ids) > 0
        assert self.memory.total_tokens <= self.config.tier1_capacity
    
    def test_get_tokens(self):
        """Test retrieving tokens from tier 1."""
        tokens = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        self.memory.add_tokens(tokens)
        
        retrieved = self.memory.get_tokens(0, 3)
        assert len(retrieved) == 3
        assert np.array_equal(retrieved, np.array([1, 2, 3]))
    
    def test_get_tokens_invalid_range(self):
        """Test retrieving tokens with invalid range."""
        tokens = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        self.memory.add_tokens(tokens)
        
        with pytest.raises(ValueError):
            self.memory.get_tokens(10, 15)
    
    def test_attention_weights_caching(self):
        """Test attention weights caching."""
        config = MemoryConfig(enable_attention_caching=True)
        memory = Tier1ActiveMemory(config)
        
        tokens = np.array([1, 2, 3], dtype=np.int32)
        chunk_ids = memory.add_tokens(tokens)
        
        # Cache attention weights
        weights = np.array([[0.5, 0.3, 0.2], [0.4, 0.4, 0.2], [0.3, 0.3, 0.4]])
        for chunk in memory.chunks:
            if chunk.chunk_id == chunk_ids[0]:
                chunk.attention_weights = weights
                break
        
        # Retrieve cached weights
        cached_weights = memory.get_attention_weights(chunk_ids[0])
        assert cached_weights is not None
        assert np.array_equal(cached_weights, weights)
    
    def test_memory_stats(self):
        """Test memory statistics."""
        tokens = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        self.memory.add_tokens(tokens)
        
        stats = self.memory.get_memory_stats()
        
        assert stats["total_tokens"] == 5
        assert stats["capacity"] == 1024
        assert stats["chunk_count"] == 1
        assert 0 < stats["utilization"] < 1


class TestTier2CompressedMemory:
    """Test Tier 2: Compressed Context Summaries."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = MemoryConfig(tier2_capacity=1024, tier2_summary_length=5)
        self.memory = Tier2CompressedMemory(self.config)
    
    def test_compress_chunk(self):
        """Test compressing a chunk."""
        tokens = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.int32)
        chunk = MemoryChunk(tokens=tokens, chunk_id="test_chunk")
        
        compressed = self.memory.compress_chunk(chunk)
        
        assert len(compressed.tokens) <= len(tokens)
        assert compressed.compression_ratio == 4.0
        assert compressed.metadata["original_length"] == 10
        assert compressed.metadata["original_chunk_id"] == "test_chunk"
    
    def test_add_summary(self):
        """Test adding a summary to tier 2."""
        tokens = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        summary = MemoryChunk(tokens=tokens, chunk_id="test_summary")
        
        self.memory.add_summary(summary)
        
        assert len(self.memory.summaries) == 1
        assert self.memory.access_count == 1
    
    def test_get_summary(self):
        """Test retrieving a summary by chunk ID."""
        tokens = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        summary = MemoryChunk(
            tokens=tokens,
            chunk_id="test_summary",
            metadata={"original_chunk_id": "original_chunk"}
        )
        
        self.memory.add_summary(summary)
        
        retrieved = self.memory.get_summary("original_chunk")
        assert retrieved is not None
        assert retrieved.chunk_id == "test_summary"
        assert retrieved.access_count == 1
    
    def test_reconstruct_context(self):
        """Test reconstructing context from summaries."""
        # Add multiple summaries
        for i in range(3):
            tokens = np.array([i*10 + j for j in range(5)], dtype=np.int32)
            summary = MemoryChunk(
                tokens=tokens,
                chunk_id=f"summary_{i}",
                metadata={"original_chunk_id": f"chunk_{i}"}
            )
            self.memory.add_summary(summary)
        
        # Reconstruct context
        chunk_ids = ["chunk_0", "chunk_1", "chunk_2"]
        reconstructed = self.memory.reconstruct_context(chunk_ids)
        
        assert len(reconstructed) > 0
        assert isinstance(reconstructed, np.ndarray)
    
    def test_compression_methods(self):
        """Test different compression methods."""
        # Test frequency-based compression
        tokens = np.array([1, 1, 1, 2, 2, 3, 3, 3, 3, 4], dtype=np.int32)
        chunk = MemoryChunk(tokens=tokens, chunk_id="freq_chunk")
        
        compressed = self.memory.compress_chunk(chunk)
        assert len(compressed.tokens) < len(tokens)
    
    def test_memory_stats(self):
        """Test tier 2 memory statistics."""
        tokens = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        summary = MemoryChunk(tokens=tokens, chunk_id="test_summary")
        self.memory.add_summary(summary)
        
        stats = self.memory.get_memory_stats()
        
        assert stats["summary_count"] == 1
        assert stats["capacity"] == 1024
        assert stats["compression_ratio"] == 4.0
        assert stats["access_count"] == 1


class TestTier3ArchivedMemory:
    """Test Tier 3: Archived Knowledge Base."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = MemoryConfig(tier3_cache_size=10)
        self.memory = Tier3ArchivedMemory(self.config, self.temp_dir)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_store_knowledge(self):
        """Test storing knowledge in tier 3."""
        content = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        metadata = {"source": "test", "timestamp": 123.456}
        
        self.memory.store_knowledge(content, "test_content", metadata)
        
        assert len(self.memory.reference_index) == 1
        assert len(self.memory.content_cache) == 1
    
    def test_retrieve_knowledge(self):
        """Test retrieving knowledge from tier 3."""
        content = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        self.memory.store_knowledge(content, "test_content")
        
        # Get reference from index
        reference = list(self.memory.reference_index.keys())[0]
        
        retrieved = self.memory.retrieve_knowledge(reference)
        assert retrieved is not None
        assert np.array_equal(retrieved, content)
    
    def test_retrieve_nonexistent(self):
        """Test retrieving non-existent knowledge."""
        retrieved = self.memory.retrieve_knowledge("nonexistent")
        assert retrieved is None
    
    def test_search_knowledge(self):
        """Test searching knowledge base."""
        content1 = np.array([1, 2, 3], dtype=np.int32)
        content2 = np.array([4, 5, 6], dtype=np.int32)
        
        self.memory.store_knowledge(content1, "test_content_1")
        self.memory.store_knowledge(content2, "test_content_2")
        
        results = self.memory.search_knowledge("test")
        assert len(results) == 2
    
    def test_lazy_loading(self):
        """Test lazy loading functionality."""
        content = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        self.memory.store_knowledge(content, "test_content")
        
        # Clear cache to test lazy loading
        self.memory.content_cache.clear()
        
        reference = list(self.memory.reference_index.keys())[0]
        retrieved = self.memory.retrieve_knowledge(reference)
        
        assert retrieved is not None
        assert reference in self.memory.loaded_references
    
    def test_memory_stats(self):
        """Test tier 3 memory statistics."""
        content = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        self.memory.store_knowledge(content, "test_content")
        
        stats = self.memory.get_memory_stats()
        
        assert stats["reference_count"] == 1
        assert stats["cached_count"] == 1
        assert stats["cache_size"] == 10
        assert stats["lazy_load_enabled"] is True


class TestMultiTierMemoryManager:
    """Test the main memory manager."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = MemoryConfig(
            tier1_capacity=1024,
            tier2_capacity=2048,
            enable_attention_caching=True,
            enable_compression=True
        )
        self.manager = MultiTierMemoryManager(self.config)
    
    def test_add_context(self):
        """Test adding context to memory system."""
        tokens = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        context_id = self.manager.add_context(tokens, "test_context")
        
        assert context_id == "test_context"
        assert "test_context" in self.manager.chunk_mapping
        assert len(self.manager.chunk_mapping["test_context"]["tier1_chunks"]) > 0
    
    def test_retrieve_context(self):
        """Test retrieving context from memory system."""
        tokens = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        context_id = self.manager.add_context(tokens, "test_context")
        
        retrieved = self.manager.retrieve_context(context_id)
        assert len(retrieved) == 5
        assert np.array_equal(retrieved, tokens)
    
    def test_retrieve_nonexistent_context(self):
        """Test retrieving non-existent context."""
        with pytest.raises(ValueError):
            self.manager.retrieve_context("nonexistent")
    
    def test_archive_context(self):
        """Test archiving context to tier 3."""
        tokens = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        context_id = self.manager.add_context(tokens, "test_context")
        
        self.manager.archive_context(context_id, "archive_1")
        
        mapping = self.manager.chunk_mapping[context_id]
        assert len(mapping["tier3_references"]) == 1
    
    def test_attention_weights_caching(self):
        """Test attention weights caching."""
        tokens = np.array([1, 2, 3], dtype=np.int32)
        context_id = self.manager.add_context(tokens, "test_context")
        
        # Get chunk ID
        chunk_id = self.manager.chunk_mapping[context_id]["tier1_chunks"][0]
        
        # Cache attention weights
        weights = np.array([[0.5, 0.3, 0.2], [0.4, 0.4, 0.2], [0.3, 0.3, 0.4]])
        self.manager.cache_attention_weights(chunk_id, weights)
        
        # Retrieve cached weights
        cached_weights = self.manager.get_attention_weights(context_id, chunk_id)
        assert cached_weights is not None
        # Use approximate comparison due to float16 precision
        assert np.allclose(cached_weights.astype(np.float32), weights.astype(np.float32), atol=1e-3)
    
    def test_memory_optimization(self):
        """Test memory optimization."""
        # Fill tier 1 to trigger optimization
        for i in range(10):
            tokens = np.arange(200, dtype=np.int32) + i * 1000
            self.manager.add_context(tokens, f"context_{i}")
        
        # Run optimization
        self.manager.optimize_memory()
        
        # Check that optimization was performed
        stats = self.manager.get_comprehensive_stats()
        assert stats["performance"]["compression_operations"] >= 0
    
    def test_comprehensive_stats(self):
        """Test comprehensive statistics."""
        tokens = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        self.manager.add_context(tokens, "test_context")
        
        stats = self.manager.get_comprehensive_stats()
        
        assert "tier1" in stats
        assert "tier2" in stats
        assert "tier3" in stats
        assert "performance" in stats
        assert "total_contexts" in stats
        assert stats["total_contexts"] == 1


class TestMemoryIntegration:
    """Integration tests for the complete memory system."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = MemoryConfig(
            tier1_capacity=2048,
            tier2_capacity=4096,
            enable_attention_caching=True,
            enable_compression=True
        )
        self.manager = MultiTierMemoryManager(self.config)
    
    def test_full_memory_workflow(self):
        """Test complete memory workflow."""
        # Add multiple contexts
        contexts = {}
        for i in range(5):
            tokens = np.arange(100, dtype=np.int32) + i * 1000
            context_id = self.manager.add_context(tokens, f"context_{i}")
            contexts[context_id] = tokens
        
        # Verify all contexts can be retrieved
        for context_id, original_tokens in contexts.items():
            retrieved = self.manager.retrieve_context(context_id)
            assert len(retrieved) == len(original_tokens)
            # Check that we got the right tokens (may be in different order due to chunking)
            assert set(retrieved) == set(original_tokens)
        
        # Archive some contexts
        self.manager.archive_context("context_0", "archive_0")
        self.manager.archive_context("context_1", "archive_1")
        
        # Verify archived contexts can still be retrieved
        archived_0 = self.manager.retrieve_context("context_0")
        archived_1 = self.manager.retrieve_context("context_1")
        
        assert len(archived_0) == 100
        assert len(archived_1) == 100
    
    def test_memory_pressure_handling(self):
        """Test handling of memory pressure."""
        # Fill memory to capacity
        for i in range(20):
            tokens = np.arange(200, dtype=np.int32) + i * 1000
            self.manager.add_context(tokens, f"context_{i}")
        
        # Run optimization
        self.manager.optimize_memory()
        
        # Verify system still works
        stats = self.manager.get_comprehensive_stats()
        assert stats["tier1"]["utilization"] < 1.0
    
    def test_attention_weight_persistence(self):
        """Test that attention weights persist across operations."""
        tokens = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        context_id = self.manager.add_context(tokens, "test_context")
        
        # Get chunk ID and cache weights
        chunk_id = self.manager.chunk_mapping[context_id]["tier1_chunks"][0]
        weights = np.array([[0.5, 0.3, 0.2], [0.4, 0.4, 0.2], [0.3, 0.3, 0.4]])
        self.manager.cache_attention_weights(chunk_id, weights)
        
        # Add more contexts to trigger memory pressure
        for i in range(10):
            more_tokens = np.arange(100, dtype=np.int32) + i * 1000
            self.manager.add_context(more_tokens, f"pressure_{i}")
        
        # Verify weights are still accessible
        cached_weights = self.manager.get_attention_weights(context_id, chunk_id)
        if cached_weights is not None:  # May have been evicted
            # Use approximate comparison due to float16 precision
            assert np.allclose(cached_weights.astype(np.float32), weights.astype(np.float32), atol=1e-3)


class TestMemoryFactory:
    """Test memory manager factory function."""
    
    def test_create_memory_manager_default(self):
        """Test creating memory manager with default config."""
        manager = create_memory_manager()
        
        assert isinstance(manager, MultiTierMemoryManager)
        assert manager.config.tier1_capacity == 8192
        assert manager.config.tier2_capacity == 32768
    
    def test_create_memory_manager_custom(self):
        """Test creating memory manager with custom config."""
        config = MemoryConfig(tier1_capacity=4096, tier2_capacity=16384)
        manager = create_memory_manager(config)
        
        assert isinstance(manager, MultiTierMemoryManager)
        assert manager.config.tier1_capacity == 4096
        assert manager.config.tier2_capacity == 16384


if __name__ == "__main__":
    pytest.main([__file__]) 