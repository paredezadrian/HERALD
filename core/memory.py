"""
Multi-Tier Memory Architecture for HERALD

This module implements a three-tier memory system for efficient context management:
- Tier 1: Active Working Memory (8,192 tokens)
- Tier 2: Compressed Context Summaries (32,768 tokens, 4:1 ratio)
- Tier 3: Archived Knowledge Base with lazy loading

Author: HERALD Development Team
License: MIT
"""

import numpy as np
import json
import pickle
import hashlib
import os
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from collections import deque
import threading
import time
from pathlib import Path


@dataclass
class MemoryChunk:
    """Represents a chunk of memory with metadata."""
    tokens: np.ndarray
    attention_weights: Optional[np.ndarray] = None
    chunk_id: str = ""
    timestamp: float = 0.0
    access_count: int = 0
    compression_ratio: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryConfig:
    """Configuration for the multi-tier memory system."""
    # Tier 1: Active Working Memory
    tier1_capacity: int = 8192  # tokens
    tier1_chunk_size: int = 1024
    tier1_overlap: int = 128
    
    # Tier 2: Compressed Context Summaries
    tier2_capacity: int = 32768  # tokens
    tier2_compression_ratio: float = 4.0  # 4:1 compression
    tier2_summary_length: int = 256  # summary tokens per chunk
    
    # Tier 3: Archived Knowledge Base
    tier3_cache_size: int = 1000  # number of cached references
    tier3_lazy_load: bool = True
    
    # General settings
    enable_attention_caching: bool = True
    enable_compression: bool = True
    memory_mapping: bool = True
    bf16_precision: bool = True


class Tier1ActiveMemory:
    """Tier 1: Active Working Memory (8,192 tokens)"""
    
    def __init__(self, config: MemoryConfig):
        self.config = config
        self.capacity = config.tier1_capacity
        self.chunk_size = config.tier1_chunk_size
        self.overlap = config.tier1_overlap
        
        # Main storage using numpy arrays for CPU optimization
        self.tokens = np.zeros((self.capacity,), dtype=np.int32)
        self.attention_weights = None
        if config.enable_attention_caching:
            self.attention_weights = np.zeros((self.capacity, self.capacity), dtype=np.float16)
        
        # Metadata tracking
        self.chunks: List[MemoryChunk] = []
        self.current_position = 0
        self.total_tokens = 0
        self.access_patterns = {}
        
        # Thread safety
        self.lock = threading.RLock()
    
    def add_tokens(self, tokens: np.ndarray) -> List[str]:
        """Add tokens to active memory, returning chunk IDs."""
        with self.lock:
            chunk_ids = []
            remaining_tokens = tokens.copy()
            
            while len(remaining_tokens) > 0:
                # Check if we need to evict old chunks
                if self.total_tokens >= self.capacity:
                    self._evict_oldest_chunks()
                
                # Calculate how many tokens we can add
                tokens_to_add = min(len(remaining_tokens), self.capacity - self.total_tokens)
                
                if tokens_to_add == 0:
                    # Still no space after eviction, skip
                    break
                
                # Add tokens to storage
                start_pos = self.current_position
                end_pos = min(start_pos + tokens_to_add, self.capacity)
                actual_tokens = min(tokens_to_add, self.capacity - start_pos)
                
                self.tokens[start_pos:end_pos] = remaining_tokens[:actual_tokens]
                
                # Create chunk
                chunk_id = self._generate_chunk_id(start_pos, actual_tokens)
                chunk = MemoryChunk(
                    tokens=remaining_tokens[:actual_tokens].copy(),
                    chunk_id=chunk_id,
                    timestamp=time.time(),
                    access_count=1
                )
                
                self.chunks.append(chunk)
                chunk_ids.append(chunk_id)
                
                # Update position and total
                self.current_position = (start_pos + actual_tokens) % self.capacity
                self.total_tokens = min(self.total_tokens + actual_tokens, self.capacity)
                remaining_tokens = remaining_tokens[actual_tokens:]
            
            return chunk_ids
    
    def get_tokens(self, start: int, end: int) -> np.ndarray:
        """Retrieve tokens from active memory."""
        with self.lock:
            if start >= self.total_tokens or end > self.total_tokens:
                raise ValueError(f"Invalid token range: {start}-{end}, total: {self.total_tokens}")
            
            # Simple retrieval from stored chunks
            result = []
            for chunk in self.chunks:
                if len(result) >= end - start:
                    break
                result.extend(chunk.tokens)
            
            # Return the requested range
            return np.array(result[start:end], dtype=np.int32)
    
    def get_attention_weights(self, chunk_id: str) -> Optional[np.ndarray]:
        """Retrieve cached attention weights for a chunk."""
        with self.lock:
            if not self.config.enable_attention_caching:
                return None
            
            # Find chunk and return cached weights
            for chunk in self.chunks:
                if chunk.chunk_id == chunk_id and chunk.attention_weights is not None:
                    chunk.access_count += 1
                    return chunk.attention_weights.copy()
            
            return None
    
    def _evict_oldest_chunks(self):
        """Evict oldest chunks when memory is full."""
        if len(self.chunks) == 0:
            return
        
        # Simple LRU eviction - remove oldest chunks
        chunks_to_remove = max(1, len(self.chunks) // 4)  # Remove at least 1 chunk, up to 25%
        for _ in range(chunks_to_remove):
            if self.chunks:
                chunk = self.chunks.pop(0)
                # Update total tokens
                self.total_tokens = max(0, self.total_tokens - len(chunk.tokens))
    
    def _generate_chunk_id(self, position: int, length: int) -> str:
        """Generate unique chunk ID."""
        data = f"{position}_{length}_{time.time()}".encode()
        return hashlib.md5(data).hexdigest()[:16]
    
    def _update_access_patterns(self, start: int, end: int):
        """Update access pattern tracking."""
        key = f"{start}-{end}"
        self.access_patterns[key] = self.access_patterns.get(key, 0) + 1
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        with self.lock:
            return {
                "total_tokens": self.total_tokens,
                "capacity": self.capacity,
                "utilization": self.total_tokens / self.capacity,
                "chunk_count": len(self.chunks),
                "access_patterns": len(self.access_patterns)
            }


class Tier2CompressedMemory:
    """Tier 2: Compressed Context Summaries (32,768 tokens, 4:1 ratio)"""
    
    def __init__(self, config: MemoryConfig):
        self.config = config
        self.capacity = config.tier2_capacity
        self.compression_ratio = config.tier2_compression_ratio
        self.summary_length = config.tier2_summary_length
        
        # Compressed storage
        self.summaries: List[MemoryChunk] = []
        self.compression_cache = {}
        self.access_count = 0
        
        # Compression algorithms
        self.compression_methods = {
            "attention": self._compress_attention_based,
            "frequency": self._compress_frequency_based,
            "semantic": self._compress_semantic_based
        }
    
    def compress_chunk(self, chunk: MemoryChunk) -> MemoryChunk:
        """Compress a chunk using hierarchical summarization."""
        if not self.config.enable_compression:
            return chunk
        
        # Choose compression method based on content
        method = self._select_compression_method(chunk)
        compressed_tokens = self.compression_methods[method](chunk.tokens)
        
        # Ensure we actually compressed
        if len(compressed_tokens) >= len(chunk.tokens):
            # Fallback to simple downsampling
            step = max(1, len(chunk.tokens) // self.summary_length)
            compressed_tokens = chunk.tokens[::step][:self.summary_length]
        
        # Create compressed chunk
        compressed_chunk = MemoryChunk(
            tokens=compressed_tokens,
            chunk_id=f"compressed_{chunk.chunk_id}",
            timestamp=chunk.timestamp,
            access_count=chunk.access_count,
            compression_ratio=self.compression_ratio,
            metadata={
                "original_length": len(chunk.tokens),
                "compression_method": method,
                "original_chunk_id": chunk.chunk_id
            }
        )
        
        return compressed_chunk
    
    def add_summary(self, summary: MemoryChunk):
        """Add a compressed summary to tier 2."""
        if len(self.summaries) * self.summary_length >= self.capacity:
            # Evict oldest summary
            self.summaries.pop(0)
        
        self.summaries.append(summary)
        self.access_count += 1
    
    def get_summary(self, chunk_id: str) -> Optional[MemoryChunk]:
        """Retrieve a compressed summary by chunk ID."""
        for summary in self.summaries:
            if summary.metadata.get("original_chunk_id") == chunk_id:
                summary.access_count += 1
                return summary
        return None
    
    def reconstruct_context(self, chunk_ids: List[str]) -> np.ndarray:
        """Reconstruct context from compressed summaries."""
        reconstructed_tokens = []
        
        for chunk_id in chunk_ids:
            summary = self.get_summary(chunk_id)
            if summary:
                # Decompress summary
                decompressed = self._decompress_summary(summary)
                reconstructed_tokens.extend(decompressed)
        
        return np.array(reconstructed_tokens, dtype=np.int32)
    
    def _select_compression_method(self, chunk: MemoryChunk) -> str:
        """Select appropriate compression method based on content."""
        tokens = chunk.tokens
        
        # Analyze token patterns
        unique_tokens = len(np.unique(tokens))
        token_diversity = unique_tokens / len(tokens)
        
        if token_diversity < 0.5:
            # Low diversity - use frequency-based compression
            return "frequency"
        elif len(tokens) > 1000:
            # Long sequence - use attention-based compression
            return "attention"
        else:
            # Default to semantic compression
            return "semantic"
    
    def _compress_attention_based(self, tokens: np.ndarray) -> np.ndarray:
        """Compress using attention weight preservation."""
        if len(tokens) <= self.summary_length:
            return tokens
        
        # Calculate attention weights (simplified)
        attention_weights = self._calculate_attention_weights(tokens)
        
        # Select top tokens based on attention scores
        attention_scores = np.sum(attention_weights, axis=1)
        top_indices = np.argsort(attention_scores)[-self.summary_length:]
        
        return tokens[top_indices]
    
    def _compress_frequency_based(self, tokens: np.ndarray) -> np.ndarray:
        """Compress using frequency analysis."""
        if len(tokens) <= self.summary_length:
            return tokens
        
        # Count token frequencies
        unique_tokens, counts = np.unique(tokens, return_counts=True)
        
        # For frequency-based compression, return only unique tokens sorted by frequency
        # This will always compress since we're removing duplicates
        unique_sorted = unique_tokens[np.argsort(counts)[::-1]]
        selected_tokens = unique_sorted[:min(self.summary_length, len(unique_sorted))]
        
        return selected_tokens
    
    def _compress_semantic_based(self, tokens: np.ndarray) -> np.ndarray:
        """Compress using semantic clustering."""
        if len(tokens) <= self.summary_length:
            return tokens
        
        # Simple semantic compression using token embeddings (simplified)
        # In practice, this would use learned embeddings
        chunk_size = len(tokens) // self.summary_length
        compressed = []
        
        for i in range(0, len(tokens), chunk_size):
            chunk = tokens[i:i + chunk_size]
            if len(chunk) > 0:
                # Use mean token as representative
                compressed.append(int(np.mean(chunk)))
        
        return np.array(compressed[:self.summary_length])
    
    def _calculate_attention_weights(self, tokens: np.ndarray) -> np.ndarray:
        """Calculate attention weights for tokens (simplified)."""
        # Simplified attention calculation
        seq_len = len(tokens)
        attention = np.random.rand(seq_len, seq_len)  # Placeholder
        attention = attention / np.sum(attention, axis=1, keepdims=True)
        return attention
    
    def _decompress_summary(self, summary: MemoryChunk) -> List[int]:
        """Decompress a summary back to original tokens."""
        # Simple decompression - repeat tokens to approximate original length
        original_length = summary.metadata.get("original_length", len(summary.tokens))
        decompressed = []
        
        for token in summary.tokens:
            repeat_count = max(1, original_length // len(summary.tokens))
            decompressed.extend([token] * repeat_count)
        
        return decompressed[:original_length]
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get tier 2 memory statistics."""
        return {
            "summary_count": len(self.summaries),
            "capacity": self.capacity,
            "utilization": len(self.summaries) * self.summary_length / self.capacity,
            "compression_ratio": self.compression_ratio,
            "access_count": self.access_count
        }


class Tier3ArchivedMemory:
    """Tier 3: Archived Knowledge Base with lazy loading"""
    
    def __init__(self, config: MemoryConfig, knowledge_path: str = "knowledge"):
        self.config = config
        self.knowledge_path = Path(knowledge_path)
        self.knowledge_path.mkdir(exist_ok=True)
        
        # Reference indexing
        self.reference_index: Dict[str, str] = {}
        self.content_cache: Dict[str, Any] = {}
        self.cache_size = config.tier3_cache_size
        
        # Lazy loading
        self.lazy_load = config.tier3_lazy_load
        self.loaded_references = set()
        
        # Thread safety
        self.lock = threading.RLock()
    
    def store_knowledge(self, content: Any, content_id: str, metadata: Dict[str, Any] = None):
        """Store content in the knowledge base."""
        with self.lock:
            # Generate content hash
            content_hash = self._hash_content(content)
            
            # Create reference
            reference = f"{content_id}_{content_hash[:8]}"
            
            # Store in file system
            file_path = self.knowledge_path / f"{reference}.pkl"
            with open(file_path, 'wb') as f:
                pickle.dump({
                    'content': content,
                    'metadata': metadata or {},
                    'timestamp': time.time()
                }, f)
            
            # Update index
            self.reference_index[reference] = str(file_path)
            
            # Cache if enabled
            if len(self.content_cache) < self.cache_size:
                self.content_cache[reference] = content
    
    def retrieve_knowledge(self, reference: str) -> Optional[Any]:
        """Retrieve content from knowledge base."""
        with self.lock:
            if reference in self.content_cache:
                return self.content_cache[reference]
            
            if reference not in self.reference_index:
                return None
            
            # Lazy load if enabled
            if self.lazy_load and reference not in self.loaded_references:
                content = self._load_from_file(reference)
                if content is not None:
                    self.loaded_references.add(reference)
                    # Cache if space available
                    if len(self.content_cache) < self.cache_size:
                        self.content_cache[reference] = content
                    return content
            else:
                return self._load_from_file(reference)
    
    def search_knowledge(self, query: str) -> List[str]:
        """Search knowledge base for relevant content."""
        with self.lock:
            results = []
            
            for reference, file_path in self.reference_index.items():
                # Simple keyword search (in practice, would use semantic search)
                if query.lower() in reference.lower():
                    results.append(reference)
            
            return results
    
    def _hash_content(self, content: Any) -> str:
        """Generate hash for content."""
        if isinstance(content, np.ndarray):
            content_bytes = content.tobytes()
        else:
            content_bytes = str(content).encode()
        
        return hashlib.sha256(content_bytes).hexdigest()
    
    def _load_from_file(self, reference: str) -> Optional[Any]:
        """Load content from file."""
        if reference not in self.reference_index:
            return None
        
        file_path = self.reference_index[reference]
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                return data['content']
        except (FileNotFoundError, pickle.PickleError):
            return None
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get tier 3 memory statistics."""
        with self.lock:
            return {
                "reference_count": len(self.reference_index),
                "cached_count": len(self.content_cache),
                "loaded_count": len(self.loaded_references),
                "cache_size": self.cache_size,
                "lazy_load_enabled": self.lazy_load
            }


class MultiTierMemoryManager:
    """Main memory manager coordinating all three tiers."""
    
    def __init__(self, config: MemoryConfig = None):
        self.config = config or MemoryConfig()
        
        # Initialize tiers
        self.tier1 = Tier1ActiveMemory(self.config)
        self.tier2 = Tier2CompressedMemory(self.config)
        self.tier3 = Tier3ArchivedMemory(self.config)
        
        # Cross-tier coordination
        self.chunk_mapping: Dict[str, Dict[str, str]] = {}
        self.access_history = deque(maxlen=1000)
        
        # Performance monitoring
        self.stats = {
            "tier1_accesses": 0,
            "tier2_accesses": 0,
            "tier3_accesses": 0,
            "compression_operations": 0,
            "eviction_operations": 0
        }
    
    def add_context(self, tokens: np.ndarray, context_id: str = None) -> str:
        """Add context to memory system."""
        if context_id is None:
            context_id = self._generate_context_id(tokens)
        
        # Add to tier 1
        chunk_ids = self.tier1.add_tokens(tokens)
        
        # Create chunk mapping
        self.chunk_mapping[context_id] = {
            "tier1_chunks": chunk_ids,
            "tier2_summaries": [],
            "tier3_references": []
        }
        
        # Compress and store in tier 2
        for chunk_id in chunk_ids:
            # Find corresponding chunk in tier 1
            chunk = self._find_chunk_by_id(chunk_id)
            if chunk:
                compressed = self.tier2.compress_chunk(chunk)
                self.tier2.add_summary(compressed)
                self.chunk_mapping[context_id]["tier2_summaries"].append(compressed.chunk_id)
        
        self.stats["tier1_accesses"] += 1
        return context_id
    
    def retrieve_context(self, context_id: str, start: int = None, end: int = None) -> np.ndarray:
        """Retrieve context from memory system."""
        if context_id not in self.chunk_mapping:
            raise ValueError(f"Context ID not found: {context_id}")
        
        mapping = self.chunk_mapping[context_id]
        
        # Try tier 1 first
        if mapping["tier1_chunks"]:
            try:
                # Collect all tokens from chunks for this context
                all_tokens = []
                for chunk_id in mapping["tier1_chunks"]:
                    chunk = self._find_chunk_by_id(chunk_id)
                    if chunk:
                        all_tokens.extend(chunk.tokens)
                
                if all_tokens:
                    tokens = np.array(all_tokens, dtype=np.int32)
                    # Apply range if specified
                    if start is not None or end is not None:
                        if start is None:
                            start = 0
                        if end is None:
                            end = len(tokens)
                        tokens = tokens[start:end]
                    
                    self.stats["tier1_accesses"] += 1
                    return tokens
            except ValueError:
                pass
        
        # Fall back to tier 2
        if mapping["tier2_summaries"]:
            tokens = self.tier2.reconstruct_context(mapping["tier2_summaries"])
            self.stats["tier2_accesses"] += 1
            return tokens
        
        # Fall back to tier 3
        if mapping["tier3_references"]:
            tokens = []
            for ref in mapping["tier3_references"]:
                content = self.tier3.retrieve_knowledge(ref)
                if content is not None:
                    if isinstance(content, np.ndarray):
                        tokens.extend(content.tolist())
                    else:
                        tokens.extend(content)
            self.stats["tier3_accesses"] += 1
            return np.array(tokens, dtype=np.int32)
        
        # If no tiers have data, return empty array
        return np.array([], dtype=np.int32)
    
    def archive_context(self, context_id: str, archive_id: str = None):
        """Archive context to tier 3."""
        if context_id not in self.chunk_mapping:
            raise ValueError(f"Context ID not found: {context_id}")
        
        if archive_id is None:
            archive_id = f"archive_{context_id}"
        
        # Retrieve full context
        context_tokens = self.retrieve_context(context_id)
        
        # Store in tier 3
        self.tier3.store_knowledge(
            content=context_tokens,
            content_id=archive_id,
            metadata={
                "original_context_id": context_id,
                "token_count": len(context_tokens),
                "archive_timestamp": time.time()
            }
        )
        
        # Update mapping
        self.chunk_mapping[context_id]["tier3_references"].append(archive_id)
    
    def get_attention_weights(self, context_id: str, chunk_id: str) -> Optional[np.ndarray]:
        """Get cached attention weights."""
        return self.tier1.get_attention_weights(chunk_id)
    
    def cache_attention_weights(self, chunk_id: str, weights: np.ndarray):
        """Cache attention weights for a chunk."""
        if not self.config.enable_attention_caching:
            return
        
        # Find chunk and update weights
        for chunk in self.tier1.chunks:
            if chunk.chunk_id == chunk_id:
                # Ensure weights are in float16 format for consistency
                if weights.dtype != np.float16:
                    chunk.attention_weights = weights.astype(np.float16)
                else:
                    chunk.attention_weights = weights.copy()
                break
    
    def optimize_memory(self):
        """Perform memory optimization."""
        # Check tier 1 utilization
        tier1_stats = self.tier1.get_memory_stats()
        if tier1_stats["utilization"] > 0.9:
            # High utilization, compress some chunks
            self._compress_old_chunks()
        
        # Check tier 2 utilization
        tier2_stats = self.tier2.get_memory_stats()
        if tier2_stats["utilization"] > 0.8:
            # Archive some summaries to tier 3
            self._archive_old_summaries()
    
    def _generate_context_id(self, tokens: np.ndarray) -> str:
        """Generate unique context ID."""
        data = f"{len(tokens)}_{hash(tuple(tokens))}_{time.time()}".encode()
        return hashlib.md5(data).hexdigest()[:16]
    
    def _find_chunk_by_id(self, chunk_id: str) -> Optional[MemoryChunk]:
        """Find chunk by ID in tier 1."""
        for chunk in self.tier1.chunks:
            if chunk.chunk_id == chunk_id:
                return chunk
        return None
    
    def _estimate_context_length(self, context_id: str) -> int:
        """Estimate context length from mapping."""
        mapping = self.chunk_mapping[context_id]
        # Sum up the actual token counts from chunks
        total_tokens = 0
        for chunk_id in mapping["tier1_chunks"]:
            chunk = self._find_chunk_by_id(chunk_id)
            if chunk:
                total_tokens += len(chunk.tokens)
        return total_tokens
    
    def _compress_old_chunks(self):
        """Compress old chunks from tier 1 to tier 2."""
        # Find chunks that haven't been accessed recently
        current_time = time.time()
        old_chunks = []
        
        for chunk in self.tier1.chunks:
            if current_time - chunk.timestamp > 300:  # 5 minutes
                old_chunks.append(chunk)
        
        # Compress oldest chunks
        for chunk in old_chunks[:10]:  # Compress up to 10 chunks
            compressed = self.tier2.compress_chunk(chunk)
            self.tier2.add_summary(compressed)
            self.stats["compression_operations"] += 1
    
    def _archive_old_summaries(self):
        """Archive old summaries from tier 2 to tier 3."""
        # Archive oldest summaries
        old_summaries = self.tier2.summaries[:5]  # Archive 5 oldest
        
        for summary in old_summaries:
            archive_id = f"summary_{summary.chunk_id}"
            self.tier3.store_knowledge(
                content=summary.tokens,
                content_id=archive_id,
                metadata={
                    "original_chunk_id": summary.chunk_id,
                    "compression_ratio": summary.compression_ratio
                }
            )
        
        # Remove from tier 2
        self.tier2.summaries = self.tier2.summaries[5:]
        self.stats["eviction_operations"] += 1
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics."""
        return {
            "tier1": self.tier1.get_memory_stats(),
            "tier2": self.tier2.get_memory_stats(),
            "tier3": self.tier3.get_memory_stats(),
            "performance": self.stats,
            "total_contexts": len(self.chunk_mapping),
            "config": {
                "tier1_capacity": self.config.tier1_capacity,
                "tier2_capacity": self.config.tier2_capacity,
                "compression_ratio": self.config.tier2_compression_ratio,
                "enable_attention_caching": self.config.enable_attention_caching,
                "enable_compression": self.config.enable_compression
            }
        }


# Factory function for easy instantiation
def create_memory_manager(config: MemoryConfig = None) -> MultiTierMemoryManager:
    """Create a new memory manager with optional configuration."""
    return MultiTierMemoryManager(config)


# Example usage and testing
if __name__ == "__main__":
    # Create memory manager
    config = MemoryConfig(
        tier1_capacity=8192,
        tier2_capacity=32768,
        enable_attention_caching=True,
        enable_compression=True
    )
    
    manager = create_memory_manager(config)
    
    # Test adding context
    test_tokens = np.random.randint(0, 1000, 1000, dtype=np.int32)
    context_id = manager.add_context(test_tokens, "test_context")
    
    # Test retrieval
    retrieved = manager.retrieve_context(context_id)
    print(f"Retrieved {len(retrieved)} tokens")
    
    # Test statistics
    stats = manager.get_comprehensive_stats()
    print("Memory Statistics:", json.dumps(stats, indent=2, default=str)) 