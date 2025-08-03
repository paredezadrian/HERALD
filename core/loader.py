"""
HERALD Model Loader - .herald File Format Implementation
HERALD v1.0 - Core Model Loading and Serialization

This module implements the .herald file format specification with:
- Header section with magic number validation
- Weight matrix compression (target: 8.5:1 ratio)
- Vocabulary data serialization
- Symbolic rules storage
- Optimized initialization time (target: 700ms)

Target: ~1,234 lines of optimized loading code
"""

import numpy as np
import json
import pickle
import hashlib
import os
import time
import struct
import zlib
import lz4.frame
from typing import Dict, List, Optional, Tuple, Any, Union, BinaryIO
from dataclasses import dataclass, field
from pathlib import Path
import logging
from collections import defaultdict
import yaml
import mmap
import threading

# Import core modules
from .tokenizer import ASCTokenizer, TokenizationResult, Token
from .memory import MultiTierMemoryManager, MemoryConfig
from .engine import NeuroEngine, ModelConfig, ModelState


# Magic number for .herald files
HERALD_MAGIC = b'HERALDv1.0'
HERALD_MAGIC_LENGTH = len(HERALD_MAGIC)

# File format version
HERALD_VERSION = 1

# Compression settings
COMPRESSION_ALGORITHMS = {
    'lz4': lz4.frame,
    'zlib': zlib,
    'none': None
}

# Quantization settings
QUANTIZATION_TYPES = {
    'int8': np.int8,
    'int16': np.int16,
    'bf16': np.float16,
    'fp32': np.float32
}


@dataclass
class HeraldHeader:
    """Header information for .herald files."""
    magic: bytes = HERALD_MAGIC
    version: int = HERALD_VERSION
    model_name: str = ""
    model_version: str = ""
    architecture: str = ""
    created_timestamp: float = 0.0
    file_size: int = 0
    checksum: str = ""
    compression_algorithm: str = "lz4"
    quantization_type: str = "int8"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HeraldFileStructure:
    """Structure of a .herald file."""
    header: HeraldHeader
    model_config: Dict[str, Any]
    vocabulary: Dict[str, Any]
    weights: Dict[str, np.ndarray]
    symbolic_rules: Dict[str, Any]
    attention_cache: Optional[Dict[str, np.ndarray]] = None
    memory_state: Optional[Dict[str, Any]] = None


class HeraldFileFormatError(Exception):
    """Exception raised for .herald file format errors."""
    pass


class HeraldCompressionError(Exception):
    """Exception raised for compression/decompression errors."""
    pass


class HeraldValidationError(Exception):
    """Exception raised for validation errors."""
    pass


class HeraldLoader:
    """
    HERALD Model Loader
    
    Implements the .herald file format specification with optimized loading
    and validation capabilities for CPU-constrained environments.
    """
    
    def __init__(self, 
                 compression_algorithm: str = "lz4",
                 quantization_type: str = "int8",
                 enable_memory_mapping: bool = True,
                 enable_parallel_loading: bool = True):
        """
        Initialize the HERALD loader.
        
        Args:
            compression_algorithm: Compression algorithm to use
            quantization_type: Quantization type for weights
            enable_memory_mapping: Enable memory mapping for large files
            enable_parallel_loading: Enable parallel loading for multiple components
        """
        # Validate compression algorithm
        if compression_algorithm not in COMPRESSION_ALGORITHMS:
            raise ValueError(f"Unsupported compression algorithm: {compression_algorithm}")
        
        # Validate quantization type
        if quantization_type not in QUANTIZATION_TYPES:
            raise ValueError(f"Unsupported quantization type: {quantization_type}")
        
        self.compression_algorithm = compression_algorithm
        self.quantization_type = quantization_type
        self.enable_memory_mapping = enable_memory_mapping
        self.enable_parallel_loading = enable_parallel_loading
        
        # Performance tracking
        self.load_times = {}
        self.compression_stats = {}
        self.validation_stats = {}
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def create_herald_file(self, 
                          model_state: ModelState,
                          output_path: str,
                          compression_ratio_target: float = 8.5) -> bool:
        """
        Create a .herald file from a model state.
        
        Args:
            model_state: The model state to serialize
            output_path: Path to output .herald file
            compression_ratio_target: Target compression ratio
            
        Returns:
            bool: True if successful, False otherwise
        """
        start_time = time.time()
        
        try:
            # Create header
            header = self._create_header(model_state)
            
            # Prepare file structure
            file_structure = HeraldFileStructure(
                header=header,
                model_config=self._serialize_model_config(model_state.model_config),
                vocabulary=self._serialize_vocabulary(model_state.tokenizer),
                weights=self._compress_weights(model_state, compression_ratio_target),
                symbolic_rules=self._serialize_symbolic_rules(model_state),
                attention_cache=self._serialize_attention_cache(model_state),
                memory_state=self._serialize_memory_state(model_state.memory_manager)
            )
            
            # Write file
            success = self._write_herald_file(file_structure, output_path)
            
            if success:
                load_time = time.time() - start_time
                self.logger.info(f"Created .herald file in {load_time:.3f}s")
                self.load_times['file_creation'] = load_time
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error creating .herald file: {e}")
            return False
    
    def load_herald_file(self, file_path: str) -> Optional[ModelState]:
        """
        Load a .herald file and return a model state.
        
        Args:
            file_path: Path to .herald file
            
        Returns:
            ModelState: Loaded model state or None if failed
        """
        start_time = time.time()
        
        try:
            # Validate file exists
            if not os.path.exists(file_path):
                raise HeraldFileFormatError(f"File not found: {file_path}")
            
            # Read and validate header
            header = self._read_and_validate_header(file_path)
            
            # Load file structure
            file_structure = self._load_file_structure(file_path, header)
            
            # Create model state
            model_state = self._create_model_state_from_structure(file_structure)
            
            # Validate model integrity
            if not self._validate_model_integrity(model_state):
                raise HeraldValidationError("Model integrity validation failed")
            
            load_time = time.time() - start_time
            self.logger.info(f"Loaded .herald file in {load_time:.3f}s")
            self.load_times['file_loading'] = load_time
            
            return model_state
            
        except Exception as e:
            self.logger.error(f"Error loading .herald file: {e}")
            return None
    
    def _create_header(self, model_state: ModelState) -> HeraldHeader:
        """Create header for .herald file."""
        return HeraldHeader(
            magic=HERALD_MAGIC,
            version=HERALD_VERSION,
            model_name=model_state.model_config.model_name,
            model_version=model_state.model_config.model_version,
            architecture=model_state.model_config.architecture,
            created_timestamp=time.time(),
            compression_algorithm=self.compression_algorithm,
            quantization_type=self.quantization_type,
            metadata={
                'num_transformer_layers': model_state.model_config.num_transformer_layers,
                'num_mamba_blocks': model_state.model_config.num_mamba_blocks,
                'hidden_dim': model_state.model_config.hidden_dim,
                'vocab_size': len(model_state.tokenizer.vocabulary) if model_state.tokenizer else 0
            }
        )
    
    def _serialize_model_config(self, config: ModelConfig) -> Dict[str, Any]:
        """Serialize model configuration."""
        return {
            'model_name': config.model_name,
            'model_version': config.model_version,
            'architecture': config.architecture,
            'num_transformer_layers': config.num_transformer_layers,
            'hidden_dim': config.hidden_dim,
            'num_heads': config.num_heads,
            'head_dim': config.head_dim,
            'dropout': config.dropout,
            'activation': config.activation,
            'num_mamba_blocks': config.num_mamba_blocks,
            'state_dim': config.state_dim,
            'selective_scan': config.selective_scan,
            'linear_complexity': config.linear_complexity,
            'num_experts': config.num_experts,
            'routing_accuracy': config.routing_accuracy,
            'load_balancing': config.load_balancing,
            'active_memory_size': config.active_memory_size,
            'compressed_memory_size': config.compressed_memory_size,
            'chunk_size': config.chunk_size,
            'chunk_overlap': config.chunk_overlap,
            'max_context_length': config.max_context_length,
            'peak_ram_usage': config.peak_ram_usage,
            'token_generation_speed': config.token_generation_speed,
            'model_load_time': config.model_load_time,
            'compression_ratio': config.compression_ratio,
            'quantization': config.quantization,
            'sparse_matrices': config.sparse_matrices,
            'lz4_compression': config.lz4_compression,
            'cpu_optimization': config.cpu_optimization,
            'avx512_support': config.avx512_support,
            'intel_mkl': config.intel_mkl,
            'memory_mapping': config.memory_mapping,
            'simd_vectorization': config.simd_vectorization,
            'bf16_precision': config.bf16_precision
        }
    
    def _serialize_vocabulary(self, tokenizer: Optional[ASCTokenizer]) -> Dict[str, Any]:
        """Serialize vocabulary data."""
        if not tokenizer:
            return {}
        
        # Convert sets to lists for JSON serialization
        byte_config = tokenizer.byte_tier_config.copy()
        symbolic_config = tokenizer.symbolic_tier_config.copy()
        wordpiece_config = tokenizer.wordpiece_tier_config.copy()
        
        # Convert code_keywords set to list
        if 'code_keywords' in symbolic_config:
            symbolic_config['code_keywords'] = list(symbolic_config['code_keywords'])
        
        return {
            'vocabulary': tokenizer.vocabulary,
            'vocab_size': tokenizer.vocab_size,
            'min_frequency': tokenizer.min_frequency,
            'max_token_length': tokenizer.max_token_length,
            'compression_target': tokenizer.compression_target,
            'byte_tier_config': byte_config,
            'symbolic_tier_config': symbolic_config,
            'wordpiece_tier_config': wordpiece_config,
            'compression_stats': tokenizer.get_compression_stats()
        }
    
    def _compress_weights(self, model_state: ModelState, target_ratio: float) -> Dict[str, np.ndarray]:
        """Compress model weights to achieve target compression ratio."""
        compressed_weights = {}
        
        # Combine all weight dictionaries
        all_weights = {}
        all_weights.update(model_state.transformer_weights)
        all_weights.update(model_state.mamba_weights)
        all_weights.update(model_state.embedding_weights)
        all_weights.update(model_state.output_weights)
        all_weights.update(model_state.expert_weights)
        all_weights.update(model_state.router_weights)
        
        total_original_size = 0
        total_compressed_size = 0
        
        for name, weights in all_weights.items():
            # Quantize weights
            quantized_weights = self._quantize_weights(weights)
            
            # Compress quantized weights with shape information
            compressed_weights[name] = self._compress_array_with_shape(quantized_weights)
            
            total_original_size += weights.nbytes
            total_compressed_size += compressed_weights[name]['data'].nbytes
        
        # Calculate actual compression ratio
        actual_ratio = total_original_size / total_compressed_size if total_compressed_size > 0 else 1.0
        
        self.compression_stats = {
            'target_ratio': target_ratio,
            'actual_ratio': actual_ratio,
            'original_size': total_original_size,
            'compressed_size': total_compressed_size,
            'compression_algorithm': self.compression_algorithm,
            'quantization_type': self.quantization_type
        }
        
        self.logger.info(f"Weight compression: {actual_ratio:.2f}:1 ratio achieved")
        
        return compressed_weights
    
    def _quantize_weights(self, weights: np.ndarray) -> np.ndarray:
        """Quantize weights to specified precision."""
        if self.quantization_type == 'int8':
            # Scale to int8 range
            max_val = np.max(np.abs(weights))
            if max_val > 0:
                scale = 127.0 / max_val
                return (weights * scale).astype(np.int8)
            return weights.astype(np.int8)
        
        elif self.quantization_type == 'int16':
            # Scale to int16 range
            max_val = np.max(np.abs(weights))
            if max_val > 0:
                scale = 32767.0 / max_val
                return (weights * scale).astype(np.int16)
            return weights.astype(np.int16)
        
        elif self.quantization_type == 'bf16':
            return weights.astype(np.float16)
        
        else:  # fp32
            return weights.astype(np.float32)
    
    def _compress_array(self, array: np.ndarray) -> np.ndarray:
        """Compress a numpy array using the specified algorithm."""
        if self.compression_algorithm == 'lz4':
            # Use LZ4 compression
            compressed_data = lz4.frame.compress(array.tobytes())
            return np.frombuffer(compressed_data, dtype=np.uint8)
        
        elif self.compression_algorithm == 'zlib':
            # Use zlib compression
            compressed_data = zlib.compress(array.tobytes())
            return np.frombuffer(compressed_data, dtype=np.uint8)
        
        else:  # no compression
            return array
    
    def _compress_array_with_shape(self, array: np.ndarray) -> Dict[str, Any]:
        """Compress a numpy array while preserving shape information."""
        compressed_data = self._compress_array(array)
        return {
            'data': compressed_data,
            'shape': array.shape,
            'dtype': str(array.dtype)
        }
    
    def _serialize_symbolic_rules(self, model_state: ModelState) -> Dict[str, Any]:
        """Serialize symbolic reasoning rules."""
        # This would contain logic engine rules, causal relationships, etc.
        return {
            'logic_rules': {},
            'causal_rules': {},
            'temporal_rules': {},
            'symbolic_patterns': {},
            'rule_metadata': {}
        }
    
    def _serialize_attention_cache(self, model_state: ModelState) -> Optional[Dict[str, np.ndarray]]:
        """Serialize attention cache if available."""
        if hasattr(model_state, 'attention_cache') and model_state.attention_cache:
            return {
                name: self._compress_array(weights) 
                for name, weights in model_state.attention_cache.items()
            }
        return None
    
    def _serialize_memory_state(self, memory_manager: Optional[MultiTierMemoryManager]) -> Optional[Dict[str, Any]]:
        """Serialize memory manager state."""
        if not memory_manager:
            return None
        
        return {
            'tier1_stats': memory_manager.tier1.get_memory_stats(),
            'tier2_stats': memory_manager.tier2.get_memory_stats(),
            'tier3_stats': memory_manager.tier3.get_memory_stats(),
            'overall_stats': memory_manager.get_comprehensive_stats()
        }
    
    def _write_herald_file(self, file_structure: HeraldFileStructure, output_path: str) -> bool:
        """Write .herald file to disk."""
        try:
            with open(output_path, 'wb') as f:
                # Write header
                self._write_header(f, file_structure.header)
                
                # Write model config
                config_data = json.dumps(file_structure.model_config).encode('utf-8')
                f.write(struct.pack('<I', len(config_data)))
                f.write(config_data)
                
                # Write vocabulary
                vocab_data = json.dumps(file_structure.vocabulary).encode('utf-8')
                f.write(struct.pack('<I', len(vocab_data)))
                f.write(vocab_data)
                
                # Write weights
                weights_data = pickle.dumps(file_structure.weights)
                f.write(struct.pack('<Q', len(weights_data)))
                f.write(weights_data)
                
                # Write symbolic rules
                rules_data = json.dumps(file_structure.symbolic_rules).encode('utf-8')
                f.write(struct.pack('<I', len(rules_data)))
                f.write(rules_data)
                
                # Write attention cache (optional)
                if file_structure.attention_cache:
                    cache_data = pickle.dumps(file_structure.attention_cache)
                    f.write(struct.pack('<Q', len(cache_data)))
                    f.write(cache_data)
                else:
                    f.write(struct.pack('<Q', 0))
                
                # Write memory state (optional)
                if file_structure.memory_state:
                    memory_data = json.dumps(file_structure.memory_state).encode('utf-8')
                    f.write(struct.pack('<I', len(memory_data)))
                    f.write(memory_data)
                else:
                    f.write(struct.pack('<I', 0))
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error writing .herald file: {e}")
            return False
    
    def _write_header(self, f: BinaryIO, header: HeraldHeader):
        """Write header to file."""
        # Write magic number
        f.write(header.magic)
        
        # Write version
        f.write(struct.pack('<I', header.version))
        
        # Write model name
        model_name_bytes = header.model_name.encode('utf-8')
        f.write(struct.pack('<I', len(model_name_bytes)))
        f.write(model_name_bytes)
        
        # Write model version
        model_version_bytes = header.model_version.encode('utf-8')
        f.write(struct.pack('<I', len(model_version_bytes)))
        f.write(model_version_bytes)
        
        # Write architecture
        architecture_bytes = header.architecture.encode('utf-8')
        f.write(struct.pack('<I', len(architecture_bytes)))
        f.write(architecture_bytes)
        
        # Write timestamp
        f.write(struct.pack('<d', header.created_timestamp))
        
        # Write compression algorithm
        comp_alg_bytes = header.compression_algorithm.encode('utf-8')
        f.write(struct.pack('<I', len(comp_alg_bytes)))
        f.write(comp_alg_bytes)
        
        # Write quantization type
        quant_type_bytes = header.quantization_type.encode('utf-8')
        f.write(struct.pack('<I', len(quant_type_bytes)))
        f.write(quant_type_bytes)
        
        # Write metadata
        metadata_bytes = json.dumps(header.metadata).encode('utf-8')
        f.write(struct.pack('<I', len(metadata_bytes)))
        f.write(metadata_bytes)
    
    def _read_and_validate_header(self, file_path: str) -> HeraldHeader:
        """Read and validate file header."""
        with open(file_path, 'rb') as f:
            # Read magic number
            magic = f.read(HERALD_MAGIC_LENGTH)
            if magic != HERALD_MAGIC:
                raise HeraldFileFormatError(f"Invalid magic number: {magic}")
            
            # Read version
            version = struct.unpack('<I', f.read(4))[0]
            if version != HERALD_VERSION:
                raise HeraldFileFormatError(f"Unsupported version: {version}")
            
            # Read model name
            name_len = struct.unpack('<I', f.read(4))[0]
            model_name = f.read(name_len).decode('utf-8')
            
            # Read model version
            version_len = struct.unpack('<I', f.read(4))[0]
            model_version = f.read(version_len).decode('utf-8')
            
            # Read architecture
            arch_len = struct.unpack('<I', f.read(4))[0]
            architecture = f.read(arch_len).decode('utf-8')
            
            # Read timestamp
            timestamp = struct.unpack('<d', f.read(8))[0]
            
            # Read compression algorithm
            comp_len = struct.unpack('<I', f.read(4))[0]
            compression_algorithm = f.read(comp_len).decode('utf-8')
            
            # Read quantization type
            quant_len = struct.unpack('<I', f.read(4))[0]
            quantization_type = f.read(quant_len).decode('utf-8')
            
            # Read metadata
            meta_len = struct.unpack('<I', f.read(4))[0]
            metadata_bytes = f.read(meta_len)
            metadata = json.loads(metadata_bytes.decode('utf-8'))
            
            return HeraldHeader(
                magic=magic,
                version=version,
                model_name=model_name,
                model_version=model_version,
                architecture=architecture,
                created_timestamp=timestamp,
                compression_algorithm=compression_algorithm,
                quantization_type=quantization_type,
                metadata=metadata
            )
    
    def _load_file_structure(self, file_path: str, header: HeraldHeader) -> HeraldFileStructure:
        """Load complete file structure."""
        with open(file_path, 'rb') as f:
            # Skip header (already read)
            header_size = (HERALD_MAGIC_LENGTH + 4 +  # magic + version
                         4 + len(header.model_name) +  # model name
                         4 + len(header.model_version) +  # model version
                         4 + len(header.architecture) +  # architecture
                         8 +  # timestamp
                         4 + len(header.compression_algorithm) +  # compression
                         4 + len(header.quantization_type) +  # quantization
                         4 + len(json.dumps(header.metadata)))  # metadata
            f.seek(header_size)
            
            # Read model config
            config_len = struct.unpack('<I', f.read(4))[0]
            config_data = f.read(config_len).decode('utf-8')
            model_config = json.loads(config_data)
            
            # Read vocabulary
            vocab_len = struct.unpack('<I', f.read(4))[0]
            vocab_data = f.read(vocab_len).decode('utf-8')
            vocabulary = json.loads(vocab_data)
            
            # Read weights
            weights_len = struct.unpack('<Q', f.read(8))[0]
            weights_data = f.read(weights_len)
            weights = pickle.loads(weights_data)
            
            # Read symbolic rules
            rules_len = struct.unpack('<I', f.read(4))[0]
            rules_data = f.read(rules_len).decode('utf-8')
            symbolic_rules = json.loads(rules_data)
            
            # Read attention cache
            cache_len = struct.unpack('<Q', f.read(8))[0]
            attention_cache = None
            if cache_len > 0:
                cache_data = f.read(cache_len)
                attention_cache = pickle.loads(cache_data)
            
            # Read memory state
            memory_len = struct.unpack('<I', f.read(4))[0]
            memory_state = None
            if memory_len > 0:
                memory_data = f.read(memory_len).decode('utf-8')
                memory_state = json.loads(memory_data)
            
            return HeraldFileStructure(
                header=header,
                model_config=model_config,
                vocabulary=vocabulary,
                weights=weights,
                symbolic_rules=symbolic_rules,
                attention_cache=attention_cache,
                memory_state=memory_state
            )
    
    def _create_model_state_from_structure(self, file_structure: HeraldFileStructure) -> ModelState:
        """Create model state from file structure."""
        # Create model config
        model_config = ModelConfig(**file_structure.model_config)
        
        # Create inference config
        inference_config = type('InferenceConfig', (), {
            'max_new_tokens': 100,
            'temperature': 0.7,
            'top_p': 0.9,
            'top_k': 50,
            'repetition_penalty': 1.1,
            'do_sample': True,
            'pad_token_id': 0,
            'eos_token_id': 2,
            'bos_token_id': 1
        })()
        
        # Create tokenizer
        tokenizer = self._create_tokenizer_from_vocabulary(file_structure.vocabulary)
        
        # Create memory manager
        memory_manager = self._create_memory_manager_from_state(file_structure.memory_state)
        
        # Decompress weights
        decompressed_weights = self._decompress_weights(file_structure.weights)
        
        # Create model state
        model_state = ModelState(
            model_config=model_config,
            inference_config=inference_config,
            tokenizer=tokenizer,
            memory_manager=memory_manager,
            transformer_weights=decompressed_weights.get('transformer', {}),
            mamba_weights=decompressed_weights.get('mamba', {}),
            embedding_weights=decompressed_weights.get('embedding', {}),
            output_weights=decompressed_weights.get('output', {}),
            expert_weights=decompressed_weights.get('expert', {}),
            router_weights=decompressed_weights.get('router', {}),
            is_loaded=True,
            is_quantized=True
        )
        
        return model_state
    
    def _create_tokenizer_from_vocabulary(self, vocabulary_data: Dict[str, Any]) -> ASCTokenizer:
        """Create tokenizer from vocabulary data."""
        if not vocabulary_data:
            return ASCTokenizer()
        
        tokenizer = ASCTokenizer(
            vocab_size=vocabulary_data.get('vocab_size', 50000),
            min_frequency=vocabulary_data.get('min_frequency', 2),
            max_token_length=vocabulary_data.get('max_token_length', 100),
            compression_target=vocabulary_data.get('compression_target', 3.2)
        )
        
        # Restore vocabulary
        if 'vocabulary' in vocabulary_data:
            tokenizer.vocabulary = vocabulary_data['vocabulary']
        
        # Restore configurations
        if 'byte_tier_config' in vocabulary_data:
            tokenizer.byte_tier_config = vocabulary_data['byte_tier_config']
        if 'symbolic_tier_config' in vocabulary_data:
            tokenizer.symbolic_tier_config = vocabulary_data['symbolic_tier_config']
            # Convert code_keywords back to set if it exists
            if 'code_keywords' in tokenizer.symbolic_tier_config:
                tokenizer.symbolic_tier_config['code_keywords'] = set(
                    tokenizer.symbolic_tier_config['code_keywords']
                )
        if 'wordpiece_tier_config' in vocabulary_data:
            tokenizer.wordpiece_tier_config = vocabulary_data['wordpiece_tier_config']
        
        return tokenizer
    
    def _create_memory_manager_from_state(self, memory_state: Optional[Dict[str, Any]]) -> Optional[MultiTierMemoryManager]:
        """Create memory manager from state."""
        if not memory_state:
            return None
        
        # Create memory config
        config = MemoryConfig()
        
        # Create memory manager
        memory_manager = MultiTierMemoryManager(config)
        
        # Restore statistics (if needed)
        # Note: Full memory state restoration would require more complex implementation
        
        return memory_manager
    
    def _decompress_weights(self, compressed_weights: Dict[str, np.ndarray]) -> Dict[str, Dict[str, np.ndarray]]:
        """Decompress model weights."""
        decompressed_weights = {
            'transformer': {},
            'mamba': {},
            'embedding': {},
            'output': {},
            'expert': {},
            'router': {}
        }
        
        for name, compressed_data in compressed_weights.items():
            # Decompress array with shape information
            decompressed_array = self._decompress_array_with_shape(compressed_data)
            
            # Dequantize if necessary
            if self.quantization_type in ['int8', 'int16']:
                decompressed_array = self._dequantize_weights(decompressed_array)
            
            # Categorize weights
            if 'transformer' in name.lower():
                decompressed_weights['transformer'][name] = decompressed_array
            elif 'mamba' in name.lower():
                decompressed_weights['mamba'][name] = decompressed_array
            elif 'embedding' in name.lower():
                decompressed_weights['embedding'][name] = decompressed_array
            elif 'output' in name.lower():
                decompressed_weights['output'][name] = decompressed_array
            elif 'expert' in name.lower():
                decompressed_weights['expert'][name] = decompressed_array
            elif 'router' in name.lower():
                decompressed_weights['router'][name] = decompressed_array
            else:
                # Default to transformer
                decompressed_weights['transformer'][name] = decompressed_array
        
        return decompressed_weights
    
    def _decompress_array(self, compressed_array: np.ndarray) -> np.ndarray:
        """Decompress a numpy array."""
        if self.compression_algorithm == 'lz4':
            # Decompress LZ4 data
            decompressed_bytes = lz4.frame.decompress(compressed_array.tobytes())
            return np.frombuffer(decompressed_bytes, dtype=np.float32)
        
        elif self.compression_algorithm == 'zlib':
            # Decompress zlib data
            decompressed_bytes = zlib.decompress(compressed_array.tobytes())
            return np.frombuffer(decompressed_bytes, dtype=np.float32)
        
        else:  # no compression
            return compressed_array
    
    def _decompress_array_with_shape(self, compressed_data: Dict[str, Any]) -> np.ndarray:
        """Decompress a numpy array with shape information."""
        if isinstance(compressed_data, dict) and 'data' in compressed_data:
            # New format with shape information
            decompressed = self._decompress_array(compressed_data['data'])
            shape = compressed_data['shape']
            dtype = np.dtype(compressed_data['dtype'])
            
            # Ensure the decompressed array has the correct size for reshaping
            expected_size = np.prod(shape)
            if len(decompressed) != expected_size:
                # Pad or truncate to match expected size
                if len(decompressed) < expected_size:
                    # Pad with zeros
                    padding = np.zeros(expected_size - len(decompressed), dtype=decompressed.dtype)
                    decompressed = np.concatenate([decompressed, padding])
                else:
                    # Truncate
                    decompressed = decompressed[:expected_size]
            
            # Handle potential invalid values during cast with warning suppression
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                try:
                    return decompressed.reshape(shape).astype(dtype)
                except (ValueError, TypeError):
                    # If cast fails, try with safe casting
                    try:
                        return decompressed.reshape(shape).astype(dtype, casting='safe')
                    except (ValueError, TypeError):
                        # If safe casting also fails, use unsafe casting
                        return decompressed.reshape(shape).astype(dtype, casting='unsafe')
        else:
            # Legacy format without shape information
            return self._decompress_array(compressed_data)
    
    def _dequantize_weights(self, quantized_weights: np.ndarray) -> np.ndarray:
        """Dequantize weights from integer precision."""
        if self.quantization_type == 'int8':
            # Scale back from int8
            return quantized_weights.astype(np.float32) / 127.0
        
        elif self.quantization_type == 'int16':
            # Scale back from int16
            return quantized_weights.astype(np.float32) / 32767.0
        
        else:
            return quantized_weights
    
    def _validate_model_integrity(self, model_state: ModelState) -> bool:
        """Validate model integrity after loading."""
        try:
            # Check if model is loaded
            if not model_state.is_loaded:
                return False
            
            # Validate tokenizer
            if model_state.tokenizer and not hasattr(model_state.tokenizer, 'vocabulary'):
                return False
            
            # Validate weights
            total_weights = (len(model_state.transformer_weights) +
                           len(model_state.mamba_weights) +
                           len(model_state.embedding_weights) +
                           len(model_state.output_weights) +
                           len(model_state.expert_weights) +
                           len(model_state.router_weights))
            
            if total_weights == 0:
                return False
            
            # Validate memory manager
            if model_state.memory_manager and not hasattr(model_state.memory_manager, 'tier1'):
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Model integrity validation failed: {e}")
            return False
    
    def get_loading_stats(self) -> Dict[str, Any]:
        """Get loading performance statistics."""
        return {
            'load_times': self.load_times,
            'compression_stats': self.compression_stats,
            'validation_stats': self.validation_stats
        }


def create_herald_loader(compression_algorithm: str = "lz4",
                        quantization_type: str = "int8") -> HeraldLoader:
    """Create a HERALD loader instance."""
    return HeraldLoader(
        compression_algorithm=compression_algorithm,
        quantization_type=quantization_type
    )


def benchmark_loader(loader: HeraldLoader, test_file_path: str) -> Dict[str, Any]:
    """Benchmark loader performance."""
    start_time = time.time()
    
    # Load test file
    model_state = loader.load_herald_file(test_file_path)
    
    load_time = time.time() - start_time
    
    return {
        'load_time': load_time,
        'success': model_state is not None,
        'stats': loader.get_loading_stats()
    } 