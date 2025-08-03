"""
ASC Tokenizer (Adaptive Symbolic Compression)
HERALD v1.0 - Core Tokenization Module

This module implements a three-tier compression strategy for efficient tokenization:
- Tier 1: Byte-level processing with UTF-8 analysis
- Tier 2: Symbolic tokenization for math/code
- Tier 3: Wordpiece integration for natural language

Target compression ratio: 3.2:1 over standard tokenization
"""

import re
import json
import hashlib
from typing import List, Dict, Tuple, Optional, Union, Set
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import numpy as np
from numba import jit, prange
import lz4.frame


@dataclass
class Token:
    """Represents a single token in the ASC tokenizer."""
    text: str
    token_id: int
    tier: int  # 1=byte, 2=symbolic, 3=wordpiece
    frequency: int = 0
    is_special: bool = False
    metadata: Dict = field(default_factory=dict)


@dataclass
class TokenizationResult:
    """Result of tokenization process."""
    tokens: List[Token]
    vocabulary: Dict[str, int]
    compression_ratio: float
    tier_distribution: Dict[int, int]
    metadata: Dict = field(default_factory=dict)


class ASCTokenizer:
    """
    Adaptive Symbolic Compression Tokenizer
    
    Implements a three-tier tokenization strategy for optimal compression
    and processing efficiency on CPU-constrained systems.
    """
    
    def __init__(self, 
                 vocab_size: int = 50000,
                 min_frequency: int = 2,
                 max_token_length: int = 100,
                 compression_target: float = 3.2):
        """
        Initialize the ASC Tokenizer.
        
        Args:
            vocab_size: Maximum vocabulary size
            min_frequency: Minimum token frequency for inclusion
            max_token_length: Maximum token length in characters
            compression_target: Target compression ratio
        """
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.max_token_length = max_token_length
        self.compression_target = compression_target
        
        # Tier-specific configurations
        self.byte_tier_config = {
            'max_byte_sequence': 8,
            'frequency_threshold': 5,
            'compression_ratio': 2.5
        }
        
        self.symbolic_tier_config = {
            'math_patterns': [
                r'[+\-*/=<>!&|^~%]',  # Mathematical operators
                r'\d+\.?\d*',          # Numbers
                r'[a-zA-Z_]\w*',       # Identifiers
                r'[{}[\]()]',          # Brackets
                r'[;:,.]',             # Punctuation
            ],
            'code_keywords': {
                'def', 'class', 'if', 'else', 'for', 'while', 'return',
                'import', 'from', 'as', 'in', 'is', 'not', 'and', 'or',
                'try', 'except', 'finally', 'with', 'lambda', 'yield'
            },
            'compression_ratio': 3.5
        }
        
        self.wordpiece_tier_config = {
            'subword_prefix': '##',
            'max_subword_length': 4,
            'compression_ratio': 3.0
        }
        
        # Initialize vocabulary and statistics
        self.vocabulary: Dict[str, int] = {}
        self.reverse_vocabulary: Dict[int, str] = {}
        self.token_frequencies: Dict[str, int] = defaultdict(int)
        self.tier_statistics: Dict[int, Dict] = defaultdict(dict)
        
        # Special tokens
        self.special_tokens = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<BOS>': 2,
            '<EOS>': 3,
            '<SEP>': 4,
            '<MASK>': 5
        }
        
        # Initialize with special tokens
        self.vocabulary.update(self.special_tokens)
        self.reverse_vocabulary.update({v: k for k, v in self.special_tokens.items()})
        
        # Performance tracking
        self.compression_history: List[float] = []
        self.processing_times: Dict[str, float] = {}
        
    def _analyze_utf8_byte_patterns(self, text: str) -> Dict[bytes, int]:
        """
        Tier 1: Analyze UTF-8 byte patterns for frequency mapping.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary mapping byte sequences to frequencies
        """
        byte_patterns = defaultdict(int)
        
        # Convert text to UTF-8 bytes
        text_bytes = text.encode('utf-8')
        
        # Analyze byte sequences up to max_byte_sequence length
        max_seq = self.byte_tier_config['max_byte_sequence']
        
        for i in range(len(text_bytes)):
            for seq_len in range(1, min(max_seq + 1, len(text_bytes) - i + 1)):
                sequence = text_bytes[i:i + seq_len]
                byte_patterns[sequence] += 1
                
        return dict(byte_patterns)
    
    def _extract_symbolic_patterns(self, text: str) -> Dict[str, int]:
        """
        Tier 2: Extract symbolic patterns for mathematical and code content.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary mapping symbolic patterns to frequencies
        """
        symbolic_patterns = defaultdict(int)
        
        # Extract mathematical patterns
        for pattern in self.symbolic_tier_config['math_patterns']:
            matches = re.finditer(pattern, text)
            for match in matches:
                pattern_text = match.group()
                if len(pattern_text) <= self.max_token_length:
                    symbolic_patterns[pattern_text] += 1
        
        # Extract code keywords
        words = re.findall(r'\b\w+\b', text)
        for word in words:
            if word.lower() in self.symbolic_tier_config['code_keywords']:
                symbolic_patterns[word] += 1
        
        # Extract common code constructs
        code_constructs = [
            r'def\s+\w+\s*\([^)]*\):',  # Function definitions
            r'class\s+\w+',              # Class definitions
            r'if\s+[^:]+:',              # If statements
            r'for\s+\w+\s+in\s+[^:]+:', # For loops
            r'while\s+[^:]+:',           # While loops
        ]
        
        for construct in code_constructs:
            matches = re.finditer(construct, text)
            for match in matches:
                construct_text = match.group()
                if len(construct_text) <= self.max_token_length:
                    symbolic_patterns[construct_text] += 1
        
        return dict(symbolic_patterns)
    
    def _apply_wordpiece_tokenization(self, text: str) -> List[str]:
        """
        Tier 3: Apply WordPiece tokenization for natural language.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            List of WordPiece tokens
        """
        # Simple WordPiece implementation
        # In production, this would use a more sophisticated algorithm
        
        tokens = []
        # Split by whitespace and punctuation
        words = re.findall(r'\b\w+\b', text)
        
        for word in words:
            if len(word) <= self.wordpiece_tier_config['max_subword_length']:
                tokens.append(word)
            else:
                # Split long words into subwords
                subwords = self._split_word_into_subwords(word)
                tokens.extend(subwords)
        
        return tokens
    
    def _split_word_into_subwords(self, word: str) -> List[str]:
        """
        Split a word into subwords using greedy algorithm.
        
        Args:
            word: Word to split
            
        Returns:
            List of subwords
        """
        subwords = []
        start = 0
        
        while start < len(word):
            end = len(word)
            cur_substr = None
            
            while start < end:
                substr = word[start:end]
                if substr in self.vocabulary or end == start + 1:
                    cur_substr = substr
                    break
                end -= 1
            
            if cur_substr is None:
                cur_substr = word[start:start + 1]
            
            subwords.append(cur_substr)
            start += len(cur_substr)
        
        return subwords
    
    @jit(nopython=True, parallel=True)
    def _optimized_byte_analysis(self, text_bytes: np.ndarray) -> np.ndarray:
        """
        Numba-optimized byte pattern analysis.
        
        Args:
            text_bytes: Text as numpy byte array
            
        Returns:
            Frequency array for byte patterns
        """
        # This is a simplified version for demonstration
        # In production, this would implement the full byte analysis
        return np.zeros(len(text_bytes), dtype=np.int32)
    
    def _build_dynamic_vocabulary(self, texts: List[str]) -> None:
        """
        Build dynamic vocabulary based on input corpus characteristics.
        
        Args:
            texts: List of input texts for vocabulary construction
        """
        # Collect all patterns from all tiers
        all_patterns = defaultdict(int)
        
        for text in texts:
            # Tier 1: Byte patterns
            byte_patterns = self._analyze_utf8_byte_patterns(text)
            for pattern, freq in byte_patterns.items():
                if freq >= self.byte_tier_config['frequency_threshold']:
                    all_patterns[str(pattern)] += freq
            
            # Tier 2: Symbolic patterns
            symbolic_patterns = self._extract_symbolic_patterns(text)
            for pattern, freq in symbolic_patterns.items():
                all_patterns[pattern] += freq
            
            # Tier 3: WordPiece patterns
            wordpiece_tokens = self._apply_wordpiece_tokenization(text)
            for token in wordpiece_tokens:
                all_patterns[token] += 1
        
        # Sort by frequency and add to vocabulary
        sorted_patterns = sorted(all_patterns.items(), 
                               key=lambda x: x[1], reverse=True)
        
        vocab_id = len(self.vocabulary)
        for pattern, frequency in sorted_patterns:
            if (vocab_id < self.vocab_size and 
                frequency >= self.min_frequency and
                len(pattern) <= self.max_token_length):
                
                self.vocabulary[pattern] = vocab_id
                self.reverse_vocabulary[vocab_id] = pattern
                self.token_frequencies[pattern] = frequency
                vocab_id += 1
    
    def _determine_token_tier(self, token: str) -> int:
        """
        Determine which tier a token belongs to.
        
        Args:
            token: Token to classify
            
        Returns:
            Tier number (1=byte, 2=symbolic, 3=wordpiece)
        """
        # Check if it's a byte pattern (starts with b' and ends with ')
        if token.startswith("b'") and token.endswith("'"):
            return 1
        
        # Check if it's a symbolic pattern
        for pattern in self.symbolic_tier_config['math_patterns']:
            if re.match(pattern, token):
                return 2
        
        # Check if it's a code keyword
        if token.lower() in self.symbolic_tier_config['code_keywords']:
            return 2
        
        # Default to wordpiece tier
        return 3
    
    def tokenize(self, text: str) -> TokenizationResult:
        """
        Main tokenization method implementing the three-tier strategy.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            TokenizationResult with tokens and metadata
        """
        import time
        start_time = time.time()
        
        tokens = []
        tier_distribution = {1: 0, 2: 0, 3: 0}
        
        # Tier 1: Byte-level processing
        byte_patterns = self._analyze_utf8_byte_patterns(text)
        for pattern, freq in byte_patterns.items():
            pattern_str = str(pattern)
            if pattern_str in self.vocabulary:
                token = Token(
                    text=pattern_str,
                    token_id=self.vocabulary[pattern_str],
                    tier=1,
                    frequency=freq
                )
                tokens.append(token)
                tier_distribution[1] += 1
        
        # Tier 2: Symbolic tokenization
        symbolic_patterns = self._extract_symbolic_patterns(text)
        for pattern, freq in symbolic_patterns.items():
            if pattern in self.vocabulary:
                token = Token(
                    text=pattern,
                    token_id=self.vocabulary[pattern],
                    tier=2,
                    frequency=freq
                )
                tokens.append(token)
                tier_distribution[2] += 1
        
        # Tier 3: WordPiece integration
        wordpiece_tokens = self._apply_wordpiece_tokenization(text)
        for token_text in wordpiece_tokens:
            if token_text in self.vocabulary:
                token = Token(
                    text=token_text,
                    token_id=self.vocabulary[token_text],
                    tier=3,
                    frequency=1
                )
                tokens.append(token)
                tier_distribution[3] += 1
        
        # Calculate compression ratio - use actual token count vs original bytes
        original_size = len(text.encode('utf-8'))
        tokenized_size = len(tokens) * 4  # Assuming 4 bytes per token ID
        compression_ratio = original_size / max(tokenized_size, 1)  # Avoid division by zero
        
        # Record processing time
        processing_time = time.time() - start_time
        self.processing_times['tokenize'] = processing_time
        
        return TokenizationResult(
            tokens=tokens,
            vocabulary=self.vocabulary,
            compression_ratio=compression_ratio,
            tier_distribution=tier_distribution,
            metadata={
                'processing_time': processing_time,
                'original_size': original_size,
                'tokenized_size': tokenized_size
            }
        )
    
    def detokenize(self, tokens: List[Token]) -> str:
        """
        Convert tokens back to text.
        
        Args:
            tokens: List of tokens to detokenize
            
        Returns:
            Reconstructed text
        """
        text_parts = []
        
        for token in tokens:
            if token.token_id in self.reverse_vocabulary:
                text_parts.append(self.reverse_vocabulary[token.token_id])
            else:
                text_parts.append('<UNK>')
        
        return ''.join(text_parts)
    
    def train(self, texts: List[str]) -> None:
        """
        Train the tokenizer on a corpus of texts.
        
        Args:
            texts: List of training texts
        """
        print(f"Training ASC Tokenizer on {len(texts)} texts...")
        
        # Build dynamic vocabulary
        self._build_dynamic_vocabulary(texts)
        
        # Calculate tier statistics
        for text in texts[:100]:  # Sample for statistics
            result = self.tokenize(text)
            for tier, count in result.tier_distribution.items():
                self.tier_statistics[tier]['total_tokens'] = \
                    self.tier_statistics[tier].get('total_tokens', 0) + count
        
        print(f"Vocabulary size: {len(self.vocabulary)}")
        print(f"Tier distribution: {dict(self.tier_statistics)}")
    
    def save_vocabulary(self, filepath: str) -> None:
        """
        Save vocabulary to file.
        
        Args:
            filepath: Path to save vocabulary
        """
        vocab_data = {
            'vocabulary': {str(k): v for k, v in self.vocabulary.items()},
            'reverse_vocabulary': {str(k): v for k, v in self.reverse_vocabulary.items()},
            'token_frequencies': {str(k): v for k, v in self.token_frequencies.items()},
            'tier_statistics': {str(k): v for k, v in self.tier_statistics.items()},
            'config': {
                'vocab_size': self.vocab_size,
                'min_frequency': self.min_frequency,
                'max_token_length': self.max_token_length,
                'compression_target': self.compression_target
            }
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, indent=2, ensure_ascii=False)
    
    def load_vocabulary(self, filepath: str) -> None:
        """
        Load vocabulary from file.
        
        Args:
            filepath: Path to vocabulary file
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        
        # Convert string keys back to appropriate types
        self.vocabulary = {k: int(v) for k, v in vocab_data['vocabulary'].items()}
        self.reverse_vocabulary = {int(k): v for k, v in vocab_data['reverse_vocabulary'].items()}
        self.token_frequencies = defaultdict(int, {k: int(v) for k, v in vocab_data['token_frequencies'].items()})
        self.tier_statistics = defaultdict(dict, {int(k): v for k, v in vocab_data['tier_statistics'].items()})
        
        # Restore configuration
        config = vocab_data['config']
        self.vocab_size = config['vocab_size']
        self.min_frequency = config['min_frequency']
        self.max_token_length = config['max_token_length']
        self.compression_target = config['compression_target']
    
    def get_compression_stats(self) -> Dict:
        """
        Get compression statistics.
        
        Returns:
            Dictionary with compression statistics
        """
        return {
            'average_compression_ratio': np.mean(self.compression_history) if self.compression_history else 0,
            'target_compression_ratio': self.compression_target,
            'tier_distribution': dict(self.tier_statistics),
            'vocabulary_size': len(self.vocabulary),
            'processing_times': self.processing_times
        }
    
    def optimize_for_cpu(self) -> None:
        """
        Apply CPU-specific optimizations.
        """
        # Pre-compile frequently used functions with Numba
        # This would be done automatically in production
        
        # Optimize memory usage
        self.vocabulary = dict(self.vocabulary)  # Convert from defaultdict
        self.token_frequencies = dict(self.token_frequencies)
        
        print("CPU optimizations applied to ASC Tokenizer")


# Utility functions for testing and validation

def create_test_corpus() -> List[str]:
    """Create a test corpus for tokenizer training."""
    return [
        "Hello world! This is a test of the ASC tokenizer.",
        "def calculate_sum(a, b): return a + b",
        "The quick brown fox jumps over the lazy dog.",
        "2 + 2 = 4 and 3 * 5 = 15",
        "import numpy as np; x = np.array([1, 2, 3])",
        "if x > 0: print('Positive') else: print('Negative')",
        "class MyClass: def __init__(self): self.value = 0",
        "for i in range(10): print(i)",
        "while condition: do_something()",
        "try: risky_operation() except: handle_error()"
    ]


def benchmark_tokenizer(tokenizer: ASCTokenizer, test_texts: List[str]) -> Dict:
    """
    Benchmark tokenizer performance.
    
    Args:
        tokenizer: ASC Tokenizer instance
        test_texts: List of test texts
        
    Returns:
        Dictionary with benchmark results
    """
    import time
    
    results = {
        'total_texts': len(test_texts),
        'total_tokens': 0,
        'average_compression_ratio': 0.0,
        'processing_times': [],
        'tier_distribution': {1: 0, 2: 0, 3: 0}
    }
    
    start_time = time.perf_counter()
    
    for text in test_texts:
        text_start = time.perf_counter()
        result = tokenizer.tokenize(text)
        text_time = time.perf_counter() - text_start
        
        # Ensure minimum measurable time
        if text_time < 0.0001:
            text_time = 0.0001
        
        results['total_tokens'] += len(result.tokens)
        results['processing_times'].append(text_time)
        results['average_compression_ratio'] += result.compression_ratio
        
        for tier, count in result.tier_distribution.items():
            results['tier_distribution'][tier] += count
    
    total_time = time.perf_counter() - start_time
    
    results['average_compression_ratio'] /= len(test_texts)
    results['total_time'] = total_time
    results['average_time_per_text'] = total_time / len(test_texts)
    
    return results


if __name__ == "__main__":
    # Example usage and testing
    print("Initializing ASC Tokenizer...")
    
    # Create tokenizer
    tokenizer = ASCTokenizer(
        vocab_size=10000,
        min_frequency=1,
        compression_target=3.2
    )
    
    # Create test corpus
    test_corpus = create_test_corpus()
    
    # Train tokenizer
    tokenizer.train(test_corpus)
    
    # Test tokenization
    test_text = "def hello_world(): print('Hello, World!')"
    result = tokenizer.tokenize(test_text)
    
    print(f"Tokenization result:")
    print(f"  Tokens: {len(result.tokens)}")
    print(f"  Compression ratio: {result.compression_ratio:.2f}")
    print(f"  Tier distribution: {result.tier_distribution}")
    
    # Benchmark
    benchmark_results = benchmark_tokenizer(tokenizer, test_corpus)
    print(f"\nBenchmark results:")
    print(f"  Total tokens: {benchmark_results['total_tokens']}")
    print(f"  Average compression ratio: {benchmark_results['average_compression_ratio']:.2f}")
    print(f"  Total time: {benchmark_results['total_time']:.4f}s")
    print(f"  Average time per text: {benchmark_results['average_time_per_text']:.4f}s")
    
    # Save vocabulary
    tokenizer.save_vocabulary("vocabulary.json")
    print("\nVocabulary saved to vocabulary.json") 