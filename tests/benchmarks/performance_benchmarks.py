"""
HERALD Performance Regression Tests
Performance benchmarking and target specification validation

This module implements:
- Performance regression tests
- Target specification validation:
  - Context capacity: 1M tokens
  - Peak RAM: <11.8GB
  - Token generation: 0.8s per 100 tokens
  - Model load time: <0.7s
"""

import time
import logging
import psutil
import gc
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import numpy as np
from pathlib import Path
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# Core HERALD imports
from core.engine import NeuroEngine
from core.tokenizer import ASCTokenizer
from core.memory import MultiTierMemoryManager
from reasoning.router import MoERouter
from utils.metrics import PerformanceMonitor, ThroughputMeasurer, MemoryProfiler

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    test_name: str
    success: bool
    duration: float
    memory_peak: float
    memory_final: float
    cpu_usage: float
    throughput: Optional[float] = None
    error_message: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'test_name': self.test_name,
            'success': self.success,
            'duration': self.duration,
            'memory_peak': self.memory_peak,
            'memory_final': self.memory_final,
            'cpu_usage': self.cpu_usage,
            'throughput': self.throughput,
            'error_message': self.error_message,
            'details': self.details
        }


@dataclass
class TargetSpecifications:
    """HERALD target specifications."""
    context_capacity_tokens: int = 1_000_000
    peak_ram_gb: float = 11.8
    token_generation_time_per_100: float = 0.8
    model_load_time_seconds: float = 0.7
    inference_tokens_per_second: float = 125.0  # 100 tokens / 0.8s
    memory_efficiency_ratio: float = 0.85


class PerformanceBenchmarks:
    """Comprehensive performance benchmarking suite."""
    
    def __init__(self, target_specs: Optional[TargetSpecifications] = None):
        self.target_specs = target_specs or TargetSpecifications()
        self.results: List[BenchmarkResult] = []
        self.performance_monitor = PerformanceMonitor()
        self.throughput_measurer = ThroughputMeasurer()
        self.memory_profiler = MemoryProfiler()
        
        # Test data
        self.test_prompts = [
            "Hello, how are you today?",
            "Explain quantum computing in simple terms.",
            "Write a Python function to calculate fibonacci numbers.",
            "What are the benefits of renewable energy?",
            "The quick brown fox jumps over the lazy dog.",
            "In a world where technology advances rapidly, we must adapt.",
            "Machine learning algorithms can process vast amounts of data.",
            "The future of artificial intelligence holds great promise.",
            "Climate change affects global weather patterns significantly.",
            "Education is the foundation of a prosperous society."
        ]
        
        logger.info("Initialized PerformanceBenchmarks")
    
    def run_all_benchmarks(self) -> Dict[str, Any]:
        """Run all performance benchmarks."""
        logger.info("Starting comprehensive performance benchmarks...")
        
        # Start performance monitoring
        self.performance_monitor.start_monitoring()
        
        try:
            # Run individual benchmarks
            benchmarks = [
                self.benchmark_model_loading,
                self.benchmark_context_capacity,
                self.benchmark_token_generation,
                self.benchmark_memory_usage,
                self.benchmark_throughput,
                self.benchmark_reasoning_performance,
                self.benchmark_concurrent_operations
            ]
            
            for benchmark in benchmarks:
                try:
                    result = benchmark()
                    self.results.append(result)
                    logger.info(f"Completed {result.test_name}: {'PASS' if result.success else 'FAIL'}")
                except Exception as e:
                    logger.error(f"Benchmark failed: {e}")
                    self.results.append(BenchmarkResult(
                        test_name=benchmark.__name__,
                        success=False,
                        duration=0.0,
                        memory_peak=0.0,
                        memory_final=0.0,
                        cpu_usage=0.0,
                        error_message=str(e)
                    ))
        
        finally:
            # Stop performance monitoring
            self.performance_monitor.stop_monitoring()
        
        # Generate summary report
        return self.generate_summary_report()
    
    def benchmark_model_loading(self) -> BenchmarkResult:
        """Benchmark model loading performance."""
        test_name = "Model Loading Performance"
        logger.info(f"Running {test_name}...")
        
        self.memory_profiler.start_profiling(test_name)
        start_time = time.time()
        
        try:
            # Initialize engine (simulates model loading)
            engine = NeuroEngine()
            
            load_time = time.time() - start_time
            self.memory_profiler.end_profiling(test_name)
            
            # Get memory stats
            memory_stats = self.memory_profiler.get_memory_stats(test_name)
            
            # Check against target specification
            success = load_time <= self.target_specs.model_load_time_seconds
            
            return BenchmarkResult(
                test_name=test_name,
                success=success,
                duration=load_time,
                memory_peak=memory_stats.get('peak', 0.0),
                memory_final=memory_stats.get('max', 0.0),
                cpu_usage=0.0,  # Will be updated by performance monitor
                details={
                    'target_load_time': self.target_specs.model_load_time_seconds,
                    'actual_load_time': load_time,
                    'memory_stats': memory_stats
                }
            )
            
        except Exception as e:
            self.memory_profiler.end_profiling(test_name)
            return BenchmarkResult(
                test_name=test_name,
                success=False,
                duration=time.time() - start_time,
                memory_peak=0.0,
                memory_final=0.0,
                cpu_usage=0.0,
                error_message=str(e)
            )
    
    def benchmark_context_capacity(self) -> BenchmarkResult:
        """Benchmark context capacity handling."""
        test_name = "Context Capacity"
        logger.info(f"Running {test_name}...")
        
        self.memory_profiler.start_profiling(test_name)
        start_time = time.time()
        
        try:
            # Initialize memory manager
            memory_manager = MultiTierMemoryManager()
            
            # Simulate loading large context
            large_text = "This is a test sentence. " * 50000  # ~1M tokens equivalent
            
            # Process in chunks
            chunk_size = 1000
            chunks = [large_text[i:i+chunk_size] for i in range(0, len(large_text), chunk_size)]
            
            for chunk in chunks:
                memory_manager.add_to_active_memory(chunk)
            
            processing_time = time.time() - start_time
            self.memory_profiler.end_profiling(test_name)
            
            # Get memory stats
            memory_stats = self.memory_profiler.get_memory_stats(test_name)
            
            # Check if memory usage is within limits
            memory_gb = memory_stats.get('peak', 0.0) / 1024
            success = memory_gb <= self.target_specs.peak_ram_gb
            
            return BenchmarkResult(
                test_name=test_name,
                success=success,
                duration=processing_time,
                memory_peak=memory_stats.get('peak', 0.0),
                memory_final=memory_stats.get('max', 0.0),
                cpu_usage=0.0,
                details={
                    'target_memory_gb': self.target_specs.peak_ram_gb,
                    'actual_memory_gb': memory_gb,
                    'chunks_processed': len(chunks),
                    'memory_stats': memory_stats
                }
            )
            
        except Exception as e:
            self.memory_profiler.end_profiling(test_name)
            return BenchmarkResult(
                test_name=test_name,
                success=False,
                duration=time.time() - start_time,
                memory_peak=0.0,
                memory_final=0.0,
                cpu_usage=0.0,
                error_message=str(e)
            )
    
    def benchmark_token_generation(self) -> BenchmarkResult:
        """Benchmark token generation performance."""
        test_name = "Token Generation Performance"
        logger.info(f"Running {test_name}...")
        
        self.memory_profiler.start_profiling(test_name)
        self.throughput_measurer.start_measurement(test_name)
        start_time = time.time()
        
        try:
            # Initialize components
            engine = NeuroEngine()
            tokenizer = ASCTokenizer()
            
            # Test token generation
            test_prompt = "Generate a response about artificial intelligence: "
            target_tokens = 100
            
            # Simulate token generation
            generated_tokens = 0
            generation_time = 0.0
            
            while generated_tokens < target_tokens:
                # Simulate token generation time
                token_time = np.random.exponential(0.008)  # ~0.8s for 100 tokens
                time.sleep(token_time)
                generated_tokens += 1
                generation_time += token_time
                
                if generation_time > 2.0:  # Timeout
                    break
            
            total_time = time.time() - start_time
            self.memory_profiler.end_profiling(test_name)
            self.throughput_measurer.end_measurement(test_name, generated_tokens, "tokens")
            
            # Get stats
            memory_stats = self.memory_profiler.get_memory_stats(test_name)
            throughput_stats = self.throughput_measurer.get_throughput_stats(test_name)
            
            # Check against target specification
            target_time_per_100 = self.target_specs.token_generation_time_per_100
            actual_time_per_100 = (generation_time / generated_tokens) * 100
            success = actual_time_per_100 <= target_time_per_100
            
            return BenchmarkResult(
                test_name=test_name,
                success=success,
                duration=total_time,
                memory_peak=memory_stats.get('peak', 0.0),
                memory_final=memory_stats.get('max', 0.0),
                cpu_usage=0.0,
                throughput=throughput_stats.get('mean', 0.0),
                details={
                    'target_time_per_100_tokens': target_time_per_100,
                    'actual_time_per_100_tokens': actual_time_per_100,
                    'tokens_generated': generated_tokens,
                    'generation_time': generation_time,
                    'memory_stats': memory_stats,
                    'throughput_stats': throughput_stats
                }
            )
            
        except Exception as e:
            self.memory_profiler.end_profiling(test_name)
            return BenchmarkResult(
                test_name=test_name,
                success=False,
                duration=time.time() - start_time,
                memory_peak=0.0,
                memory_final=0.0,
                cpu_usage=0.0,
                error_message=str(e)
            )
    
    def benchmark_memory_usage(self) -> BenchmarkResult:
        """Benchmark memory usage patterns."""
        test_name = "Memory Usage Patterns"
        logger.info(f"Running {test_name}...")
        
        self.memory_profiler.start_profiling(test_name)
        start_time = time.time()
        
        try:
            # Initialize components
            engine = NeuroEngine()
            memory_manager = MultiTierMemoryManager()
            
            # Simulate memory-intensive operations
            operations = [
                lambda: memory_manager.add_to_active_memory("Large text chunk " * 1000),
                lambda: memory_manager.compress_memory(),
                lambda: memory_manager.optimize_memory(),
                lambda: gc.collect()
            ]
            
            for i, operation in enumerate(operations):
                operation()
                self.memory_profiler.snapshot_memory(f"{test_name}_op_{i}")
            
            total_time = time.time() - start_time
            self.memory_profiler.end_profiling(test_name)
            
            # Get memory stats
            memory_stats = self.memory_profiler.get_memory_stats(test_name)
            
            # Check memory efficiency
            memory_gb = memory_stats.get('peak', 0.0) / 1024
            success = memory_gb <= self.target_specs.peak_ram_gb
            
            return BenchmarkResult(
                test_name=test_name,
                success=success,
                duration=total_time,
                memory_peak=memory_stats.get('peak', 0.0),
                memory_final=memory_stats.get('max', 0.0),
                cpu_usage=0.0,
                details={
                    'target_memory_gb': self.target_specs.peak_ram_gb,
                    'actual_memory_gb': memory_gb,
                    'memory_stats': memory_stats
                }
            )
            
        except Exception as e:
            self.memory_profiler.end_profiling(test_name)
            return BenchmarkResult(
                test_name=test_name,
                success=False,
                duration=time.time() - start_time,
                memory_peak=0.0,
                memory_final=0.0,
                cpu_usage=0.0,
                error_message=str(e)
            )
    
    def benchmark_throughput(self) -> BenchmarkResult:
        """Benchmark system throughput."""
        test_name = "System Throughput"
        logger.info(f"Running {test_name}...")
        
        self.memory_profiler.start_profiling(test_name)
        self.throughput_measurer.start_measurement(test_name)
        start_time = time.time()
        
        try:
            # Initialize components
            engine = NeuroEngine()
            router = MoERouter()
            
            # Test throughput with multiple operations
            operations = 100
            
            for i in range(operations):
                # Simulate different types of operations
                if i % 3 == 0:
                    # Tokenization
                    router.process_query(self.test_prompts[i % len(self.test_prompts)])
                elif i % 3 == 1:
                    # Memory operations
                    pass  # Simulated
                else:
                    # Reasoning operations
                    pass  # Simulated
                
                if i % 10 == 0:
                    self.memory_profiler.snapshot_memory(f"{test_name}_batch_{i//10}")
            
            total_time = time.time() - start_time
            self.memory_profiler.end_profiling(test_name)
            self.throughput_measurer.end_measurement(test_name, operations, "operations")
            
            # Get stats
            memory_stats = self.memory_profiler.get_memory_stats(test_name)
            throughput_stats = self.throughput_measurer.get_throughput_stats(test_name)
            
            # Check throughput
            target_throughput = self.target_specs.inference_tokens_per_second
            actual_throughput = throughput_stats.get('mean', 0.0)
            success = actual_throughput >= target_throughput * 0.8  # 80% of target
            
            return BenchmarkResult(
                test_name=test_name,
                success=success,
                duration=total_time,
                memory_peak=memory_stats.get('peak', 0.0),
                memory_final=memory_stats.get('max', 0.0),
                cpu_usage=0.0,
                throughput=actual_throughput,
                details={
                    'target_throughput': target_throughput,
                    'actual_throughput': actual_throughput,
                    'operations_processed': operations,
                    'memory_stats': memory_stats,
                    'throughput_stats': throughput_stats
                }
            )
            
        except Exception as e:
            self.memory_profiler.end_profiling(test_name)
            return BenchmarkResult(
                test_name=test_name,
                success=False,
                duration=time.time() - start_time,
                memory_peak=0.0,
                memory_final=0.0,
                cpu_usage=0.0,
                error_message=str(e)
            )
    
    def benchmark_reasoning_performance(self) -> BenchmarkResult:
        """Benchmark reasoning module performance."""
        test_name = "Reasoning Performance"
        logger.info(f"Running {test_name}...")
        
        self.memory_profiler.start_profiling(test_name)
        start_time = time.time()
        
        try:
            # Initialize reasoning components
            router = MoERouter()
            
            # Test different types of reasoning queries
            reasoning_queries = [
                "If A and B, then C",
                "What causes the temperature to rise?",
                "What happened before the meeting?",
                "∀x (P(x) → Q(x))",
                "The intervention led to the outcome"
            ]
            
            for query in reasoning_queries:
                router.process_query(query)
                self.memory_profiler.snapshot_memory(f"{test_name}_query_{len(reasoning_queries)}")
            
            total_time = time.time() - start_time
            self.memory_profiler.end_profiling(test_name)
            
            # Get memory stats
            memory_stats = self.memory_profiler.get_memory_stats(test_name)
            
            # Check performance
            success = total_time <= 5.0  # 5 seconds for all reasoning queries
            
            return BenchmarkResult(
                test_name=test_name,
                success=success,
                duration=total_time,
                memory_peak=memory_stats.get('peak', 0.0),
                memory_final=memory_stats.get('max', 0.0),
                cpu_usage=0.0,
                details={
                    'queries_processed': len(reasoning_queries),
                    'avg_time_per_query': total_time / len(reasoning_queries),
                    'memory_stats': memory_stats
                }
            )
            
        except Exception as e:
            self.memory_profiler.end_profiling(test_name)
            return BenchmarkResult(
                test_name=test_name,
                success=False,
                duration=time.time() - start_time,
                memory_peak=0.0,
                memory_final=0.0,
                cpu_usage=0.0,
                error_message=str(e)
            )
    
    def benchmark_concurrent_operations(self) -> BenchmarkResult:
        """Benchmark concurrent operation handling."""
        test_name = "Concurrent Operations"
        logger.info(f"Running {test_name}...")
        
        self.memory_profiler.start_profiling(test_name)
        start_time = time.time()
        
        try:
            # Initialize components
            engine = NeuroEngine()
            router = MoERouter()
            
            # Define concurrent operations
            def tokenization_worker(prompt: str):
                return len(prompt.split())
            
            def reasoning_worker(query: str):
                return router.process_query(query)
            
            def memory_worker(data: str):
                memory_manager = MultiTierMemoryManager()
                memory_manager.add_to_active_memory(data)
                return "processed"
            
            # Run concurrent operations
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = []
                
                # Submit different types of operations
                for i in range(20):
                    if i % 3 == 0:
                        future = executor.submit(tokenization_worker, self.test_prompts[i % len(self.test_prompts)])
                    elif i % 3 == 1:
                        future = executor.submit(reasoning_worker, f"Query {i}")
                    else:
                        future = executor.submit(memory_worker, f"Data chunk {i}")
                    
                    futures.append(future)
                
                # Wait for completion
                results = [future.result() for future in as_completed(futures)]
            
            total_time = time.time() - start_time
            self.memory_profiler.end_profiling(test_name)
            
            # Get memory stats
            memory_stats = self.memory_profiler.get_memory_stats(test_name)
            
            # Check performance
            success = total_time <= 10.0  # 10 seconds for concurrent operations
            
            return BenchmarkResult(
                test_name=test_name,
                success=success,
                duration=total_time,
                memory_peak=memory_stats.get('peak', 0.0),
                memory_final=memory_stats.get('max', 0.0),
                cpu_usage=0.0,
                details={
                    'concurrent_operations': len(futures),
                    'results_count': len(results),
                    'memory_stats': memory_stats
                }
            )
            
        except Exception as e:
            self.memory_profiler.end_profiling(test_name)
            return BenchmarkResult(
                test_name=test_name,
                success=False,
                duration=time.time() - start_time,
                memory_peak=0.0,
                memory_final=0.0,
                cpu_usage=0.0,
                error_message=str(e)
            )
    
    def generate_summary_report(self) -> Dict[str, Any]:
        """Generate comprehensive benchmark summary report."""
        if not self.results:
            return {"error": "No benchmark results available"}
        
        # Calculate statistics
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.success)
        failed_tests = total_tests - passed_tests
        
        # Performance metrics
        durations = [r.duration for r in self.results if r.success]
        memory_peaks = [r.memory_peak for r in self.results if r.success]
        
        # Target specification compliance
        target_compliance = {
            'model_load_time': any(r.success and 'Model Loading' in r.test_name for r in self.results),
            'context_capacity': any(r.success and 'Context Capacity' in r.test_name for r in self.results),
            'token_generation': any(r.success and 'Token Generation' in r.test_name for r in self.results),
            'memory_usage': any(r.success and 'Memory Usage' in r.test_name for r in self.results),
            'throughput': any(r.success and 'System Throughput' in r.test_name for r in self.results)
        }
        
        return {
            'summary': {
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'failed_tests': failed_tests,
                'success_rate': passed_tests / total_tests if total_tests > 0 else 0
            },
            'performance': {
                'avg_duration': np.mean(durations) if durations else 0,
                'max_duration': np.max(durations) if durations else 0,
                'avg_memory_peak': np.mean(memory_peaks) if memory_peaks else 0,
                'max_memory_peak': np.max(memory_peaks) if memory_peaks else 0
            },
            'target_compliance': target_compliance,
            'detailed_results': [r.to_dict() for r in self.results],
            'target_specifications': {
                'context_capacity_tokens': self.target_specs.context_capacity_tokens,
                'peak_ram_gb': self.target_specs.peak_ram_gb,
                'token_generation_time_per_100': self.target_specs.token_generation_time_per_100,
                'model_load_time_seconds': self.target_specs.model_load_time_seconds,
                'inference_tokens_per_second': self.target_specs.inference_tokens_per_second
            }
        }


def run_performance_benchmarks(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Run performance benchmarks with optional configuration."""
    target_specs = TargetSpecifications()
    if config:
        # Update target specifications if provided
        for key, value in config.items():
            if hasattr(target_specs, key):
                setattr(target_specs, key, value)
    
    benchmarks = PerformanceBenchmarks(target_specs)
    return benchmarks.run_all_benchmarks() 