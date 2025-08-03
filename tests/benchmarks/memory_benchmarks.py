"""
HERALD Memory Benchmarking
Memory usage profiling and optimization testing

This module implements:
- Memory usage profiling
- Memory efficiency testing
- Memory optimization benchmarks
- Memory leak detection
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
from core.memory import MultiTierMemoryManager
from utils.metrics import MemoryProfiler, PerformanceMonitor

logger = logging.getLogger(__name__)


@dataclass
class MemoryBenchmarkResult:
    """Container for memory benchmark results."""
    benchmark_name: str
    success: bool
    duration: float
    initial_memory: float
    peak_memory: float
    final_memory: float
    memory_efficiency: float
    memory_leak_detected: bool
    error_message: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'benchmark_name': self.benchmark_name,
            'success': self.success,
            'duration': self.duration,
            'initial_memory': self.initial_memory,
            'peak_memory': self.peak_memory,
            'final_memory': self.final_memory,
            'memory_efficiency': self.memory_efficiency,
            'memory_leak_detected': self.memory_leak_detected,
            'error_message': self.error_message,
            'details': self.details
        }


class MemoryBenchmarks:
    """Memory-specific benchmarking suite."""
    
    def __init__(self):
        self.results: List[MemoryBenchmarkResult] = []
        self.memory_profiler = MemoryProfiler()
        self.performance_monitor = PerformanceMonitor()
        
        # Test data sizes
        self.test_sizes = {
            'small': 1024,      # 1KB
            'medium': 10240,     # 10KB
            'large': 102400,     # 100KB
            'xlarge': 1024000    # 1MB
        }
        
        logger.info("Initialized MemoryBenchmarks")
    
    def benchmark_memory_allocation(self, size_category: str = "medium") -> MemoryBenchmarkResult:
        """Benchmark memory allocation patterns."""
        benchmark_name = f"Memory Allocation ({size_category})"
        logger.info(f"Running {benchmark_name}...")
        
        self.memory_profiler.start_profiling(benchmark_name)
        start_time = time.time()
        
        try:
            # Get test size
            test_size = self.test_sizes.get(size_category, self.test_sizes['medium'])
            
            # Initialize memory manager
            memory_manager = MultiTierMemoryManager()
            
            # Simulate memory allocation
            test_data = "Test data chunk " * (test_size // 20)  # Approximate size
            
            # Allocate memory in chunks
            chunk_size = test_size // 10
            chunks = [test_data[i:i+chunk_size] for i in range(0, len(test_data), chunk_size)]
            
            for i, chunk in enumerate(chunks):
                memory_manager.add_to_active_memory(chunk)
                self.memory_profiler.snapshot_memory(f"{benchmark_name}_chunk_{i}")
            
            total_time = time.time() - start_time
            self.memory_profiler.end_profiling(benchmark_name)
            
            # Get memory stats
            memory_stats = self.memory_profiler.get_memory_stats(benchmark_name)
            
            # Calculate efficiency
            initial_memory = memory_stats.get('min', 0.0)
            peak_memory = memory_stats.get('peak', 0.0)
            final_memory = memory_stats.get('max', 0.0)
            
            memory_efficiency = final_memory / peak_memory if peak_memory > 0 else 1.0
            memory_leak = (final_memory - initial_memory) > (peak_memory * 0.1)  # 10% threshold
            
            return MemoryBenchmarkResult(
                benchmark_name=benchmark_name,
                success=True,
                duration=total_time,
                initial_memory=initial_memory,
                peak_memory=peak_memory,
                final_memory=final_memory,
                memory_efficiency=memory_efficiency,
                memory_leak_detected=memory_leak,
                details={
                    'test_size': test_size,
                    'chunks_allocated': len(chunks),
                    'memory_stats': memory_stats
                }
            )
            
        except Exception as e:
            self.memory_profiler.end_profiling(benchmark_name)
            return MemoryBenchmarkResult(
                benchmark_name=benchmark_name,
                success=False,
                duration=time.time() - start_time,
                initial_memory=0.0,
                peak_memory=0.0,
                final_memory=0.0,
                memory_efficiency=0.0,
                memory_leak_detected=False,
                error_message=str(e)
            )
    
    def benchmark_memory_compression(self) -> MemoryBenchmarkResult:
        """Benchmark memory compression efficiency."""
        benchmark_name = "Memory Compression"
        logger.info(f"Running {benchmark_name}...")
        
        self.memory_profiler.start_profiling(benchmark_name)
        start_time = time.time()
        
        try:
            # Initialize memory manager
            memory_manager = MultiTierMemoryManager()
            
            # Load data into active memory
            test_data = "Large data chunk for compression testing " * 1000
            memory_manager.add_to_active_memory(test_data)
            
            # Record memory before compression
            self.memory_profiler.snapshot_memory(f"{benchmark_name}_before")
            
            # Perform compression
            compression_start = time.time()
            memory_manager.compress_memory()
            compression_time = time.time() - compression_start
            
            # Record memory after compression
            self.memory_profiler.snapshot_memory(f"{benchmark_name}_after")
            
            total_time = time.time() - start_time
            self.memory_profiler.end_profiling(benchmark_name)
            
            # Get memory stats
            memory_stats = self.memory_profiler.get_memory_stats(benchmark_name)
            
            # Calculate compression efficiency
            before_compression = memory_stats.get('min', 0.0)
            after_compression = memory_stats.get('max', 0.0)
            
            compression_ratio = before_compression / after_compression if after_compression > 0 else 1.0
            memory_efficiency = 1.0 / compression_ratio if compression_ratio > 0 else 1.0
            
            return MemoryBenchmarkResult(
                benchmark_name=benchmark_name,
                success=True,
                duration=total_time,
                initial_memory=before_compression,
                peak_memory=memory_stats.get('peak', 0.0),
                final_memory=after_compression,
                memory_efficiency=memory_efficiency,
                memory_leak_detected=False,
                details={
                    'compression_time': compression_time,
                    'compression_ratio': compression_ratio,
                    'memory_stats': memory_stats
                }
            )
            
        except Exception as e:
            self.memory_profiler.end_profiling(benchmark_name)
            return MemoryBenchmarkResult(
                benchmark_name=benchmark_name,
                success=False,
                duration=time.time() - start_time,
                initial_memory=0.0,
                peak_memory=0.0,
                final_memory=0.0,
                memory_efficiency=0.0,
                memory_leak_detected=False,
                error_message=str(e)
            )
    
    def benchmark_memory_optimization(self) -> MemoryBenchmarkResult:
        """Benchmark memory optimization strategies."""
        benchmark_name = "Memory Optimization"
        logger.info(f"Running {benchmark_name}...")
        
        self.memory_profiler.start_profiling(benchmark_name)
        start_time = time.time()
        
        try:
            # Initialize memory manager
            memory_manager = MultiTierMemoryManager()
            
            # Load various types of data
            data_types = [
                "Text data for processing",
                "Numerical data for analysis",
                "Structured data for reasoning",
                "Large document for summarization"
            ]
            
            for i, data in enumerate(data_types):
                memory_manager.add_to_active_memory(data * 100)  # Multiply for larger datasets
                self.memory_profiler.snapshot_memory(f"{benchmark_name}_load_{i}")
            
            # Record memory before optimization
            self.memory_profiler.snapshot_memory(f"{benchmark_name}_before_opt")
            
            # Perform optimization
            optimization_start = time.time()
            memory_manager.optimize_memory()
            optimization_time = time.time() - optimization_start
            
            # Record memory after optimization
            self.memory_profiler.snapshot_memory(f"{benchmark_name}_after_opt")
            
            total_time = time.time() - start_time
            self.memory_profiler.end_profiling(benchmark_name)
            
            # Get memory stats
            memory_stats = self.memory_profiler.get_memory_stats(benchmark_name)
            
            # Calculate optimization efficiency
            before_optimization = memory_stats.get('min', 0.0)
            after_optimization = memory_stats.get('max', 0.0)
            
            optimization_efficiency = after_optimization / before_optimization if before_optimization > 0 else 1.0
            
            return MemoryBenchmarkResult(
                benchmark_name=benchmark_name,
                success=True,
                duration=total_time,
                initial_memory=before_optimization,
                peak_memory=memory_stats.get('peak', 0.0),
                final_memory=after_optimization,
                memory_efficiency=optimization_efficiency,
                memory_leak_detected=False,
                details={
                    'optimization_time': optimization_time,
                    'data_types_processed': len(data_types),
                    'memory_stats': memory_stats
                }
            )
            
        except Exception as e:
            self.memory_profiler.end_profiling(benchmark_name)
            return MemoryBenchmarkResult(
                benchmark_name=benchmark_name,
                success=False,
                duration=time.time() - start_time,
                initial_memory=0.0,
                peak_memory=0.0,
                final_memory=0.0,
                memory_efficiency=0.0,
                memory_leak_detected=False,
                error_message=str(e)
            )
    
    def benchmark_memory_leak_detection(self) -> MemoryBenchmarkResult:
        """Benchmark memory leak detection capabilities."""
        benchmark_name = "Memory Leak Detection"
        logger.info(f"Running {benchmark_name}...")
        
        self.memory_profiler.start_profiling(benchmark_name)
        start_time = time.time()
        
        try:
            # Initialize memory manager
            memory_manager = MultiTierMemoryManager()
            
            # Simulate memory operations that might cause leaks
            leak_simulation_rounds = 5
            
            for round_num in range(leak_simulation_rounds):
                # Add data
                test_data = f"Test data for round {round_num} " * 100
                memory_manager.add_to_active_memory(test_data)
                
                # Simulate some processing
                time.sleep(0.1)
                
                # Record memory after each round
                self.memory_profiler.snapshot_memory(f"{benchmark_name}_round_{round_num}")
                
                # Force garbage collection to detect leaks
                gc.collect()
            
            total_time = time.time() - start_time
            self.memory_profiler.end_profiling(benchmark_name)
            
            # Get memory stats
            memory_stats = self.memory_profiler.get_memory_stats(benchmark_name)
            
            # Analyze for memory leaks
            initial_memory = memory_stats.get('min', 0.0)
            final_memory = memory_stats.get('max', 0.0)
            memory_growth = final_memory - initial_memory
            
            # Detect memory leak (growth > 10% of peak)
            peak_memory = memory_stats.get('peak', 0.0)
            leak_threshold = peak_memory * 0.1
            memory_leak_detected = memory_growth > leak_threshold
            
            memory_efficiency = initial_memory / final_memory if final_memory > 0 else 1.0
            
            return MemoryBenchmarkResult(
                benchmark_name=benchmark_name,
                success=True,
                duration=total_time,
                initial_memory=initial_memory,
                peak_memory=peak_memory,
                final_memory=final_memory,
                memory_efficiency=memory_efficiency,
                memory_leak_detected=memory_leak_detected,
                details={
                    'leak_simulation_rounds': leak_simulation_rounds,
                    'memory_growth_mb': memory_growth,
                    'leak_threshold_mb': leak_threshold,
                    'memory_stats': memory_stats
                }
            )
            
        except Exception as e:
            self.memory_profiler.end_profiling(benchmark_name)
            return MemoryBenchmarkResult(
                benchmark_name=benchmark_name,
                success=False,
                duration=time.time() - start_time,
                initial_memory=0.0,
                peak_memory=0.0,
                final_memory=0.0,
                memory_efficiency=0.0,
                memory_leak_detected=False,
                error_message=str(e)
            )
    
    def benchmark_concurrent_memory_access(self) -> MemoryBenchmarkResult:
        """Benchmark concurrent memory access patterns."""
        benchmark_name = "Concurrent Memory Access"
        logger.info(f"Running {benchmark_name}...")
        
        self.memory_profiler.start_profiling(benchmark_name)
        start_time = time.time()
        
        try:
            # Initialize memory manager
            memory_manager = MultiTierMemoryManager()
            
            # Define concurrent operations
            def read_operation(data_id: int):
                return f"Reading data chunk {data_id}"
            
            def write_operation(data_id: int):
                return f"Writing data chunk {data_id}"
            
            def process_operation(data_id: int):
                return f"Processing data chunk {data_id}"
            
            # Run concurrent operations
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = []
                
                # Submit different types of operations
                for i in range(20):
                    if i % 3 == 0:
                        future = executor.submit(read_operation, i)
                    elif i % 3 == 1:
                        future = executor.submit(write_operation, i)
                    else:
                        future = executor.submit(process_operation, i)
                    
                    futures.append(future)
                
                # Wait for completion
                results = [future.result() for future in as_completed(futures)]
            
            total_time = time.time() - start_time
            self.memory_profiler.end_profiling(benchmark_name)
            
            # Get memory stats
            memory_stats = self.memory_profiler.get_memory_stats(benchmark_name)
            
            # Calculate efficiency
            initial_memory = memory_stats.get('min', 0.0)
            peak_memory = memory_stats.get('peak', 0.0)
            final_memory = memory_stats.get('max', 0.0)
            
            memory_efficiency = final_memory / peak_memory if peak_memory > 0 else 1.0
            memory_leak = (final_memory - initial_memory) > (peak_memory * 0.1)
            
            return MemoryBenchmarkResult(
                benchmark_name=benchmark_name,
                success=True,
                duration=total_time,
                initial_memory=initial_memory,
                peak_memory=peak_memory,
                final_memory=final_memory,
                memory_efficiency=memory_efficiency,
                memory_leak_detected=memory_leak,
                details={
                    'concurrent_operations': len(futures),
                    'results_count': len(results),
                    'memory_stats': memory_stats
                }
            )
            
        except Exception as e:
            self.memory_profiler.end_profiling(benchmark_name)
            return MemoryBenchmarkResult(
                benchmark_name=benchmark_name,
                success=False,
                duration=time.time() - start_time,
                initial_memory=0.0,
                peak_memory=0.0,
                final_memory=0.0,
                memory_efficiency=0.0,
                memory_leak_detected=False,
                error_message=str(e)
            )
    
    def run_comprehensive_memory_benchmarks(self) -> Dict[str, Any]:
        """Run comprehensive memory benchmarks."""
        logger.info("Running comprehensive memory benchmarks...")
        
        # Start performance monitoring
        self.performance_monitor.start_monitoring()
        
        try:
            # Run individual benchmarks
            benchmarks = [
                lambda: self.benchmark_memory_allocation("small"),
                lambda: self.benchmark_memory_allocation("medium"),
                lambda: self.benchmark_memory_allocation("large"),
                lambda: self.benchmark_memory_compression(),
                lambda: self.benchmark_memory_optimization(),
                lambda: self.benchmark_memory_leak_detection(),
                lambda: self.benchmark_concurrent_memory_access()
            ]
            
            for benchmark in benchmarks:
                try:
                    result = benchmark()
                    self.results.append(result)
                    logger.info(f"Completed {result.benchmark_name}: {'PASS' if result.success else 'FAIL'}")
                except Exception as e:
                    logger.error(f"Benchmark failed: {e}")
        
        finally:
            # Stop performance monitoring
            self.performance_monitor.stop_monitoring()
        
        # Generate summary report
        return self.generate_memory_summary_report()
    
    def generate_memory_summary_report(self) -> Dict[str, Any]:
        """Generate comprehensive memory benchmark summary report."""
        if not self.results:
            return {"error": "No memory benchmark results available"}
        
        # Calculate statistics
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.success)
        failed_tests = total_tests - passed_tests
        
        # Memory metrics
        memory_efficiencies = [r.memory_efficiency for r in self.results if r.success]
        memory_leaks = sum(1 for r in self.results if r.memory_leak_detected)
        
        # Performance metrics
        durations = [r.duration for r in self.results if r.success]
        peak_memories = [r.peak_memory for r in self.results if r.success]
        
        return {
            'summary': {
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'failed_tests': failed_tests,
                'success_rate': passed_tests / total_tests if total_tests > 0 else 0
            },
            'memory_metrics': {
                'avg_memory_efficiency': np.mean(memory_efficiencies) if memory_efficiencies else 0,
                'min_memory_efficiency': np.min(memory_efficiencies) if memory_efficiencies else 0,
                'max_memory_efficiency': np.max(memory_efficiencies) if memory_efficiencies else 0,
                'memory_leaks_detected': memory_leaks,
                'leak_rate': memory_leaks / total_tests if total_tests > 0 else 0
            },
            'performance_metrics': {
                'avg_duration': np.mean(durations) if durations else 0,
                'max_duration': np.max(durations) if durations else 0,
                'avg_peak_memory': np.mean(peak_memories) if peak_memories else 0,
                'max_peak_memory': np.max(peak_memories) if peak_memories else 0
            },
            'detailed_results': [r.to_dict() for r in self.results]
        }


def run_memory_benchmarks(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Run memory benchmarks with optional configuration."""
    benchmarks = MemoryBenchmarks()
    return benchmarks.run_comprehensive_memory_benchmarks() 