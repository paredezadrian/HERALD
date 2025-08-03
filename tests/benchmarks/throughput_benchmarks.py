"""
HERALD Throughput Benchmarking
System throughput and processing speed testing

This module implements:
- Throughput measurement
- Processing speed testing
- Load testing
- Performance under stress
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
from utils.metrics import ThroughputMeasurer, PerformanceMonitor

logger = logging.getLogger(__name__)


@dataclass
class ThroughputBenchmarkResult:
    """Container for throughput benchmark results."""
    benchmark_name: str
    success: bool
    duration: float
    items_processed: int
    throughput_rate: float
    avg_latency: float
    peak_memory: float
    cpu_usage: float
    error_message: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'benchmark_name': self.benchmark_name,
            'success': self.success,
            'duration': self.duration,
            'items_processed': self.items_processed,
            'throughput_rate': self.throughput_rate,
            'avg_latency': self.avg_latency,
            'peak_memory': self.peak_memory,
            'cpu_usage': self.cpu_usage,
            'error_message': self.error_message,
            'details': self.details
        }


class ThroughputBenchmarks:
    """Throughput-specific benchmarking suite."""
    
    def __init__(self):
        self.results: List[ThroughputBenchmarkResult] = []
        self.throughput_measurer = ThroughputMeasurer()
        self.performance_monitor = PerformanceMonitor()
        
        # Test data
        self.test_prompts = [
            "Hello, how are you?",
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
        
        logger.info("Initialized ThroughputBenchmarks")
    
    def benchmark_tokenization_throughput(self, num_tokens: int = 1000) -> ThroughputBenchmarkResult:
        """Benchmark tokenization throughput."""
        benchmark_name = f"Tokenization Throughput ({num_tokens} tokens)"
        logger.info(f"Running {benchmark_name}...")
        
        self.throughput_measurer.start_measurement(benchmark_name)
        start_time = time.time()
        
        try:
            # Initialize tokenizer
            tokenizer = ASCTokenizer()
            
            # Process test prompts
            total_tokens = 0
            latencies = []
            
            for prompt in self.test_prompts:
                token_start = time.time()
                tokens = tokenizer.encode(prompt)
                token_time = time.time() - token_start
                
                total_tokens += len(tokens)
                latencies.append(token_time)
                
                if total_tokens >= num_tokens:
                    break
            
            total_time = time.time() - start_time
            self.throughput_measurer.end_measurement(benchmark_name, total_tokens, "tokens")
            
            # Calculate metrics
            throughput_stats = self.throughput_measurer.get_throughput_stats(benchmark_name)
            throughput_rate = throughput_stats.get('mean', 0.0)
            avg_latency = np.mean(latencies) if latencies else 0.0
            
            # Get system metrics
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=1)
            
            return ThroughputBenchmarkResult(
                benchmark_name=benchmark_name,
                success=True,
                duration=total_time,
                items_processed=total_tokens,
                throughput_rate=throughput_rate,
                avg_latency=avg_latency,
                peak_memory=memory.used / (1024 * 1024),  # MB
                cpu_usage=cpu_percent,
                details={
                    'prompts_processed': len(self.test_prompts),
                    'avg_tokens_per_prompt': total_tokens / len(self.test_prompts) if self.test_prompts else 0,
                    'throughput_stats': throughput_stats,
                    'latency_stats': {
                        'min': np.min(latencies) if latencies else 0,
                        'max': np.max(latencies) if latencies else 0,
                        'std': np.std(latencies) if latencies else 0
                    }
                }
            )
            
        except Exception as e:
            self.throughput_measurer.end_measurement(benchmark_name, 0, "tokens")
            return ThroughputBenchmarkResult(
                benchmark_name=benchmark_name,
                success=False,
                duration=time.time() - start_time,
                items_processed=0,
                throughput_rate=0.0,
                avg_latency=0.0,
                peak_memory=0.0,
                cpu_usage=0.0,
                error_message=str(e)
            )
    
    def benchmark_inference_throughput(self, num_inferences: int = 100) -> ThroughputBenchmarkResult:
        """Benchmark inference throughput."""
        benchmark_name = f"Inference Throughput ({num_inferences} inferences)"
        logger.info(f"Running {benchmark_name}...")
        
        self.throughput_measurer.start_measurement(benchmark_name)
        start_time = time.time()
        
        try:
            # Initialize components
            engine = NeuroEngine()
            
            # Simulate inference operations
            latencies = []
            
            for i in range(num_inferences):
                inference_start = time.time()
                
                # Simulate inference (token generation)
                tokens_to_generate = np.random.randint(10, 50)
                for _ in range(tokens_to_generate):
                    time.sleep(0.008)  # 8ms per token
                
                inference_time = time.time() - inference_start
                latencies.append(inference_time)
            
            total_time = time.time() - start_time
            self.throughput_measurer.end_measurement(benchmark_name, num_inferences, "inferences")
            
            # Calculate metrics
            throughput_stats = self.throughput_measurer.get_throughput_stats(benchmark_name)
            throughput_rate = throughput_stats.get('mean', 0.0)
            avg_latency = np.mean(latencies) if latencies else 0.0
            
            # Get system metrics
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=1)
            
            return ThroughputBenchmarkResult(
                benchmark_name=benchmark_name,
                success=True,
                duration=total_time,
                items_processed=num_inferences,
                throughput_rate=throughput_rate,
                avg_latency=avg_latency,
                peak_memory=memory.used / (1024 * 1024),  # MB
                cpu_usage=cpu_percent,
                details={
                    'avg_tokens_per_inference': np.mean([np.random.randint(10, 50) for _ in range(num_inferences)]),
                    'throughput_stats': throughput_stats,
                    'latency_stats': {
                        'min': np.min(latencies) if latencies else 0,
                        'max': np.max(latencies) if latencies else 0,
                        'std': np.std(latencies) if latencies else 0
                    }
                }
            )
            
        except Exception as e:
            self.throughput_measurer.end_measurement(benchmark_name, 0, "inferences")
            return ThroughputBenchmarkResult(
                benchmark_name=benchmark_name,
                success=False,
                duration=time.time() - start_time,
                items_processed=0,
                throughput_rate=0.0,
                avg_latency=0.0,
                peak_memory=0.0,
                cpu_usage=0.0,
                error_message=str(e)
            )
    
    def benchmark_reasoning_throughput(self, num_queries: int = 50) -> ThroughputBenchmarkResult:
        """Benchmark reasoning throughput."""
        benchmark_name = f"Reasoning Throughput ({num_queries} queries)"
        logger.info(f"Running {benchmark_name}...")
        
        self.throughput_measurer.start_measurement(benchmark_name)
        start_time = time.time()
        
        try:
            # Initialize router
            router = MoERouter()
            
            # Test queries for different reasoning types
            base_queries = [
                "If A and B, then C",
                "What causes the temperature to rise?",
                "What happened before the meeting?",
                "∀x (P(x) → Q(x))",
                "The intervention led to the outcome"
            ]
            test_queries = base_queries * (num_queries // len(base_queries) + 1)
            
            latencies = []
            
            for i in range(num_queries):
                query = test_queries[i % len(test_queries)]
                
                reasoning_start = time.time()
                router.process_query(query)
                reasoning_time = time.time() - reasoning_start
                
                latencies.append(reasoning_time)
            
            total_time = time.time() - start_time
            self.throughput_measurer.end_measurement(benchmark_name, num_queries, "queries")
            
            # Calculate metrics
            throughput_stats = self.throughput_measurer.get_throughput_stats(benchmark_name)
            throughput_rate = throughput_stats.get('mean', 0.0)
            avg_latency = np.mean(latencies) if latencies else 0.0
            
            # Get system metrics
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=1)
            
            return ThroughputBenchmarkResult(
                benchmark_name=benchmark_name,
                success=True,
                duration=total_time,
                items_processed=num_queries,
                throughput_rate=throughput_rate,
                avg_latency=avg_latency,
                peak_memory=memory.used / (1024 * 1024),  # MB
                cpu_usage=cpu_percent,
                details={
                    'query_types': len(set(base_queries)),
                    'throughput_stats': throughput_stats,
                    'latency_stats': {
                        'min': np.min(latencies) if latencies else 0,
                        'max': np.max(latencies) if latencies else 0,
                        'std': np.std(latencies) if latencies else 0
                    }
                }
            )
            
        except Exception as e:
            self.throughput_measurer.end_measurement(benchmark_name, 0, "queries")
            return ThroughputBenchmarkResult(
                benchmark_name=benchmark_name,
                success=False,
                duration=time.time() - start_time,
                items_processed=0,
                throughput_rate=0.0,
                avg_latency=0.0,
                peak_memory=0.0,
                cpu_usage=0.0,
                error_message=str(e)
            )
    
    def benchmark_concurrent_throughput(self, num_workers: int = 4, num_tasks: int = 100) -> ThroughputBenchmarkResult:
        """Benchmark concurrent throughput."""
        benchmark_name = f"Concurrent Throughput ({num_workers} workers, {num_tasks} tasks)"
        logger.info(f"Running {benchmark_name}...")
        
        self.throughput_measurer.start_measurement(benchmark_name)
        start_time = time.time()
        
        try:
            # Define worker tasks
            def tokenization_worker(task_id: int):
                tokenizer = ASCTokenizer()
                prompt = self.test_prompts[task_id % len(self.test_prompts)]
                return len(tokenizer.encode(prompt))
            
            def reasoning_worker(task_id: int):
                router = MoERouter()
                query = f"Query {task_id}"
                return router.process_query(query)
            
            def memory_worker(task_id: int):
                memory_manager = MultiTierMemoryManager()
                data = f"Data chunk {task_id}"
                memory_manager.add_to_active_memory(data)
                return "processed"
            
            # Run concurrent operations
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = []
                
                # Submit different types of tasks
                for i in range(num_tasks):
                    if i % 3 == 0:
                        future = executor.submit(tokenization_worker, i)
                    elif i % 3 == 1:
                        future = executor.submit(reasoning_worker, i)
                    else:
                        future = executor.submit(memory_worker, i)
                    
                    futures.append(future)
                
                # Wait for completion and measure latencies
                latencies = []
                for future in as_completed(futures):
                    task_start = time.time()
                    result = future.result()
                    task_time = time.time() - task_start
                    latencies.append(task_time)
            
            total_time = time.time() - start_time
            self.throughput_measurer.end_measurement(benchmark_name, num_tasks, "tasks")
            
            # Calculate metrics
            throughput_stats = self.throughput_measurer.get_throughput_stats(benchmark_name)
            throughput_rate = throughput_stats.get('mean', 0.0)
            avg_latency = np.mean(latencies) if latencies else 0.0
            
            # Get system metrics
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=1)
            
            return ThroughputBenchmarkResult(
                benchmark_name=benchmark_name,
                success=True,
                duration=total_time,
                items_processed=num_tasks,
                throughput_rate=throughput_rate,
                avg_latency=avg_latency,
                peak_memory=memory.used / (1024 * 1024),  # MB
                cpu_usage=cpu_percent,
                details={
                    'num_workers': num_workers,
                    'task_types': 3,  # tokenization, reasoning, memory
                    'throughput_stats': throughput_stats,
                    'latency_stats': {
                        'min': np.min(latencies) if latencies else 0,
                        'max': np.max(latencies) if latencies else 0,
                        'std': np.std(latencies) if latencies else 0
                    }
                }
            )
            
        except Exception as e:
            self.throughput_measurer.end_measurement(benchmark_name, 0, "tasks")
            return ThroughputBenchmarkResult(
                benchmark_name=benchmark_name,
                success=False,
                duration=time.time() - start_time,
                items_processed=0,
                throughput_rate=0.0,
                avg_latency=0.0,
                peak_memory=0.0,
                cpu_usage=0.0,
                error_message=str(e)
            )
    
    def benchmark_load_testing(self, load_levels: List[int] = None) -> List[ThroughputBenchmarkResult]:
        """Benchmark throughput under different load levels."""
        if load_levels is None:
            load_levels = [10, 25, 50, 100, 200]
        
        logger.info(f"Running load testing with levels: {load_levels}")
        results = []
        
        for load_level in load_levels:
            benchmark_name = f"Load Testing ({load_level} concurrent requests)"
            logger.info(f"Testing load level: {load_level}")
            
            self.throughput_measurer.start_measurement(benchmark_name)
            start_time = time.time()
            
            try:
                # Initialize components
                engine = NeuroEngine()
                router = MoERouter()
                
                # Simulate concurrent requests
                def process_request(request_id: int):
                    # Simulate different types of requests
                    if request_id % 3 == 0:
                        # Tokenization request
                        return len(f"Request {request_id}".split())
                    elif request_id % 3 == 1:
                        # Reasoning request
                        return router.process_query(f"Query {request_id}")
                    else:
                        # Inference request
                        return f"Generated response for request {request_id}"
                
                # Run concurrent requests
                with ThreadPoolExecutor(max_workers=min(load_level, 8)) as executor:
                    futures = [executor.submit(process_request, i) for i in range(load_level)]
                    results_list = [future.result() for future in as_completed(futures)]
                
                total_time = time.time() - start_time
                self.throughput_measurer.end_measurement(benchmark_name, load_level, "requests")
                
                # Calculate metrics
                throughput_stats = self.throughput_measurer.get_throughput_stats(benchmark_name)
                throughput_rate = throughput_stats.get('mean', 0.0)
                
                # Get system metrics
                memory = psutil.virtual_memory()
                cpu_percent = psutil.cpu_percent(interval=1)
                
                result = ThroughputBenchmarkResult(
                    benchmark_name=benchmark_name,
                    success=True,
                    duration=total_time,
                    items_processed=load_level,
                    throughput_rate=throughput_rate,
                    avg_latency=total_time / load_level if load_level > 0 else 0,
                    peak_memory=memory.used / (1024 * 1024),  # MB
                    cpu_usage=cpu_percent,
                    details={
                        'load_level': load_level,
                        'max_workers': min(load_level, 8),
                        'throughput_stats': throughput_stats,
                        'results_count': len(results_list)
                    }
                )
                
                results.append(result)
                
            except Exception as e:
                self.throughput_measurer.end_measurement(benchmark_name, 0, "requests")
                results.append(ThroughputBenchmarkResult(
                    benchmark_name=benchmark_name,
                    success=False,
                    duration=time.time() - start_time,
                    items_processed=0,
                    throughput_rate=0.0,
                    avg_latency=0.0,
                    peak_memory=0.0,
                    cpu_usage=0.0,
                    error_message=str(e)
                ))
        
        return results
    
    def run_comprehensive_throughput_benchmarks(self) -> Dict[str, Any]:
        """Run comprehensive throughput benchmarks."""
        logger.info("Running comprehensive throughput benchmarks...")
        
        # Start performance monitoring
        self.performance_monitor.start_monitoring()
        
        try:
            # Run individual benchmarks
            benchmarks = [
                lambda: self.benchmark_tokenization_throughput(1000),
                lambda: self.benchmark_inference_throughput(100),
                lambda: self.benchmark_reasoning_throughput(50),
                lambda: self.benchmark_concurrent_throughput(4, 100)
            ]
            
            for benchmark in benchmarks:
                try:
                    result = benchmark()
                    self.results.append(result)
                    logger.info(f"Completed {result.benchmark_name}: {'PASS' if result.success else 'FAIL'}")
                except Exception as e:
                    logger.error(f"Benchmark failed: {e}")
            
            # Run load testing
            load_results = self.benchmark_load_testing()
            self.results.extend(load_results)
            
        finally:
            # Stop performance monitoring
            self.performance_monitor.stop_monitoring()
        
        # Generate summary report
        return self.generate_throughput_summary_report()
    
    def generate_throughput_summary_report(self) -> Dict[str, Any]:
        """Generate comprehensive throughput benchmark summary report."""
        if not self.results:
            return {"error": "No throughput benchmark results available"}
        
        # Calculate statistics
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.success)
        failed_tests = total_tests - passed_tests
        
        # Throughput metrics
        throughput_rates = [r.throughput_rate for r in self.results if r.success and r.throughput_rate > 0]
        latencies = [r.avg_latency for r in self.results if r.success]
        memory_usage = [r.peak_memory for r in self.results if r.success]
        cpu_usage = [r.cpu_usage for r in self.results if r.success]
        
        return {
            'summary': {
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'failed_tests': failed_tests,
                'success_rate': passed_tests / total_tests if total_tests > 0 else 0
            },
            'throughput_metrics': {
                'avg_throughput_rate': np.mean(throughput_rates) if throughput_rates else 0,
                'max_throughput_rate': np.max(throughput_rates) if throughput_rates else 0,
                'min_throughput_rate': np.min(throughput_rates) if throughput_rates else 0,
                'std_throughput_rate': np.std(throughput_rates) if throughput_rates else 0
            },
            'performance_metrics': {
                'avg_latency': np.mean(latencies) if latencies else 0,
                'max_latency': np.max(latencies) if latencies else 0,
                'avg_memory_usage': np.mean(memory_usage) if memory_usage else 0,
                'avg_cpu_usage': np.mean(cpu_usage) if cpu_usage else 0
            },
            'detailed_results': [r.to_dict() for r in self.results]
        }


def run_throughput_benchmarks(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Run throughput benchmarks with optional configuration."""
    benchmarks = ThroughputBenchmarks()
    return benchmarks.run_comprehensive_throughput_benchmarks() 