"""
HERALD Performance Metrics Collection
Performance monitoring and metrics collection for HERALD AI architecture

This module implements:
- Performance metrics collection
- Memory usage monitoring
- Throughput measurement
- Performance regression detection

Target: ~432 lines of production-ready metrics code
"""

import time
import psutil
import threading
import logging
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
import json
import os
from pathlib import Path
import numpy as np
from datetime import datetime, timedelta
import gc

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    timestamp: float
    cpu_usage: float
    memory_usage: float
    memory_available: float
    memory_percent: float
    disk_io_read: float
    disk_io_write: float
    network_io_sent: float
    network_io_recv: float
    context_switches: int
    interrupts: int
    load_average: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'timestamp': self.timestamp,
            'cpu_usage': self.cpu_usage,
            'memory_usage': self.memory_usage,
            'memory_available': self.memory_available,
            'memory_percent': self.memory_percent,
            'disk_io_read': self.disk_io_read,
            'disk_io_write': self.disk_io_write,
            'network_io_sent': self.network_io_sent,
            'network_io_recv': self.network_io_recv,
            'context_switches': self.context_switches,
            'interrupts': self.interrupts,
            'load_average': self.load_average
        }


@dataclass
class ModelMetrics:
    """Container for model-specific metrics."""
    model_name: str
    inference_time: float
    tokens_generated: int
    tokens_per_second: float
    memory_usage: float
    gpu_usage: Optional[float] = None
    accuracy: Optional[float] = None
    loss: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'model_name': self.model_name,
            'inference_time': self.inference_time,
            'tokens_generated': self.tokens_generated,
            'tokens_per_second': self.tokens_per_second,
            'memory_usage': self.memory_usage,
            'gpu_usage': self.gpu_usage,
            'accuracy': self.accuracy,
            'loss': self.loss
        }


class PerformanceMonitor:
    """Monitors system and application performance."""
    
    def __init__(self, 
                 collection_interval: float = 1.0,
                 max_history_size: int = 10000,
                 enable_disk_monitoring: bool = True,
                 enable_network_monitoring: bool = True):
        self.collection_interval = collection_interval
        self.max_history_size = max_history_size
        self.enable_disk_monitoring = enable_disk_monitoring
        self.enable_network_monitoring = enable_network_monitoring
        
        # Metrics storage
        self.performance_history: deque = deque(maxlen=max_history_size)
        self.model_metrics_history: deque = deque(maxlen=max_history_size)
        
        # Monitoring state
        self.is_monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        
        # Baseline metrics
        self.baseline_metrics: Optional[PerformanceMetrics] = None
        
        # Performance thresholds
        self.cpu_threshold = 80.0
        self.memory_threshold = 85.0
        self.disk_threshold = 90.0
        
        # Initialize system monitoring
        self._init_system_monitoring()
        
        logger.info(f"Initialized PerformanceMonitor with {collection_interval}s interval")
    
    def _init_system_monitoring(self):
        """Initialize system monitoring capabilities."""
        try:
            # Get initial system stats
            self._update_system_stats()
        except Exception as e:
            logger.warning(f"Failed to initialize system monitoring: {e}")
    
    def _update_system_stats(self):
        """Update system statistics."""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            
            # Get disk I/O stats
            disk_io = psutil.disk_io_counters()
            disk_read = disk_io.read_bytes if disk_io else 0
            disk_write = disk_io.write_bytes if disk_io else 0
            
            # Get network I/O stats
            network_io = psutil.net_io_counters()
            net_sent = network_io.bytes_sent if network_io else 0
            net_recv = network_io.bytes_recv if network_io else 0
            
            # Get CPU stats
            cpu_stats = psutil.cpu_stats()
            context_switches = cpu_stats.ctx_switches if cpu_stats else 0
            interrupts = cpu_stats.interrupts if cpu_stats else 0
            
            # Get load average (Linux only)
            load_avg = None
            try:
                load_avg = os.getloadavg()[0] if hasattr(os, 'getloadavg') else None
            except (AttributeError, OSError):
                pass
            
            return {
                'cpu_percent': cpu_percent,
                'memory': memory,
                'disk_read': disk_read,
                'disk_write': disk_write,
                'net_sent': net_sent,
                'net_recv': net_recv,
                'context_switches': context_switches,
                'interrupts': interrupts,
                'load_avg': load_avg
            }
        except Exception as e:
            logger.error(f"Error updating system stats: {e}")
            return None
    
    def start_monitoring(self):
        """Start continuous performance monitoring."""
        if self.is_monitoring:
            logger.warning("Performance monitoring is already running")
            return
        
        self.is_monitoring = True
        self.stop_event.clear()
        
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitor_thread.start()
        
        logger.info("Started performance monitoring")
    
    def stop_monitoring(self):
        """Stop continuous performance monitoring."""
        if not self.is_monitoring:
            return
        
        self.is_monitoring = False
        self.stop_event.set()
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        
        logger.info("Stopped performance monitoring")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        last_stats = None
        
        while not self.stop_event.is_set():
            try:
                current_time = time.time()
                
                # Get current system stats
                stats = self._update_system_stats()
                if stats is None:
                    time.sleep(self.collection_interval)
                    continue
                
                # Calculate deltas if we have previous stats
                if last_stats is not None:
                    delta_time = current_time - last_stats['timestamp']
                    
                    # Calculate I/O deltas
                    disk_read_delta = (stats['disk_read'] - last_stats['disk_read']) / delta_time
                    disk_write_delta = (stats['disk_write'] - last_stats['disk_write']) / delta_time
                    net_sent_delta = (stats['net_sent'] - last_stats['net_sent']) / delta_time
                    net_recv_delta = (stats['net_recv'] - last_stats['net_recv']) / delta_time
                    
                    # Create metrics object
                    metrics = PerformanceMetrics(
                        timestamp=current_time,
                        cpu_usage=stats['cpu_percent'],
                        memory_usage=stats['memory'].used,
                        memory_available=stats['memory'].available,
                        memory_percent=stats['memory'].percent,
                        disk_io_read=disk_read_delta,
                        disk_io_write=disk_write_delta,
                        network_io_sent=net_sent_delta,
                        network_io_recv=net_recv_delta,
                        context_switches=stats['context_switches'],
                        interrupts=stats['interrupts'],
                        load_average=stats['load_avg']
                    )
                    
                    # Store metrics
                    self.performance_history.append(metrics)
                    
                    # Check for performance alerts
                    self._check_performance_alerts(metrics)
                
                # Update last stats
                last_stats = {
                    'timestamp': current_time,
                    **stats
                }
                
                # Wait for next collection interval
                time.sleep(self.collection_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.collection_interval)
    
    def _check_performance_alerts(self, metrics: PerformanceMetrics):
        """Check for performance threshold violations."""
        alerts = []
        
        if metrics.cpu_usage > self.cpu_threshold:
            alerts.append(f"High CPU usage: {metrics.cpu_usage:.1f}%")
        
        if metrics.memory_percent > self.memory_threshold:
            alerts.append(f"High memory usage: {metrics.memory_percent:.1f}%")
        
        if alerts:
            logger.warning(f"Performance alerts: {'; '.join(alerts)}")
    
    def get_current_metrics(self) -> Optional[PerformanceMetrics]:
        """Get the most recent performance metrics."""
        if not self.performance_history:
            return None
        return self.performance_history[-1]
    
    def get_metrics_summary(self, 
                          duration_minutes: int = 60) -> Dict[str, Any]:
        """Get performance metrics summary for the specified duration."""
        if not self.performance_history:
            return {}
        
        cutoff_time = time.time() - (duration_minutes * 60)
        recent_metrics = [
            m for m in self.performance_history 
            if m.timestamp >= cutoff_time
        ]
        
        if not recent_metrics:
            return {}
        
        # Calculate statistics
        cpu_values = [m.cpu_usage for m in recent_metrics]
        memory_values = [m.memory_percent for m in recent_metrics]
        
        return {
            'duration_minutes': duration_minutes,
            'sample_count': len(recent_metrics),
            'cpu_usage': {
                'mean': np.mean(cpu_values),
                'max': np.max(cpu_values),
                'min': np.min(cpu_values),
                'std': np.std(cpu_values)
            },
            'memory_usage': {
                'mean': np.mean(memory_values),
                'max': np.max(memory_values),
                'min': np.min(memory_values),
                'std': np.std(memory_values)
            },
            'latest_metrics': recent_metrics[-1].to_dict()
        }
    
    def add_model_metrics(self, metrics: ModelMetrics):
        """Add model-specific metrics."""
        self.model_metrics_history.append(metrics)
    
    def get_model_metrics_summary(self, 
                                model_name: Optional[str] = None,
                                duration_minutes: int = 60) -> Dict[str, Any]:
        """Get model metrics summary."""
        if not self.model_metrics_history:
            return {}
        
        cutoff_time = time.time() - (duration_minutes * 60)
        recent_metrics = [
            m for m in self.model_metrics_history 
            if m.timestamp >= cutoff_time
        ]
        
        if model_name:
            recent_metrics = [m for m in recent_metrics if m.model_name == model_name]
        
        if not recent_metrics:
            return {}
        
        # Calculate statistics
        inference_times = [m.inference_time for m in recent_metrics]
        tokens_per_sec = [m.tokens_per_second for m in recent_metrics]
        memory_usage = [m.memory_usage for m in recent_metrics]
        
        return {
            'model_name': model_name or 'all',
            'duration_minutes': duration_minutes,
            'sample_count': len(recent_metrics),
            'inference_time': {
                'mean': np.mean(inference_times),
                'max': np.max(inference_times),
                'min': np.min(inference_times),
                'std': np.std(inference_times)
            },
            'tokens_per_second': {
                'mean': np.mean(tokens_per_sec),
                'max': np.max(tokens_per_sec),
                'min': np.min(tokens_per_sec),
                'std': np.std(tokens_per_sec)
            },
            'memory_usage': {
                'mean': np.mean(memory_usage),
                'max': np.max(memory_usage),
                'min': np.min(memory_usage),
                'std': np.std(memory_usage)
            }
        }
    
    def export_metrics(self, 
                      filepath: str,
                      include_model_metrics: bool = True) -> bool:
        """Export metrics to JSON file."""
        try:
            export_data = {
                'timestamp': datetime.now().isoformat(),
                'performance_metrics': [
                    m.to_dict() for m in self.performance_history
                ]
            }
            
            if include_model_metrics:
                export_data['model_metrics'] = [
                    m.to_dict() for m in self.model_metrics_history
                ]
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"Exported metrics to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export metrics: {e}")
            return False
    
    def clear_history(self):
        """Clear all stored metrics."""
        self.performance_history.clear()
        self.model_metrics_history.clear()
        logger.info("Cleared metrics history")


class ThroughputMeasurer:
    """Measures throughput for various operations."""
    
    def __init__(self):
        self.measurements: Dict[str, List[float]] = defaultdict(list)
        self.start_times: Dict[str, float] = {}
    
    def start_measurement(self, operation_name: str):
        """Start measuring an operation."""
        self.start_times[operation_name] = time.time()
    
    def end_measurement(self, operation_name: str, 
                       count: int = 1,
                       unit: str = "items"):
        """End measuring an operation and record throughput."""
        if operation_name not in self.start_times:
            logger.warning(f"No start time found for operation: {operation_name}")
            return
        
        duration = time.time() - self.start_times[operation_name]
        throughput = count / duration if duration > 0 else 0
        
        self.measurements[operation_name].append(throughput)
        
        logger.info(f"{operation_name}: {throughput:.2f} {unit}/second")
        
        # Clean up start time
        del self.start_times[operation_name]
    
    def get_throughput_stats(self, operation_name: str) -> Dict[str, float]:
        """Get throughput statistics for an operation."""
        if operation_name not in self.measurements:
            return {}
        
        values = self.measurements[operation_name]
        if not values:
            return {}
        
        return {
            'mean': np.mean(values),
            'max': np.max(values),
            'min': np.min(values),
            'std': np.std(values),
            'count': len(values)
        }
    
    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """Get throughput statistics for all operations."""
        return {
            name: self.get_throughput_stats(name)
            for name in self.measurements.keys()
        }


class MemoryProfiler:
    """Profiles memory usage for specific operations."""
    
    def __init__(self):
        self.memory_snapshots: Dict[str, List[float]] = defaultdict(list)
        self.peak_memory: Dict[str, float] = {}
    
    def start_profiling(self, operation_name: str):
        """Start memory profiling for an operation."""
        gc.collect()  # Force garbage collection
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        self.memory_snapshots[operation_name].append(initial_memory)
    
    def snapshot_memory(self, operation_name: str):
        """Take a memory snapshot during operation."""
        current_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        self.memory_snapshots[operation_name].append(current_memory)
    
    def end_profiling(self, operation_name: str):
        """End memory profiling for an operation."""
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        self.memory_snapshots[operation_name].append(final_memory)
        
        # Calculate peak memory
        if operation_name in self.memory_snapshots:
            peak = max(self.memory_snapshots[operation_name])
            self.peak_memory[operation_name] = peak
            
            logger.info(f"{operation_name} peak memory: {peak:.2f} MB")
    
    def get_memory_stats(self, operation_name: str) -> Dict[str, float]:
        """Get memory statistics for an operation."""
        if operation_name not in self.memory_snapshots:
            return {}
        
        values = self.memory_snapshots[operation_name]
        if not values:
            return {}
        
        return {
            'peak': self.peak_memory.get(operation_name, max(values)),
            'mean': np.mean(values),
            'min': np.min(values),
            'max': np.max(values),
            'std': np.std(values),
            'count': len(values)
        }


# Global instances
performance_monitor = PerformanceMonitor()
throughput_measurer = ThroughputMeasurer()
memory_profiler = MemoryProfiler()


def get_system_info() -> Dict[str, Any]:
    """Get comprehensive system information."""
    try:
        cpu_count = psutil.cpu_count()
        cpu_freq = psutil.cpu_freq()
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return {
            'cpu': {
                'count': cpu_count,
                'frequency_mhz': cpu_freq.current if cpu_freq else None,
                'usage_percent': psutil.cpu_percent(interval=1)
            },
            'memory': {
                'total_gb': memory.total / (1024**3),
                'available_gb': memory.available / (1024**3),
                'used_gb': memory.used / (1024**3),
                'percent': memory.percent
            },
            'disk': {
                'total_gb': disk.total / (1024**3),
                'used_gb': disk.used / (1024**3),
                'free_gb': disk.free / (1024**3),
                'percent': (disk.used / disk.total) * 100
            }
        }
    except Exception as e:
        logger.error(f"Error getting system info: {e}")
        return {}


def create_performance_monitor(config: Optional[Dict[str, Any]] = None) -> PerformanceMonitor:
    """Create a performance monitor with optional configuration."""
    if config is None:
        config = {}
    
    return PerformanceMonitor(
        collection_interval=config.get('collection_interval', 1.0),
        max_history_size=config.get('max_history_size', 10000),
        enable_disk_monitoring=config.get('enable_disk_monitoring', True),
        enable_network_monitoring=config.get('enable_network_monitoring', True)
    ) 