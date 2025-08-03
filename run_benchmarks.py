#!/usr/bin/env python3
"""
HERALD Benchmarking Suite Runner
Runs all benchmark tests and displays results
"""

import sys
import traceback
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

def run_performance_benchmarks():
    """Run performance benchmarks."""
    print("=" * 60)
    print("RUNNING PERFORMANCE BENCHMARKS")
    print("=" * 60)
    
    try:
        from tests.benchmarks.performance_benchmarks import run_performance_benchmarks
        result = run_performance_benchmarks()
        
        print(f"Total tests: {result['summary']['total_tests']}")
        print(f"Passed: {result['summary']['passed_tests']}")
        print(f"Failed: {result['summary']['failed_tests']}")
        print(f"Success rate: {result['summary']['success_rate']:.2%}")
        
        if result['summary']['failed_tests'] > 0:
            print("\nFailed tests:")
            for test_result in result['detailed_results']:
                if not test_result['success']:
                    print(f"  - {test_result['test_name']}: {test_result.get('error_message', 'Unknown error')}")
        
        return result
    except Exception as e:
        print(f"Error running performance benchmarks: {e}")
        traceback.print_exc()
        return None

def run_model_benchmarks():
    """Run model benchmarks."""
    print("\n" + "=" * 60)
    print("RUNNING MODEL BENCHMARKS")
    print("=" * 60)
    
    try:
        from tests.benchmarks.model_benchmarks import run_model_benchmarks
        result = run_model_benchmarks()
        
        print(f"Total tests: {result['summary']['total_tests']}")
        print(f"Passed: {result['summary']['passed_tests']}")
        print(f"Failed: {result['summary']['failed_tests']}")
        print(f"Success rate: {result['summary']['success_rate']:.2%}")
        
        return result
    except Exception as e:
        print(f"Error running model benchmarks: {e}")
        traceback.print_exc()
        return None

def run_memory_benchmarks():
    """Run memory benchmarks."""
    print("\n" + "=" * 60)
    print("RUNNING MEMORY BENCHMARKS")
    print("=" * 60)
    
    try:
        from tests.benchmarks.memory_benchmarks import run_memory_benchmarks
        result = run_memory_benchmarks()
        
        print(f"Total tests: {result['summary']['total_tests']}")
        print(f"Passed: {result['summary']['passed_tests']}")
        print(f"Failed: {result['summary']['failed_tests']}")
        print(f"Success rate: {result['summary']['success_rate']:.2%}")
        
        return result
    except Exception as e:
        print(f"Error running memory benchmarks: {e}")
        traceback.print_exc()
        return None

def run_throughput_benchmarks():
    """Run throughput benchmarks."""
    print("\n" + "=" * 60)
    print("RUNNING THROUGHPUT BENCHMARKS")
    print("=" * 60)
    
    try:
        from tests.benchmarks.throughput_benchmarks import run_throughput_benchmarks
        result = run_throughput_benchmarks()
        
        print(f"Total tests: {result['summary']['total_tests']}")
        print(f"Passed: {result['summary']['passed_tests']}")
        print(f"Failed: {result['summary']['failed_tests']}")
        print(f"Success rate: {result['summary']['success_rate']:.2%}")
        
        return result
    except Exception as e:
        print(f"Error running throughput benchmarks: {e}")
        traceback.print_exc()
        return None

def main():
    """Run all benchmark suites."""
    print("HERALD Benchmarking Suite")
    print("Starting comprehensive benchmark tests...\n")
    
    results = {}
    
    # Run all benchmark suites
    results['performance'] = run_performance_benchmarks()
    results['model'] = run_model_benchmarks()
    results['memory'] = run_memory_benchmarks()
    results['throughput'] = run_throughput_benchmarks()
    
    # Summary
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    
    total_tests = 0
    total_passed = 0
    
    for suite_name, result in results.items():
        if result and 'summary' in result:
            tests = result['summary']['total_tests']
            passed = result['summary']['passed_tests']
            total_tests += tests
            total_passed += passed
            print(f"{suite_name.title()}: {passed}/{tests} passed")
    
    if total_tests > 0:
        overall_success_rate = total_passed / total_tests
        print(f"\nOverall: {total_passed}/{total_tests} tests passed ({overall_success_rate:.2%})")
    
    print("\nBenchmark suite completed!")

if __name__ == "__main__":
    main() 