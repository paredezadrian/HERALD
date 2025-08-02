#!/usr/bin/env python3
"""
Integration test runner for HERALD.

This script runs all integration tests and provides a comprehensive report
of the system's performance and functionality.
"""

import sys
import os
import time
import subprocess
import json
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

def run_integration_tests():
    """Run all integration tests and generate a report."""
    print("🚀 Starting HERALD Integration Testing")
    print("=" * 50)
    
    # Get the project root directory
    project_root = Path(__file__).parent.parent.parent
    tests_dir = project_root / "tests" / "integration"
    
    # List of test files to run
    test_files = [
        "test_pipeline_integration.py",
        "test_performance_benchmarks.py"
    ]
    
    results = {}
    total_start_time = time.time()
    
    for test_file in test_files:
        test_path = tests_dir / test_file
        if not test_path.exists():
            print(f"⚠️  Warning: Test file {test_file} not found")
            continue
            
        print(f"\n📋 Running {test_file}...")
        print("-" * 30)
        
        # Run the test file
        start_time = time.time()
        try:
            result = subprocess.run([
                sys.executable, "-m", "pytest", str(test_path),
                "-v", "-s", "--tb=short"
            ], capture_output=True, text=True, cwd=project_root)
            
            test_time = time.time() - start_time
            
            # Parse results
            if result.returncode == 0:
                status = "✅ PASSED"
                print(f"✅ {test_file} completed successfully")
            else:
                status = "❌ FAILED"
                print(f"❌ {test_file} failed")
                print(f"Error output:\n{result.stderr}")
            
            results[test_file] = {
                "status": status,
                "time": test_time,
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr
            }
            
        except Exception as e:
            status = "❌ ERROR"
            results[test_file] = {
                "status": status,
                "time": time.time() - start_time,
                "error": str(e)
            }
            print(f"❌ Error running {test_file}: {e}")
    
    total_time = time.time() - total_start_time
    
    # Generate report
    print("\n" + "=" * 50)
    print("📊 INTEGRATION TEST REPORT")
    print("=" * 50)
    
    passed = 0
    failed = 0
    
    for test_file, result in results.items():
        print(f"\n{test_file}:")
        print(f"  Status: {result['status']}")
        print(f"  Time: {result['time']:.2f}s")
        
        if result['status'] == "✅ PASSED":
            passed += 1
        else:
            failed += 1
    
    print(f"\n📈 SUMMARY:")
    print(f"  Total tests: {len(results)}")
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")
    print(f"  Total time: {total_time:.2f}s")
    
    if failed == 0:
        print("\n🎉 All integration tests passed!")
        return True
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please review the output above.")
        return False


def run_performance_validation():
    """Run specific performance validation tests."""
    print("\n🔍 Running Performance Validation...")
    print("-" * 30)
    
    try:
        # Import and run performance tests
        from test_performance_benchmarks import TestPerformanceBenchmarks
        
        # Create test instance
        test_instance = TestPerformanceBenchmarks()
        
        # Run key performance tests
        print("Testing token generation speed...")
        test_instance.test_token_generation_speed(
            test_instance.tokenizer(),
            test_instance.engine()
        )
        
        print("Testing memory usage limits...")
        test_instance.test_memory_usage_limits(
            test_instance.tokenizer(),
            test_instance.memory_manager(),
            test_instance.engine()
        )
        
        print("Testing model load time...")
        test_instance.test_model_load_time()
        
        print("✅ Performance validation completed")
        return True
        
    except Exception as e:
        print(f"❌ Performance validation failed: {e}")
        return False


def generate_test_report(results):
    """Generate a detailed test report."""
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_tests": len(results),
        "passed": sum(1 for r in results.values() if r['status'] == "✅ PASSED"),
        "failed": sum(1 for r in results.values() if r['status'] != "✅ PASSED"),
        "results": results
    }
    
    # Save report to file
    report_file = Path(__file__).parent / "integration_test_report.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📄 Detailed report saved to: {report_file}")
    return report


if __name__ == "__main__":
    print("HERALD Integration Test Suite")
    print("=" * 50)
    
    # Run integration tests
    success = run_integration_tests()
    
    # Run performance validation
    perf_success = run_performance_validation()
    
    # Generate report
    if 'results' in locals():
        generate_test_report(results)
    
    # Final status
    if success and perf_success:
        print("\n🎉 All tests completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Please review the output above.")
        sys.exit(1) 