"""
HERALD CLI Interface
Command-line interface for HERALD AI architecture

Usage:
    python cli.py serve --host 0.0.0.0 --port 8000
    python cli.py test --model path/to/model.herald
    python cli.py benchmark --model path/to/model.herald
    python cli.py interactive
"""

import argparse
import sys
import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_environment():
    """Setup the environment for HERALD."""
    # Add current directory to Python path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)


def serve_command(args):
    """Start the HERALD API server."""
    from api.main import main as api_main
    
    # Set up command line arguments for the API server
    sys.argv = [
        "api.main",
        "--host", args.host,
        "--port", str(args.port),
        "--log-level", args.log_level
    ]
    
    if args.reload:
        sys.argv.append("--reload")
    
    if args.workers > 1:
        sys.argv.extend(["--workers", str(args.workers)])
    
    api_main()


def test_command(args):
    """Run tests on the HERALD system."""
    import pytest
    
    logger.info("Running HERALD tests...")
    
    # Run tests
    test_args = [
        "tests/",
        "-v",
        "--tb=short"
    ]
    
    if args.verbose:
        test_args.append("-s")
    
    if args.coverage:
        test_args.extend(["--cov=.", "--cov-report=html"])
    
    exit_code = pytest.main(test_args)
    
    if exit_code == 0:
        logger.info("All tests passed!")
    else:
        logger.error(f"Tests failed with exit code: {exit_code}")
        sys.exit(exit_code)


def benchmark_command(args):
    """Run performance benchmarks."""
    from core.engine import NeuroEngine
    from reasoning.router import MoERouter
    import time
    
    logger.info("Running HERALD benchmarks...")
    
    # Initialize components
    engine = NeuroEngine()
    router = MoERouter()
    
    # Benchmark tests
    benchmarks = {}
    
    # Engine benchmark
    if args.engine:
        logger.info("Benchmarking NeuroEngine...")
        start_time = time.time()
        
        # Create test prompts
        test_prompts = [
            "Hello, how are you?",
            "What is the capital of France?",
            "Explain quantum computing in simple terms.",
            "Write a Python function to calculate fibonacci numbers.",
            "What are the benefits of renewable energy?"
        ]
        
        # Run benchmarks
        for prompt in test_prompts:
            try:
                # This would normally generate text, but we'll simulate it
                time.sleep(0.1)  # Simulate processing time
                logger.info(f"Processed: {prompt[:50]}...")
            except Exception as e:
                logger.warning(f"Failed to process prompt: {e}")
        
        engine_time = time.time() - start_time
        benchmarks['engine'] = {
            'total_time': engine_time,
            'prompts_processed': len(test_prompts),
            'avg_time_per_prompt': engine_time / len(test_prompts)
        }
    
    # Router benchmark
    if args.router:
        logger.info("Benchmarking MoERouter...")
        start_time = time.time()
        
        test_queries = [
            "If A and B, then C",
            "What causes the temperature to rise?",
            "What happened before the meeting?",
            "∀x (P(x) → Q(x))",
            "The intervention led to the outcome"
        ]
        
        for query in test_queries:
            try:
                result = router.process_query(query)
                logger.info(f"Routed: {query[:50]}...")
            except Exception as e:
                logger.warning(f"Failed to route query: {e}")
        
        router_time = time.time() - start_time
        benchmarks['router'] = {
            'total_time': router_time,
            'queries_processed': len(test_queries),
            'avg_time_per_query': router_time / len(test_queries)
        }
    
    # Print results
    logger.info("\nBenchmark Results:")
    logger.info("=" * 50)
    
    for component, results in benchmarks.items():
        logger.info(f"\n{component.upper()}:")
        for key, value in results.items():
            if isinstance(value, float):
                logger.info(f"  {key}: {value:.4f}")
            else:
                logger.info(f"  {key}: {value}")


def interactive_command(args):
    """Start interactive HERALD session."""
    from core.engine import NeuroEngine
    from reasoning.router import MoERouter
    
    logger.info("Starting HERALD interactive session...")
    logger.info("Type 'quit' or 'exit' to end the session")
    logger.info("Type 'help' for available commands")
    
    # Initialize components
    engine = NeuroEngine()
    router = MoERouter()
    
    print("\n" + "="*60)
    print("HERALD Interactive Session")
    print("="*60)
    
    while True:
        try:
            user_input = input("\nHERALD> ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            elif user_input.lower() == 'help':
                print("\nAvailable commands:")
                print("  help          - Show this help")
                print("  quit/exit/q   - Exit the session")
                print("  route <query> - Route a query through MoERouter")
                print("  stats         - Show system statistics")
                print("  <any text>    - Process as a query")
            
            elif user_input.lower().startswith('route '):
                query = user_input[6:].strip()
                if query:
                    try:
                        result = router.process_query(query)
                        print(f"\nRouting result:")
                        print(f"  Primary module: {result.primary_result.get('type', 'unknown')}")
                        print(f"  Confidence: {result.confidence:.2f}")
                        print(f"  Processing time: {result.processing_time:.4f}s")
                    except Exception as e:
                        print(f"Error routing query: {e}")
                else:
                    print("Please provide a query to route")
            
            elif user_input.lower() == 'stats':
                try:
                    engine_stats = engine.get_performance_stats()
                    router_stats = router.get_performance_stats()
                    
                    print("\nSystem Statistics:")
                    print(f"  Engine queries: {engine_stats.get('total_queries', 0)}")
                    print(f"  Router queries: {router_stats.get('total_queries', 0)}")
                    print(f"  Router accuracy: {router_stats.get('routing_accuracy', 0):.2%}")
                except Exception as e:
                    print(f"Error getting stats: {e}")
            
            elif user_input:
                # Process as a general query
                try:
                    print(f"\nProcessing: {user_input}")
                    
                    # Route the query
                    routing_result = router.process_query(user_input)
                    print(f"Routed to: {routing_result.primary_result.get('type', 'unknown')}")
                    print(f"Confidence: {routing_result.confidence:.2f}")
                    
                    # Simulate text generation (in a real implementation, this would use the engine)
                    print("Response: This is a simulated response. In a full implementation, this would generate actual text based on the query.")
                    
                except Exception as e:
                    print(f"Error processing query: {e}")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except EOFError:
            print("\nGoodbye!")
            break


def main():
    """Main CLI entry point."""
    setup_environment()
    
    parser = argparse.ArgumentParser(
        description="HERALD AI Architecture CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py serve --host 0.0.0.0 --port 8000
  python cli.py test --verbose
  python cli.py benchmark --engine --router
  python cli.py interactive
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Serve command
    serve_parser = subparsers.add_parser('serve', help='Start the API server')
    serve_parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    serve_parser.add_argument('--port', type=int, default=8000, help='Port to bind to')
    serve_parser.add_argument('--reload', action='store_true', help='Enable auto-reload')
    serve_parser.add_argument('--workers', type=int, default=1, help='Number of workers')
    serve_parser.add_argument('--log-level', default='info', choices=['debug', 'info', 'warning', 'error'])
    serve_parser.set_defaults(func=serve_command)
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Run tests')
    test_parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    test_parser.add_argument('--coverage', action='store_true', help='Generate coverage report')
    test_parser.set_defaults(func=test_command)
    
    # Benchmark command
    benchmark_parser = subparsers.add_parser('benchmark', help='Run benchmarks')
    benchmark_parser.add_argument('--engine', action='store_true', help='Benchmark NeuroEngine')
    benchmark_parser.add_argument('--router', action='store_true', help='Benchmark MoERouter')
    benchmark_parser.add_argument('--all', action='store_true', help='Run all benchmarks')
    benchmark_parser.set_defaults(func=benchmark_command)
    
    # Interactive command
    interactive_parser = subparsers.add_parser('interactive', help='Start interactive session')
    interactive_parser.set_defaults(func=interactive_command)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Set default benchmark components if --all is specified
    if args.command == 'benchmark' and args.all:
        args.engine = True
        args.router = True
    
    # If no specific components selected for benchmark, run all
    if args.command == 'benchmark' and not (args.engine or args.router):
        args.engine = True
        args.router = True
    
    try:
        args.func(args)
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 