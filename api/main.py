"""
HERALD FastAPI Server Main Entry Point
Main entry point for running the HERALD API server

Usage:
    python -m api.main
    uvicorn api.main:app --host 0.0.0.0 --port 8000
"""

import uvicorn
import argparse
import sys
import os

# Add the parent directory to the path so we can import HERALD modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.server import app, run_server
from core.engine import NeuroEngine
from reasoning.router import MoERouter
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point for the HERALD API server."""
    parser = argparse.ArgumentParser(description="HERALD FastAPI Server")
    parser.add_argument(
        "--host", 
        default="0.0.0.0", 
        help="Host to bind to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=8000, 
        help="Port to bind to (default: 8000)"
    )
    parser.add_argument(
        "--reload", 
        action="store_true", 
        help="Enable auto-reload for development"
    )
    parser.add_argument(
        "--workers", 
        type=int, 
        default=1, 
        help="Number of worker processes (default: 1)"
    )
    parser.add_argument(
        "--log-level", 
        default="info", 
        choices=["debug", "info", "warning", "error"],
        help="Log level (default: info)"
    )
    
    args = parser.parse_args()
    
    logger.info(f"Starting HERALD API server on {args.host}:{args.port}")
    logger.info(f"Log level: {args.log_level}")
    logger.info(f"Workers: {args.workers}")
    logger.info(f"Auto-reload: {args.reload}")
    
    try:
        # Run the server
        run_server(
            host=args.host,
            port=args.port,
            reload=args.reload,
            workers=args.workers
        )
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 