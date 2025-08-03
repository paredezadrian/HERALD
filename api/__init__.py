"""
HERALD API Module
RESTful API interface and endpoints

This module contains the API components:
- Server: FastAPI server implementation
- Endpoints: REST API endpoints
- Middleware: Request processing middleware
"""

__version__ = "1.0.0"
__author__ = "HERALD Development Team"

# API components
from .server import app, run_server
from .endpoints import HeraldEndpoints
from .middleware import HeraldMiddleware

__all__ = [
    "app",
    "run_server",
    "HeraldEndpoints", 
    "HeraldMiddleware"
] 