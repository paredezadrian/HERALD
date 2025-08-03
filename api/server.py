"""
HERALD FastAPI Server Implementation
RESTful API interface for HERALD AI architecture

This module implements the main FastAPI server with:
- RESTful API interface
- Request/response handling
- Error handling and validation
- Performance monitoring
- Security middleware

Target: ~876 lines of production-ready API code
"""

import asyncio
import logging
import time
import uuid
import json
import hashlib
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from contextlib import asynccontextmanager
import threading
import traceback

# FastAPI and web framework imports
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, field_validator
import uvicorn

# Core HERALD imports
from core.engine import NeuroEngine, ModelConfig, InferenceConfig, ModelState
from core.tokenizer import ASCTokenizer
from core.memory import MultiTierMemoryManager
from reasoning.router import MoERouter
from reasoning.logic import LogicEngine
from reasoning.causal import CausalReasoningEngine as CausalEngine
from reasoning.temporal import TemporalLogicEngine as TemporalEngine

# Import API components
from .endpoints import HeraldEndpoints
from .middleware import HeraldMiddleware

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Security
security = HTTPBearer(auto_error=False)

# Global state
class ServerState:
    """Global server state management."""
    def __init__(self):
        self.engine: Optional[NeuroEngine] = None
        self.is_initialized: bool = False
        self.model_loaded: bool = False
        self.current_model_path: Optional[str] = None
        self.request_count: int = 0
        self.error_count: int = 0
        self.start_time: float = time.time()
        self.active_requests: Dict[str, Dict] = {}
        self.performance_stats: Dict[str, Any] = {}
        self.lock = threading.RLock()

# Global server state
server_state = ServerState()

# Pydantic models for API requests/responses
class HealthCheckResponse(BaseModel):
    """Health check response model."""
    status: str = Field(..., description="Server status")
    version: str = Field(..., description="HERALD version")
    uptime: float = Field(..., description="Server uptime in seconds")
    model_loaded: bool = Field(..., description="Whether a model is loaded")
    memory_usage: Dict[str, float] = Field(..., description="Memory usage statistics")
    performance_stats: Dict[str, Any] = Field(..., description="Performance statistics")

class ModelLoadRequest(BaseModel):
    """Model loading request model."""
    model_path: str = Field(..., description="Path to the .herald model file")
    config_overrides: Optional[Dict[str, Any]] = Field(None, description="Optional configuration overrides")
    
    @field_validator('model_path')
    @classmethod
    def validate_model_path(cls, v):
        if not v.endswith('.herald'):
            raise ValueError('Model path must point to a .herald file')
        return v

class ModelLoadResponse(BaseModel):
    """Model loading response model."""
    success: bool = Field(..., description="Whether model loading was successful")
    model_name: Optional[str] = Field(None, description="Name of the loaded model")
    model_version: Optional[str] = Field(None, description="Version of the loaded model")
    load_time: Optional[float] = Field(None, description="Model loading time in seconds")
    memory_usage: Optional[Dict[str, float]] = Field(None, description="Memory usage after loading")
    error_message: Optional[str] = Field(None, description="Error message if loading failed")

class InferenceRequest(BaseModel):
    """Inference request model."""
    prompt: str = Field(..., description="Input prompt for generation", min_length=1, max_length=100000)
    max_new_tokens: Optional[int] = Field(50, description="Maximum number of tokens to generate", ge=1, le=1000)
    temperature: Optional[float] = Field(0.7, description="Sampling temperature", ge=0.0, le=2.0)
    top_p: Optional[float] = Field(0.9, description="Top-p sampling parameter", ge=0.0, le=1.0)
    top_k: Optional[int] = Field(50, description="Top-k sampling parameter", ge=1, le=1000)
    repetition_penalty: Optional[float] = Field(1.1, description="Repetition penalty", ge=0.5, le=2.0)
    stream: Optional[bool] = Field(False, description="Whether to stream the response")
    reasoning_mode: Optional[str] = Field("auto", description="Reasoning mode: auto, logic, causal, temporal")
    
    @field_validator('prompt')
    @classmethod
    def validate_prompt(cls, v):
        if not v.strip():
            raise ValueError('Prompt cannot be empty')
        return v.strip()

class InferenceResponse(BaseModel):
    """Inference response model."""
    generated_text: str = Field(..., description="Generated text")
    tokens_generated: int = Field(..., description="Number of tokens generated")
    generation_time: float = Field(..., description="Generation time in seconds")
    tokens_per_second: float = Field(..., description="Tokens generated per second")
    memory_usage: Dict[str, float] = Field(..., description="Memory usage during generation")
    reasoning_stats: Optional[Dict[str, Any]] = Field(None, description="Reasoning module statistics")

class BatchInferenceRequest(BaseModel):
    """Batch inference request model."""
    prompts: List[str] = Field(..., description="List of prompts to process", min_items=1, max_items=10)
    max_new_tokens: Optional[int] = Field(50, description="Maximum number of tokens to generate per prompt")
    temperature: Optional[float] = Field(0.7, description="Sampling temperature")
    top_p: Optional[float] = Field(0.9, description="Top-p sampling parameter")
    top_k: Optional[int] = Field(50, description="Top-k sampling parameter")
    repetition_penalty: Optional[float] = Field(1.1, description="Repetition penalty")

class BatchInferenceResponse(BaseModel):
    """Batch inference response model."""
    results: List[Dict[str, Any]] = Field(..., description="List of generation results")
    total_time: float = Field(..., description="Total processing time")
    average_tokens_per_second: float = Field(..., description="Average tokens per second")

class PerformanceStatsResponse(BaseModel):
    """Performance statistics response model."""
    request_count: int = Field(..., description="Total number of requests")
    error_count: int = Field(..., description="Total number of errors")
    average_response_time: float = Field(..., description="Average response time")
    memory_usage: Dict[str, float] = Field(..., description="Current memory usage")
    model_stats: Optional[Dict[str, Any]] = Field(None, description="Model-specific statistics")

class ErrorResponse(BaseModel):
    """Error response model."""
    error: str = Field(..., description="Error message")
    error_code: str = Field(..., description="Error code")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: float = Field(..., description="Error timestamp")

# Custom exceptions
class HeraldAPIError(Exception):
    """Base exception for HERALD API errors."""
    def __init__(self, message: str, error_code: str = "INTERNAL_ERROR", details: Optional[Dict] = None):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(message)

class ModelNotLoadedError(HeraldAPIError):
    """Exception raised when no model is loaded."""
    def __init__(self):
        super().__init__("No model is currently loaded", "MODEL_NOT_LOADED")

class InvalidRequestError(HeraldAPIError):
    """Exception raised for invalid requests."""
    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(message, "INVALID_REQUEST", details)

class ModelLoadError(HeraldAPIError):
    """Exception raised when model loading fails."""
    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(message, "MODEL_LOAD_ERROR", details)

# Dependency functions
async def get_server_state() -> ServerState:
    """Get the global server state."""
    return server_state

async def verify_model_loaded(state: ServerState = Depends(get_server_state)) -> NeuroEngine:
    """Verify that a model is loaded and return the engine."""
    if not state.model_loaded or state.engine is None:
        raise ModelNotLoadedError()
    return state.engine

async def get_authorization(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> Optional[str]:
    """Get authorization token if provided."""
    return credentials.credentials if credentials else None

# Background tasks
async def update_performance_stats():
    """Update performance statistics in the background."""
    while True:
        try:
            with server_state.lock:
                if server_state.engine:
                    stats = server_state.engine.get_performance_stats()
                    server_state.performance_stats.update(stats)
        except Exception as e:
            logger.error(f"Error updating performance stats: {e}")
        
        await asyncio.sleep(30)  # Update every 30 seconds

async def cleanup_expired_requests():
    """Clean up expired request tracking."""
    while True:
        try:
            current_time = time.time()
            with server_state.lock:
                expired_requests = [
                    req_id for req_id, req_data in server_state.active_requests.items()
                    if current_time - req_data['start_time'] > 300  # 5 minutes
                ]
                for req_id in expired_requests:
                    del server_state.active_requests[req_id]
        except Exception as e:
            logger.error(f"Error cleaning up expired requests: {e}")
        
        await asyncio.sleep(60)  # Clean up every minute

# Application lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting HERALD API Server...")
    
    # Start background tasks
    asyncio.create_task(update_performance_stats())
    asyncio.create_task(cleanup_expired_requests())
    
    yield
    
    # Shutdown
    logger.info("Shutting down HERALD API Server...")
    if server_state.engine:
        server_state.engine.clear_cache()

# Create FastAPI application
app = FastAPI(
    title="HERALD API",
    description="Hybrid Efficient Reasoning Architecture for Local Deployment - RESTful API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# Add custom middleware
app.add_middleware(HeraldMiddleware, enable_security=False, enable_rate_limiting=False)

# Create endpoints instance
endpoints = HeraldEndpoints(server_state)
app.include_router(endpoints.get_router())

# Root endpoint
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "HERALD API",
        "version": "1.0.0",
        "description": "Hybrid Efficient Reasoning Architecture for Local Deployment",
        "docs": "/docs",
        "health": "/health"
    }

# Health check endpoint
@app.get("/health", response_model=HealthCheckResponse)
async def health_check(state: ServerState = Depends(get_server_state)):
    """Health check endpoint."""
    import psutil
    
    uptime = time.time() - state.start_time
    memory_usage = {
        "total": psutil.virtual_memory().total / (1024**3),  # GB
        "available": psutil.virtual_memory().available / (1024**3),  # GB
        "used": psutil.virtual_memory().used / (1024**3),  # GB
        "percent": psutil.virtual_memory().percent
    }
    
    return HealthCheckResponse(
        status="healthy",
        version="1.0.0",
        uptime=uptime,
        model_loaded=state.model_loaded,
        memory_usage=memory_usage,
        performance_stats=state.performance_stats
    )

# Model management endpoints
@app.post("/models/load", response_model=ModelLoadResponse)
async def load_model(
    request: ModelLoadRequest,
    background_tasks: BackgroundTasks,
    state: ServerState = Depends(get_server_state)
):
    """Load a HERALD model."""
    try:
        start_time = time.time()
        
        # Validate model file exists
        model_path = Path(request.model_path)
        if not model_path.exists():
            raise ModelLoadError(f"Model file not found: {request.model_path}")
        
        # Initialize engine if not already done
        if not state.is_initialized:
            logger.info("Initializing HERALD engine...")
            state.engine = NeuroEngine()
            state.is_initialized = True
        
        # Load the model
        logger.info(f"Loading model from: {request.model_path}")
        success = state.engine.load_model(request.model_path)
        
        if not success:
            raise ModelLoadError("Failed to load model")
        
        load_time = time.time() - start_time
        state.model_loaded = True
        state.current_model_path = request.model_path
        
        # Get memory usage
        memory_stats = {}
        if state.engine:
            stats = state.engine.get_performance_stats()
            memory_stats = stats.get('memory_usage', {})
        
        return ModelLoadResponse(
            success=True,
            model_name=state.engine.model_state.model_config.model_name,
            model_version=state.engine.model_state.model_config.model_version,
            load_time=load_time,
            memory_usage=memory_stats
        )
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        state.model_loaded = False
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/models/unload")
async def unload_model(state: ServerState = Depends(get_server_state)):
    """Unload the currently loaded model."""
    try:
        if state.engine:
            state.engine.clear_cache()
        
        state.model_loaded = False
        state.current_model_path = None
        
        return {"success": True, "message": "Model unloaded successfully"}
        
    except Exception as e:
        logger.error(f"Error unloading model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models/status")
async def get_model_status(state: ServerState = Depends(get_server_state)):
    """Get the status of the currently loaded model."""
    if not state.model_loaded or not state.engine:
        return {
            "model_loaded": False,
            "model_path": None,
            "model_name": None,
            "model_version": None,
            "architecture": None,
            "current_context_length": 0,
            "performance_stats": {}
        }
    
    engine = state.engine
    model_config = engine.model_state.model_config
    
    return {
        "model_loaded": True,
        "model_path": state.current_model_path,
        "model_name": model_config.model_name,
        "model_version": model_config.model_version,
        "architecture": model_config.architecture,
        "current_context_length": engine.model_state.current_context_length,
        "performance_stats": engine.get_performance_stats()
    }

# Inference endpoints
@app.post("/inference/generate", response_model=InferenceResponse)
async def generate_text(
    request: InferenceRequest,
    engine: NeuroEngine = Depends(verify_model_loaded),
    state: ServerState = Depends(get_server_state)
):
    """Generate text using the loaded model."""
    try:
        start_time = time.time()
        
        # Update request tracking
        request_id = str(uuid.uuid4())
        with state.lock:
            state.request_count += 1
            state.active_requests[request_id] = {
                'start_time': start_time,
                'prompt_length': len(request.prompt),
                'max_tokens': request.max_new_tokens
            }
        
        # Generate text
        generated_text = engine.generate(
            prompt=request.prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            repetition_penalty=request.repetition_penalty
        )
        
        generation_time = time.time() - start_time
        tokens_generated = len(generated_text.split())  # Approximate
        
        # Calculate tokens per second
        tokens_per_second = tokens_generated / generation_time if generation_time > 0 else 0
        
        # Get memory usage
        memory_stats = engine.get_performance_stats().get('memory_usage', {})
        
        # Clean up request tracking
        with state.lock:
            if request_id in state.active_requests:
                del state.active_requests[request_id]
        
        return InferenceResponse(
            generated_text=generated_text,
            tokens_generated=tokens_generated,
            generation_time=generation_time,
            tokens_per_second=tokens_per_second,
            memory_usage=memory_stats
        )
        
    except Exception as e:
        logger.error(f"Error during text generation: {e}")
        with state.lock:
            state.error_count += 1
        
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/inference/generate/stream")
async def generate_text_stream(
    request: InferenceRequest,
    engine: NeuroEngine = Depends(verify_model_loaded),
    state: ServerState = Depends(get_server_state)
):
    """Generate text with streaming response."""
    if not request.stream:
        raise HTTPException(status_code=400, detail="Stream parameter must be True for streaming endpoint")
    
    try:
        start_time = time.time()
        
        # Update request tracking
        request_id = str(uuid.uuid4())
        with state.lock:
            state.request_count += 1
            state.active_requests[request_id] = {
                'start_time': start_time,
                'prompt_length': len(request.prompt),
                'max_tokens': request.max_new_tokens
            }
        
        async def generate_stream():
            try:
                # For now, we'll simulate streaming by generating in chunks
                # In a real implementation, the engine would support streaming
                full_text = engine.generate(
                    prompt=request.prompt,
                    max_new_tokens=request.max_new_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    top_k=request.top_k,
                    repetition_penalty=request.repetition_penalty
                )
                
                # Split into chunks for streaming
                words = full_text.split()
                chunk_size = max(1, len(words) // 10)  # 10 chunks
                
                for i in range(0, len(words), chunk_size):
                    chunk = " ".join(words[i:i + chunk_size])
                    yield f"data: {json.dumps({'text': chunk, 'finished': False})}\n\n"
                    await asyncio.sleep(0.1)  # Small delay for streaming effect
                
                # Send completion signal
                yield f"data: {json.dumps({'text': '', 'finished': True, 'total_time': time.time() - start_time})}\n\n"
                
            except Exception as e:
                logger.error(f"Error in streaming generation: {e}")
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
            finally:
                # Clean up request tracking
                with state.lock:
                    if request_id in state.active_requests:
                        del state.active_requests[request_id]
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/plain",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
        )
        
    except Exception as e:
        logger.error(f"Error setting up streaming generation: {e}")
        with state.lock:
            state.error_count += 1
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/inference/batch", response_model=BatchInferenceResponse)
async def batch_generate(
    request: BatchInferenceRequest,
    engine: NeuroEngine = Depends(verify_model_loaded),
    state: ServerState = Depends(get_server_state)
):
    """Generate text for multiple prompts in batch."""
    try:
        start_time = time.time()
        results = []
        
        # Update request tracking
        request_id = str(uuid.uuid4())
        with state.lock:
            state.request_count += 1
            state.active_requests[request_id] = {
                'start_time': start_time,
                'batch_size': len(request.prompts)
            }
        
        for i, prompt in enumerate(request.prompts):
            try:
                prompt_start = time.time()
                
                generated_text = engine.generate(
                    prompt=prompt,
                    max_new_tokens=request.max_new_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    top_k=request.top_k,
                    repetition_penalty=request.repetition_penalty
                )
                
                prompt_time = time.time() - prompt_start
                tokens_generated = len(generated_text.split())
                
                results.append({
                    "prompt_index": i,
                    "generated_text": generated_text,
                    "tokens_generated": tokens_generated,
                    "generation_time": prompt_time,
                    "success": True
                })
                
            except Exception as e:
                logger.error(f"Error generating for prompt {i}: {e}")
                results.append({
                    "prompt_index": i,
                    "error": str(e),
                    "success": False
                })
        
        total_time = time.time() - start_time
        
        # Calculate average tokens per second
        successful_results = [r for r in results if r.get('success', False)]
        if successful_results:
            total_tokens = sum(r['tokens_generated'] for r in successful_results)
            avg_tokens_per_second = total_tokens / total_time
        else:
            avg_tokens_per_second = 0
        
        # Clean up request tracking
        with state.lock:
            if request_id in state.active_requests:
                del state.active_requests[request_id]
        
        return BatchInferenceResponse(
            results=results,
            total_time=total_time,
            average_tokens_per_second=avg_tokens_per_second
        )
        
    except Exception as e:
        logger.error(f"Error in batch generation: {e}")
        with state.lock:
            state.error_count += 1
        raise HTTPException(status_code=500, detail=str(e))

# Performance and monitoring endpoints
@app.get("/performance/stats", response_model=PerformanceStatsResponse)
async def get_performance_stats(state: ServerState = Depends(get_server_state)):
    """Get performance statistics."""
    try:
        import psutil
        
        # Calculate average response time
        uptime = time.time() - state.start_time
        avg_response_time = uptime / max(state.request_count, 1)
        
        # Get memory usage
        memory_usage = {
            "total": psutil.virtual_memory().total / (1024**3),  # GB
            "available": psutil.virtual_memory().available / (1024**3),  # GB
            "used": psutil.virtual_memory().used / (1024**3),  # GB
            "percent": psutil.virtual_memory().percent
        }
        
        # Get model stats if available
        model_stats = None
        if state.engine:
            model_stats = state.engine.get_performance_stats()
        
        return PerformanceStatsResponse(
            request_count=state.request_count,
            error_count=state.error_count,
            average_response_time=avg_response_time,
            memory_usage=memory_usage,
            model_stats=model_stats
        )
        
    except Exception as e:
        logger.error(f"Error getting performance stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/performance/active-requests")
async def get_active_requests(state: ServerState = Depends(get_server_state)):
    """Get information about active requests."""
    try:
        with state.lock:
            active_requests = list(state.active_requests.values())
        
        return {
            "active_requests": len(active_requests),
            "requests": active_requests
        }
        
    except Exception as e:
        logger.error(f"Error getting active requests: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/inference/config")
async def get_inference_config(engine: NeuroEngine = Depends(verify_model_loaded)):
    """Get current inference configuration."""
    try:
        config = engine.inference_config
        return {
            "max_new_tokens": config.max_new_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "top_k": config.top_k,
            "repetition_penalty": config.repetition_penalty,
            "do_sample": config.do_sample
        }
    except Exception as e:
        logger.error(f"Error getting inference config: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/test-error")
async def test_error():
    """Test endpoint for error handling."""
    raise HTTPException(status_code=500, detail="Test error for error handling")

@app.get("/metrics")
async def get_metrics():
    """Get Prometheus metrics."""
    try:
        import psutil
        
        metrics = []
        metrics.append("# HELP herald_requests_total Total number of requests")
        metrics.append("# TYPE herald_requests_total counter")
        metrics.append(f"herald_requests_total {server_state.request_count}")
        
        metrics.append("# HELP herald_errors_total Total number of errors")
        metrics.append("# TYPE herald_errors_total counter")
        metrics.append(f"herald_errors_total {server_state.error_count}")
        
        metrics.append("# HELP herald_memory_usage_bytes Memory usage in bytes")
        metrics.append("# TYPE herald_memory_usage_bytes gauge")
        memory = psutil.virtual_memory()
        metrics.append(f"herald_memory_usage_bytes {memory.used}")
        
        metrics.append("# HELP herald_model_loaded Model loaded status")
        metrics.append("# TYPE herald_model_loaded gauge")
        metrics.append(f"herald_model_loaded {1 if server_state.model_loaded else 0}")
        
        return Response(content="\n".join(metrics), media_type="text/plain")
        
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ready")
async def readiness_probe():
    """Kubernetes readiness probe endpoint."""
    try:
        # Check if server is ready to receive traffic
        return {"ready": True, "status": "ready"}
    except Exception as e:
        logger.error(f"Readiness probe failed: {e}")
        raise HTTPException(status_code=503, detail="Service not ready")

@app.get("/live")
async def liveness_probe():
    """Kubernetes liveness probe endpoint."""
    try:
        # Check if server is alive
        return {"alive": True, "status": "alive"}
    except Exception as e:
        logger.error(f"Liveness probe failed: {e}")
        raise HTTPException(status_code=503, detail="Service not alive")

# Error handling
@app.exception_handler(HeraldAPIError)
async def herald_api_exception_handler(request: Request, exc: HeraldAPIError):
    """Handle HERALD API exceptions."""
    return JSONResponse(
        status_code=400,
        content=ErrorResponse(
            error=exc.message,
            error_code=exc.error_code,
            details=exc.details,
            timestamp=time.time()
        ).dict()
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}")
    logger.error(traceback.format_exc())
    
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            error_code="INTERNAL_ERROR",
            details={"exception_type": type(exc).__name__},
            timestamp=time.time()
        ).dict()
    )

# Main function for running the server
def run_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    reload: bool = False,
    workers: int = 1
):
    """Run the HERALD API server."""
    logger.info(f"Starting HERALD API server on {host}:{port}")
    
    uvicorn.run(
        "api.server:app",
        host=host,
        port=port,
        reload=reload,
        workers=workers,
        log_level="info"
    )

if __name__ == "__main__":
    run_server() 