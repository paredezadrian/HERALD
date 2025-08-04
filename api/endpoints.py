"""
HERALD API Endpoints Implementation
REST API endpoints for HERALD AI architecture

This module implements the API endpoints with:
- Inference endpoints
- Model management endpoints
- Health check endpoints
- Performance monitoring endpoints

Target: ~654 lines of endpoint implementations
"""

import asyncio
import logging
import time
import uuid
import json
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import threading

# FastAPI imports
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, field_validator
import uvicorn

# Core HERALD imports
from core.engine import NeuroEngine, ModelConfig, InferenceConfig
from core.tokenizer import ASCTokenizer
from core.memory import MultiTierMemoryManager
from reasoning.router import MoERouter
from reasoning.logic import LogicEngine
from reasoning.causal import CausalReasoningEngine as CausalEngine
from reasoning.temporal import TemporalLogicEngine as TemporalEngine

logger = logging.getLogger(__name__)

# Security
security = HTTPBearer(auto_error=False)

# Pydantic models for additional endpoints
class TokenizeRequest(BaseModel):
    """Tokenization request model."""
    text: str = Field(..., description="Text to tokenize", min_length=1, max_length=100000)
    include_metadata: Optional[bool] = Field(False, description="Include token metadata")
    
    @field_validator('text')
    @classmethod
    def validate_text(cls, v):
        if not v.strip():
            raise ValueError('Text cannot be empty')
        return v.strip()

class TokenizeResponse(BaseModel):
    """Tokenization response model."""
    tokens: List[int] = Field(..., description="Token IDs")
    token_texts: List[str] = Field(..., description="Token texts")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Token metadata")
    token_count: int = Field(..., description="Number of tokens")
    compression_ratio: float = Field(..., description="Compression ratio achieved")

class ReasoningRequest(BaseModel):
    """Reasoning request model."""
    query: str = Field(..., description="Query for reasoning", min_length=1, max_length=10000)
    reasoning_type: str = Field(..., description="Type of reasoning: logic, causal, temporal, auto")
    context: Optional[str] = Field(None, description="Additional context")
    max_steps: Optional[int] = Field(10, description="Maximum reasoning steps", ge=1, le=100)
    
    @field_validator('reasoning_type')
    @classmethod
    def validate_reasoning_type(cls, v):
        valid_types = ['logic', 'causal', 'temporal', 'auto']
        if v not in valid_types:
            raise ValueError(f'Reasoning type must be one of: {valid_types}')
        return v

class ReasoningResponse(BaseModel):
    """Reasoning response model."""
    result: str = Field(..., description="Reasoning result")
    reasoning_steps: List[Dict[str, Any]] = Field(..., description="Detailed reasoning steps")
    confidence: float = Field(..., description="Confidence score", ge=0.0, le=1.0)
    reasoning_time: float = Field(..., description="Reasoning time in seconds")
    module_used: str = Field(..., description="Reasoning module used")

class ModelConfigRequest(BaseModel):
    """Model configuration request model."""
    config_updates: Dict[str, Any] = Field(..., description="Configuration updates to apply")
    
    @field_validator('config_updates')
    @classmethod
    def validate_config_updates(cls, v):
        valid_keys = [
            'temperature', 'top_p', 'top_k', 'repetition_penalty',
            'max_new_tokens', 'do_sample'
        ]
        for key in v.keys():
            if key not in valid_keys:
                raise ValueError(f'Invalid config key: {key}')
        return v

class ModelConfigResponse(BaseModel):
    """Model configuration response model."""
    success: bool = Field(..., description="Whether configuration update was successful")
    current_config: Dict[str, Any] = Field(..., description="Current model configuration")
    applied_changes: List[str] = Field(..., description="List of applied configuration changes")

class MemoryStatsResponse(BaseModel):
    """Memory statistics response model."""
    active_memory_usage: Dict[str, Any] = Field(..., description="Active memory usage")
    compressed_memory_usage: Dict[str, Any] = Field(..., description="Compressed memory usage")
    archived_memory_usage: Dict[str, Any] = Field(..., description="Archived memory usage")
    total_memory_usage: Dict[str, Any] = Field(..., description="Total memory usage")
    memory_efficiency: float = Field(..., description="Memory efficiency ratio")

class SystemInfoResponse(BaseModel):
    """System information response model."""
    system_info: Dict[str, Any] = Field(..., description="System information")
    hardware_info: Dict[str, Any] = Field(..., description="Hardware information")
    performance_info: Dict[str, Any] = Field(..., description="Performance information")
    model_info: Optional[Dict[str, Any]] = Field(None, description="Model information")

class HeraldEndpoints:
    """HERALD API endpoints implementation."""
    
    def __init__(self, server_state):
        """Initialize endpoints with server state."""
        self.server_state = server_state
        self.router = APIRouter(prefix="/api/v1", tags=["HERALD API"])
        self._setup_routes()
    
    def _setup_routes(self):
        """Set up all API routes."""
        
        # Tokenization endpoints
        @self.router.post("/tokenize", response_model=TokenizeResponse)
        async def tokenize_text(
            request: TokenizeRequest,
            engine: NeuroEngine = Depends(self._verify_model_loaded)
        ):
            """Tokenize text using the loaded model's tokenizer."""
            try:
                start_time = time.time()
                
                # Get tokenizer from engine
                tokenizer = engine.model_state.tokenizer
                if not tokenizer:
                    raise HTTPException(status_code=500, detail="Tokenizer not available")
                
                # Tokenize text
                result = tokenizer.tokenize(request.text)
                
                tokenize_time = time.time() - start_time
                
                # Prepare response using actual TokenizationResult fields
                token_ids = [t.token_id for t in result.tokens]
                token_texts = [t.text for t in result.tokens]

                response_data = {
                    "tokens": token_ids,
                    "token_texts": token_texts,
                    "token_count": len(token_ids),
                    "compression_ratio": result.compression_ratio,
                }

                if request.include_metadata:
                    response_data["metadata"] = {
                        "tokenization_time": tokenize_time,
                        "compression_ratio": result.compression_ratio,
                        "tier_usage": result.tier_distribution,
                    }

                return TokenizeResponse(**response_data)
                
            except Exception as e:
                logger.error(f"Error during tokenization: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # Reasoning endpoints
        @self.router.post("/reasoning", response_model=ReasoningResponse)
        async def perform_reasoning(
            request: ReasoningRequest,
            engine: NeuroEngine = Depends(self._verify_model_loaded)
        ):
            """Perform reasoning using the appropriate reasoning module."""
            try:
                start_time = time.time()
                
                # Determine reasoning module to use
                if request.reasoning_type == "auto":
                    # Use MoE router to determine best module
                    router = MoERouter()
                    module_decision = router.route_query(request.query)
                    module_type = module_decision.primary_module.value
                else:
                    module_type = request.reasoning_type

                # Initialize appropriate reasoning engine
                if module_type == "logic":
                    reasoning_engine = LogicEngine()
                elif module_type == "causal":
                    reasoning_engine = CausalEngine()
                elif module_type == "temporal":
                    reasoning_engine = TemporalEngine()
                else:
                    raise HTTPException(status_code=400, detail=f"Unknown reasoning type: {module_type}")
                
                # Perform reasoning
                result = reasoning_engine.reason(
                    query=request.query,
                    context=request.context,
                    max_steps=request.max_steps
                )
                
                reasoning_time = time.time() - start_time
                
                return ReasoningResponse(
                    result=result.get('result', ''),
                    reasoning_steps=result.get('steps', []),
                    confidence=result.get('confidence', 0.0),
                    reasoning_time=reasoning_time,
                    module_used=module_type
                )
                
            except Exception as e:
                logger.error(f"Error during reasoning: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # Model configuration endpoints
        @self.router.put("/models/config", response_model=ModelConfigResponse)
        async def update_model_config(
            request: ModelConfigRequest,
            engine: NeuroEngine = Depends(self._verify_model_loaded)
        ):
            """Update model configuration parameters."""
            try:
                applied_changes = []
                current_config = engine.model_state.inference_config.__dict__.copy()
                
                # Apply configuration updates
                for key, value in request.config_updates.items():
                    if hasattr(engine.model_state.inference_config, key):
                        setattr(engine.model_state.inference_config, key, value)
                        applied_changes.append(f"Updated {key} to {value}")
                    else:
                        logger.warning(f"Invalid config key: {key}")
                
                # Get updated configuration
                updated_config = engine.model_state.inference_config.__dict__.copy()
                
                return ModelConfigResponse(
                    success=len(applied_changes) > 0,
                    current_config=updated_config,
                    applied_changes=applied_changes
                )
                
            except Exception as e:
                logger.error(f"Error updating model config: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.router.get("/models/config")
        async def get_model_config(engine: NeuroEngine = Depends(self._verify_model_loaded)):
            """Get current model configuration."""
            try:
                model_config = engine.model_state.model_config.__dict__.copy()
                inference_config = engine.model_state.inference_config.__dict__.copy()
                
                return {
                    "model_config": model_config,
                    "inference_config": inference_config
                }
                
            except Exception as e:
                logger.error(f"Error getting model config: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # Memory management endpoints
        @self.router.get("/memory/stats", response_model=MemoryStatsResponse)
        async def get_memory_stats(engine: NeuroEngine = Depends(self._verify_model_loaded)):
            """Get detailed memory statistics."""
            try:
                memory_manager = engine.model_state.memory_manager
                if not memory_manager:
                    raise HTTPException(status_code=500, detail="Memory manager not available")
                
                # Get memory statistics
                active_stats = memory_manager.get_active_memory_stats()
                compressed_stats = memory_manager.get_compressed_memory_stats()
                archived_stats = memory_manager.get_archived_memory_stats()
                total_stats = memory_manager.get_total_memory_stats()
                
                # Calculate efficiency
                efficiency = memory_manager.get_memory_efficiency()
                
                return MemoryStatsResponse(
                    active_memory_usage=active_stats,
                    compressed_memory_usage=compressed_stats,
                    archived_memory_usage=archived_stats,
                    total_memory_usage=total_stats,
                    memory_efficiency=efficiency
                )
                
            except Exception as e:
                logger.error(f"Error getting memory stats: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.router.post("/memory/clear")
        async def clear_memory(engine: NeuroEngine = Depends(self._verify_model_loaded)):
            """Clear all memory tiers."""
            try:
                memory_manager = engine.model_state.memory_manager
                if memory_manager:
                    memory_manager.clear_all_memory()
                
                return {"success": True, "message": "Memory cleared successfully"}
                
            except Exception as e:
                logger.error(f"Error clearing memory: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.router.post("/memory/optimize")
        async def optimize_memory(engine: NeuroEngine = Depends(self._verify_model_loaded)):
            """Optimize memory usage."""
            try:
                memory_manager = engine.model_state.memory_manager
                if memory_manager:
                    optimization_result = memory_manager.optimize_memory()
                    return {
                        "success": True,
                        "message": "Memory optimized successfully",
                        "optimization_stats": optimization_result
                    }
                else:
                    raise HTTPException(status_code=500, detail="Memory manager not available")
                
            except Exception as e:
                logger.error(f"Error optimizing memory: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # System information endpoints
        @self.router.get("/system/info", response_model=SystemInfoResponse)
        async def get_system_info():
            """Get comprehensive system information."""
            try:
                import psutil
                import platform
                
                # System information
                system_info = {
                    "platform": platform.platform(),
                    "python_version": platform.python_version(),
                    "processor": platform.processor(),
                    "machine": platform.machine(),
                    "node": platform.node()
                }
                
                # Hardware information
                cpu_info = {
                    "cpu_count": psutil.cpu_count(),
                    "cpu_count_logical": psutil.cpu_count(logical=True),
                    "cpu_percent": psutil.cpu_percent(interval=1),
                    "cpu_freq": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None
                }
                
                memory_info = psutil.virtual_memory()._asdict()
                disk_info = psutil.disk_usage('/')._asdict()
                
                hardware_info = {
                    "cpu": cpu_info,
                    "memory": memory_info,
                    "disk": disk_info
                }
                
                # Performance information
                performance_info = {
                    "uptime": time.time() - self.server_state.start_time,
                    "request_count": self.server_state.request_count,
                    "error_count": self.server_state.error_count,
                    "active_requests": len(self.server_state.active_requests)
                }
                
                # Model information
                model_info = None
                if self.server_state.model_loaded and self.server_state.engine:
                    engine = self.server_state.engine
                    model_info = {
                        "model_name": engine.model_state.model_config.model_name,
                        "model_version": engine.model_state.model_config.model_version,
                        "architecture": engine.model_state.model_config.architecture,
                        "context_length": engine.model_state.current_context_length,
                        "performance_stats": engine.get_performance_stats()
                    }
                
                return SystemInfoResponse(
                    system_info=system_info,
                    hardware_info=hardware_info,
                    performance_info=performance_info,
                    model_info=model_info
                )
                
            except Exception as e:
                logger.error(f"Error getting system info: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # Advanced inference endpoints
        @self.router.post("/inference/chat")
        async def chat_completion(
            request: Dict[str, Any],
            engine: NeuroEngine = Depends(self._verify_model_loaded)
        ):
            """Chat completion endpoint similar to OpenAI API."""
            try:
                messages = request.get('messages', [])
                max_tokens = request.get('max_tokens', 50)
                temperature = request.get('temperature', 0.7)
                stream = request.get('stream', False)
                
                # Convert messages to prompt
                prompt = self._messages_to_prompt(messages)
                
                if stream:
                    return StreamingResponse(
                        self._stream_chat_response(engine, prompt, max_tokens, temperature),
                        media_type="text/plain"
                    )
                else:
                    # Generate response
                    response_text = engine.generate(
                        prompt=prompt,
                        max_new_tokens=max_tokens,
                        temperature=temperature
                    )
                    
                    return {
                        "choices": [{
                            "message": {
                                "role": "assistant",
                                "content": response_text
                            },
                            "finish_reason": "stop"
                        }],
                        "usage": {
                            "prompt_tokens": len(prompt.split()),
                            "completion_tokens": len(response_text.split()),
                            "total_tokens": len(prompt.split()) + len(response_text.split())
                        }
                    }
                
            except Exception as e:
                logger.error(f"Error in chat completion: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # Utility endpoints
        @self.router.get("/utils/version")
        async def get_version():
            """Get HERALD version information."""
            return {
                "version": "1.0.0",
                "api_version": "v1",
                "build_date": "2024-01-01",
                "commit_hash": "development"
            }
        
        @self.router.post("/utils/validate")
        async def validate_model_file(request: Dict[str, str]):
            """Validate a HERALD model file."""
            try:
                model_path = request.get('model_path')
                if not model_path:
                    raise HTTPException(status_code=400, detail="model_path is required")
                
                path = Path(model_path)
                if not path.exists():
                    return {"valid": False, "error": "File not found"}
                
                # Basic validation
                if not path.suffix == '.herald':
                    return {"valid": False, "error": "File must have .herald extension"}
                
                # Check file size
                file_size = path.stat().st_size
                if file_size < 1024:  # Less than 1KB
                    return {"valid": False, "error": "File too small to be a valid model"}
                
                return {
                    "valid": True,
                    "file_size": file_size,
                    "file_path": str(path)
                }
                
            except Exception as e:
                logger.error(f"Error validating model file: {e}")
                raise HTTPException(status_code=500, detail=str(e))
    
    def _verify_model_loaded(self) -> NeuroEngine:
        """Verify that a model is loaded and return the engine."""
        if not self.server_state.model_loaded or self.server_state.engine is None:
            raise HTTPException(status_code=400, detail="No model is currently loaded")
        return self.server_state.engine
    
    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert chat messages to a single prompt."""
        prompt_parts = []
        
        for message in messages:
            role = message.get('role', 'user')
            content = message.get('content', '')
            
            if role == 'system':
                prompt_parts.append(f"System: {content}")
            elif role == 'user':
                prompt_parts.append(f"User: {content}")
            elif role == 'assistant':
                prompt_parts.append(f"Assistant: {content}")
        
        return "\n".join(prompt_parts) + "\nAssistant:"
    
    async def _stream_chat_response(self, engine: NeuroEngine, prompt: str, max_tokens: int, temperature: float):
        """Stream chat response."""
        try:
            # For now, simulate streaming
            response_text = engine.generate(
                prompt=prompt,
                max_new_tokens=max_tokens,
                temperature=temperature
            )
            
            # Split into chunks for streaming
            words = response_text.split()
            chunk_size = max(1, len(words) // 10)
            
            for i in range(0, len(words), chunk_size):
                chunk = " ".join(words[i:i + chunk_size])
                yield f"data: {json.dumps({'content': chunk, 'finished': False})}\n\n"
                await asyncio.sleep(0.1)
            
            yield f"data: {json.dumps({'content': '', 'finished': True})}\n\n"
            
        except Exception as e:
            logger.error(f"Error in streaming chat response: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    def get_router(self) -> APIRouter:
        """Get the API router."""
        return self.router 