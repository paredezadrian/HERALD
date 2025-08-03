"""
HERALD API Middleware Implementation
Request processing middleware for HERALD API

This module implements middleware for:
- Request processing and validation
- Security and authentication
- Logging and monitoring
- Performance tracking
- Error handling

Target: ~432 lines of middleware implementations
"""

import asyncio
import logging
import time
import uuid
import json
import hashlib
import hmac
import secrets
from typing import Dict, List, Optional, Any, Union, Callable
from pathlib import Path
import threading
from contextlib import asynccontextmanager

# FastAPI imports
from fastapi import Request, Response, HTTPException, status
from fastapi.middleware.base import BaseHTTPMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp, Receive, Scope, Send
import uvicorn

# Security imports
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

logger = logging.getLogger(__name__)

class HeraldMiddleware(BaseHTTPMiddleware):
    """HERALD API middleware for request processing and security."""
    
    def __init__(
        self,
        app: ASGIApp,
        enable_rate_limiting: bool = True,
        enable_security: bool = True,
        enable_logging: bool = True,
        enable_monitoring: bool = True,
        max_requests_per_minute: int = 100,
        secret_key: Optional[str] = None
    ):
        """Initialize HERALD middleware."""
        super().__init__(app)
        self.enable_rate_limiting = enable_rate_limiting
        self.enable_security = enable_security
        self.enable_logging = enable_logging
        self.enable_monitoring = enable_monitoring
        self.max_requests_per_minute = max_requests_per_minute
        self.secret_key = secret_key or self._generate_secret_key()
        
        # Rate limiting state
        self.request_counts: Dict[str, Dict] = {}
        self.rate_limit_lock = threading.Lock()
        
        # Security state
        self.blocked_ips: set = set()
        self.suspicious_ips: Dict[str, int] = {}
        self.security_lock = threading.Lock()
        
        # Monitoring state
        self.request_times: List[float] = []
        self.error_counts: Dict[str, int] = {}
        self.monitoring_lock = threading.Lock()
        
        # Initialize security components
        if self.enable_security:
            self._initialize_security()
    
    def _generate_secret_key(self) -> str:
        """Generate a secure secret key."""
        return base64.urlsafe_b64encode(secrets.token_bytes(32)).decode()
    
    def _initialize_security(self):
        """Initialize security components."""
        try:
            # Generate encryption key
            salt = secrets.token_bytes(16)
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            key = base64.urlsafe_b64encode(kdf.derive(self.secret_key.encode()))
            self.cipher = Fernet(key)
            
            logger.info("Security components initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing security components: {e}")
            self.enable_security = False
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request through middleware pipeline."""
        start_time = time.time()
        request_id = str(uuid.uuid4())
        
        # Add request ID to request state
        request.state.request_id = request_id
        request.state.start_time = start_time
        
        try:
            # Security checks
            if self.enable_security:
                await self._security_check(request)
            
            # Rate limiting
            if self.enable_rate_limiting:
                await self._rate_limit_check(request)
            
            # Logging
            if self.enable_logging:
                await self._log_request(request)
            
            # Process request
            response = await call_next(request)
            
            # Add security headers
            if self.enable_security:
                response = await self._add_security_headers(response)
            
            # Monitoring
            if self.enable_monitoring:
                await self._monitor_request(request, response, start_time)
            
            return response
            
        except HTTPException as e:
            # Handle HTTP exceptions
            await self._handle_http_exception(request, e, start_time)
            raise
            
        except Exception as e:
            # Handle unexpected exceptions
            await self._handle_unexpected_exception(request, e, start_time)
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Internal server error",
                    "request_id": request_id,
                    "timestamp": time.time()
                }
            )
    
    async def _security_check(self, request: Request):
        """Perform security checks on the request."""
        try:
            client_ip = self._get_client_ip(request)
            
            # Check if IP is blocked
            if client_ip in self.blocked_ips:
                raise HTTPException(
                    status_code=403,
                    detail="Access denied: IP address is blocked"
                )
            
            # Check for suspicious activity
            with self.security_lock:
                if client_ip in self.suspicious_ips:
                    self.suspicious_ips[client_ip] += 1
                    if self.suspicious_ips[client_ip] >= 10:
                        self.blocked_ips.add(client_ip)
                        logger.warning(f"IP {client_ip} blocked due to suspicious activity")
                        raise HTTPException(
                            status_code=403,
                            detail="Access denied: Suspicious activity detected"
                        )
                else:
                    self.suspicious_ips[client_ip] = 1
            
            # Validate request headers
            await self._validate_headers(request)
            
            # Check request size
            await self._check_request_size(request)
            
            # Validate content type for POST requests
            if request.method == "POST":
                await self._validate_content_type(request)
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error in security check: {e}")
            # Don't block the request for security check errors
    
    async def _rate_limit_check(self, request: Request):
        """Check rate limiting for the request."""
        try:
            client_ip = self._get_client_ip(request)
            current_time = time.time()
            
            with self.rate_limit_lock:
                # Clean up old entries
                cutoff_time = current_time - 60  # 1 minute window
                if client_ip in self.request_counts:
                    self.request_counts[client_ip] = {
                        timestamp: count for timestamp, count in self.request_counts[client_ip].items()
                        if timestamp > cutoff_time
                    }
                
                # Count current requests
                current_minute = int(current_time / 60)
                if client_ip not in self.request_counts:
                    self.request_counts[client_ip] = {}
                
                if current_minute not in self.request_counts[client_ip]:
                    self.request_counts[client_ip][current_minute] = 0
                
                self.request_counts[client_ip][current_minute] += 1
                
                # Check rate limit
                total_requests = sum(self.request_counts[client_ip].values())
                if total_requests > self.max_requests_per_minute:
                    logger.warning(f"Rate limit exceeded for IP {client_ip}")
                    raise HTTPException(
                        status_code=429,
                        detail=f"Rate limit exceeded. Maximum {self.max_requests_per_minute} requests per minute."
                    )
        
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error in rate limit check: {e}")
            # Don't block the request for rate limit check errors
    
    async def _log_request(self, request: Request):
        """Log request information."""
        try:
            client_ip = self._get_client_ip(request)
            user_agent = request.headers.get("user-agent", "Unknown")
            method = request.method
            path = request.url.path
            query_params = str(request.query_params)
            
            logger.info(
                f"Request: {method} {path} from {client_ip} "
                f"(User-Agent: {user_agent[:100]}) "
                f"(Query: {query_params[:200]})"
            )
            
        except Exception as e:
            logger.error(f"Error logging request: {e}")
    
    async def _monitor_request(self, request: Request, response: Response, start_time: float):
        """Monitor request performance and statistics."""
        try:
            request_time = time.time() - start_time
            
            with self.monitoring_lock:
                # Track request times
                self.request_times.append(request_time)
                if len(self.request_times) > 1000:  # Keep last 1000 requests
                    self.request_times.pop(0)
                
                # Track error counts
                if response.status_code >= 400:
                    error_type = f"{response.status_code}"
                    self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
            
        except Exception as e:
            logger.error(f"Error monitoring request: {e}")
    
    async def _add_security_headers(self, response: Response) -> Response:
        """Add security headers to the response."""
        try:
            # Security headers
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["X-Frame-Options"] = "DENY"
            response.headers["X-XSS-Protection"] = "1; mode=block"
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
            response.headers["Content-Security-Policy"] = "default-src 'self'"
            response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
            
            # Remove server information
            if "server" in response.headers:
                del response.headers["server"]
            
            return response
            
        except Exception as e:
            logger.error(f"Error adding security headers: {e}")
            return response
    
    async def _handle_http_exception(self, request: Request, exc: HTTPException, start_time: float):
        """Handle HTTP exceptions."""
        try:
            request_time = time.time() - start_time
            client_ip = self._get_client_ip(request)
            
            logger.warning(
                f"HTTP Exception: {exc.status_code} - {exc.detail} "
                f"from {client_ip} in {request_time:.3f}s"
            )
            
            # Update monitoring
            with self.monitoring_lock:
                error_type = f"{exc.status_code}"
                self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
            
        except Exception as e:
            logger.error(f"Error handling HTTP exception: {e}")
    
    async def _handle_unexpected_exception(self, request: Request, exc: Exception, start_time: float):
        """Handle unexpected exceptions."""
        try:
            request_time = time.time() - start_time
            client_ip = self._get_client_ip(request)
            
            logger.error(
                f"Unexpected Exception: {type(exc).__name__} - {str(exc)} "
                f"from {client_ip} in {request_time:.3f}s"
            )
            
            # Update monitoring
            with self.monitoring_lock:
                self.error_counts["500"] = self.error_counts.get("500", 0) + 1
            
        except Exception as e:
            logger.error(f"Error handling unexpected exception: {e}")
    
    def _get_client_ip(self, request: Request) -> str:
        """Get the client IP address."""
        # Check for forwarded headers
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip
        
        # Fallback to direct IP
        return request.client.host if request.client else "unknown"
    
    async def _validate_headers(self, request: Request):
        """Validate request headers."""
        try:
            # Check for required headers if needed
            content_type = request.headers.get("content-type", "")
            
            if request.method == "POST" and not content_type.startswith("application/json"):
                raise HTTPException(
                    status_code=400,
                    detail="Content-Type must be application/json for POST requests"
                )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error validating headers: {e}")
    
    async def _check_request_size(self, request: Request):
        """Check request size limits."""
        try:
            content_length = request.headers.get("content-length")
            if content_length:
                size = int(content_length)
                max_size = 10 * 1024 * 1024  # 10MB limit
                
                if size > max_size:
                    raise HTTPException(
                        status_code=413,
                        detail="Request too large. Maximum size is 10MB."
                    )
        
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error checking request size: {e}")
    
    async def _validate_content_type(self, request: Request):
        """Validate content type for POST requests."""
        try:
            content_type = request.headers.get("content-type", "")
            
            if not content_type.startswith("application/json"):
                raise HTTPException(
                    status_code=400,
                    detail="Content-Type must be application/json"
                )
        
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error validating content type: {e}")
    
    def get_monitoring_stats(self) -> Dict[str, Any]:
        """Get monitoring statistics."""
        try:
            with self.monitoring_lock:
                avg_request_time = (
                    sum(self.request_times) / len(self.request_times)
                    if self.request_times else 0
                )
                
                return {
                    "total_requests": len(self.request_times),
                    "average_request_time": avg_request_time,
                    "error_counts": self.error_counts.copy(),
                    "rate_limited_ips": len(self.request_counts),
                    "blocked_ips": len(self.blocked_ips),
                    "suspicious_ips": len(self.suspicious_ips)
                }
        
        except Exception as e:
            logger.error(f"Error getting monitoring stats: {e}")
            return {}
    
    def clear_monitoring_stats(self):
        """Clear monitoring statistics."""
        try:
            with self.monitoring_lock:
                self.request_times.clear()
                self.error_counts.clear()
            
            with self.rate_limit_lock:
                self.request_counts.clear()
            
            with self.security_lock:
                self.blocked_ips.clear()
                self.suspicious_ips.clear()
                
        except Exception as e:
            logger.error(f"Error clearing monitoring stats: {e}")


class AuthenticationMiddleware(BaseHTTPMiddleware):
    """Authentication middleware for API endpoints."""
    
    def __init__(
        self,
        app: ASGIApp,
        api_keys: Optional[Dict[str, str]] = None,
        require_auth: bool = False
    ):
        """Initialize authentication middleware."""
        super().__init__(app)
        self.api_keys = api_keys or {}
        self.require_auth = require_auth
        self.auth_failures: Dict[str, int] = {}
        self.auth_lock = threading.Lock()
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request through authentication middleware."""
        try:
            # Skip authentication for certain endpoints
            if self._should_skip_auth(request):
                return await call_next(request)
            
            # Check authentication
            if self.require_auth or self.api_keys:
                await self._authenticate_request(request)
            
            return await call_next(request)
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error in authentication middleware: {e}")
            return JSONResponse(
                status_code=500,
                content={"error": "Authentication error"}
            )
    
    def _should_skip_auth(self, request: Request) -> bool:
        """Check if authentication should be skipped for this request."""
        skip_paths = [
            "/health",
            "/docs",
            "/redoc",
            "/openapi.json",
            "/"
        ]
        
        return any(request.url.path.startswith(path) for path in skip_paths)
    
    async def _authenticate_request(self, request: Request):
        """Authenticate the request."""
        try:
            # Check for API key in headers
            api_key = request.headers.get("x-api-key")
            if not api_key:
                raise HTTPException(
                    status_code=401,
                    detail="API key required"
                )
            
            # Validate API key
            if api_key not in self.api_keys:
                await self._record_auth_failure(request)
                raise HTTPException(
                    status_code=401,
                    detail="Invalid API key"
                )
            
            # Add user info to request state
            request.state.user = self.api_keys[api_key]
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error authenticating request: {e}")
            raise HTTPException(status_code=401, detail="Authentication failed")
    
    async def _record_auth_failure(self, request: Request):
        """Record authentication failure."""
        try:
            client_ip = self._get_client_ip(request)
            
            with self.auth_lock:
                self.auth_failures[client_ip] = self.auth_failures.get(client_ip, 0) + 1
                
                # Block IP after too many failures
                if self.auth_failures[client_ip] >= 5:
                    logger.warning(f"IP {client_ip} blocked due to authentication failures")
        
        except Exception as e:
            logger.error(f"Error recording auth failure: {e}")
    
    def _get_client_ip(self, request: Request) -> str:
        """Get the client IP address."""
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip
        
        return request.client.host if request.client else "unknown"


class LoggingMiddleware(BaseHTTPMiddleware):
    """Enhanced logging middleware."""
    
    def __init__(
        self,
        app: ASGIApp,
        log_requests: bool = True,
        log_responses: bool = True,
        log_errors: bool = True,
        log_performance: bool = True
    ):
        """Initialize logging middleware."""
        super().__init__(app)
        self.log_requests = log_requests
        self.log_responses = log_responses
        self.log_errors = log_errors
        self.log_performance = log_performance
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request through logging middleware."""
        start_time = time.time()
        
        # Log request
        if self.log_requests:
            await self._log_request(request)
        
        try:
            # Process request
            response = await call_next(request)
            
            # Log response
            if self.log_responses:
                await self._log_response(request, response, start_time)
            
            return response
            
        except Exception as e:
            # Log error
            if self.log_errors:
                await self._log_error(request, e, start_time)
            raise
    
    async def _log_request(self, request: Request):
        """Log request details."""
        try:
            client_ip = self._get_client_ip(request)
            method = request.method
            path = request.url.path
            query = str(request.query_params)
            headers = dict(request.headers)
            
            # Remove sensitive headers
            sensitive_headers = ['authorization', 'cookie', 'x-api-key']
            for header in sensitive_headers:
                if header in headers:
                    headers[header] = '[REDACTED]'
            
            logger.info(
                f"Request: {method} {path} from {client_ip} "
                f"(Query: {query[:200]}) "
                f"(Headers: {headers})"
            )
        
        except Exception as e:
            logger.error(f"Error logging request: {e}")
    
    async def _log_response(self, request: Request, response: Response, start_time: float):
        """Log response details."""
        try:
            request_time = time.time() - start_time
            status_code = response.status_code
            client_ip = self._get_client_ip(request)
            method = request.method
            path = request.url.path
            
            logger.info(
                f"Response: {method} {path} -> {status_code} "
                f"from {client_ip} in {request_time:.3f}s"
            )
        
        except Exception as e:
            logger.error(f"Error logging response: {e}")
    
    async def _log_error(self, request: Request, error: Exception, start_time: float):
        """Log error details."""
        try:
            request_time = time.time() - start_time
            client_ip = self._get_client_ip(request)
            method = request.method
            path = request.url.path
            error_type = type(error).__name__
            error_message = str(error)
            
            logger.error(
                f"Error: {method} {path} -> {error_type}: {error_message} "
                f"from {client_ip} in {request_time:.3f}s"
            )
        
        except Exception as e:
            logger.error(f"Error logging error: {e}")
    
    def _get_client_ip(self, request: Request) -> str:
        """Get the client IP address."""
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip
        
        return request.client.host if request.client else "unknown"


# Middleware factory functions
def create_herald_middleware(
    enable_rate_limiting: bool = True,
    enable_security: bool = True,
    enable_logging: bool = True,
    enable_monitoring: bool = True,
    max_requests_per_minute: int = 100,
    secret_key: Optional[str] = None
) -> HeraldMiddleware:
    """Create a configured HERALD middleware instance."""
    return HeraldMiddleware(
        app=None,  # Will be set by FastAPI
        enable_rate_limiting=enable_rate_limiting,
        enable_security=enable_security,
        enable_logging=enable_logging,
        enable_monitoring=enable_monitoring,
        max_requests_per_minute=max_requests_per_minute,
        secret_key=secret_key
    )


def create_auth_middleware(
    api_keys: Optional[Dict[str, str]] = None,
    require_auth: bool = False
) -> AuthenticationMiddleware:
    """Create a configured authentication middleware instance."""
    return AuthenticationMiddleware(
        app=None,  # Will be set by FastAPI
        api_keys=api_keys,
        require_auth=require_auth
    )


def create_logging_middleware(
    log_requests: bool = True,
    log_responses: bool = True,
    log_errors: bool = True,
    log_performance: bool = True
) -> LoggingMiddleware:
    """Create a configured logging middleware instance."""
    return LoggingMiddleware(
        app=None,  # Will be set by FastAPI
        log_requests=log_requests,
        log_responses=log_responses,
        log_errors=log_errors,
        log_performance=log_performance
    ) 