"""
Authentication and rate limiting for the API.
"""
import time
import hashlib
import secrets
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from loguru import logger

from fastapi import HTTPException, Security, Depends, Request
from fastapi.security import APIKeyHeader, APIKeyQuery


# --- API Key Management ---

@dataclass
class APIKey:
    """API key with metadata."""
    key: str
    name: str
    created_at: datetime
    expires_at: Optional[datetime] = None
    rate_limit: int = 100  # requests per minute
    scopes: List[str] = field(default_factory=lambda: ["search", "index"])
    enabled: bool = True


class APIKeyManager:
    """Manages API keys for authentication."""
    
    def __init__(self):
        self._keys: Dict[str, APIKey] = {}
    
    def generate_key(
        self,
        name: str,
        rate_limit: int = 100,
        scopes: List[str] = None,
        expires_in_days: Optional[int] = None
    ) -> APIKey:
        """Generate a new API key."""
        key = secrets.token_urlsafe(32)
        
        expires_at = None
        if expires_in_days:
            expires_at = datetime.utcnow() + timedelta(days=expires_in_days)
        
        api_key = APIKey(
            key=key,
            name=name,
            created_at=datetime.utcnow(),
            expires_at=expires_at,
            rate_limit=rate_limit,
            scopes=scopes or ["search", "index"],
            enabled=True
        )
        
        self._keys[key] = api_key
        logger.info(f"Generated API key for: {name}")
        return api_key
    
    def validate(self, key: str, required_scope: str = None) -> Optional[APIKey]:
        """Validate an API key."""
        api_key = self._keys.get(key)
        
        if api_key is None:
            return None
        
        if not api_key.enabled:
            return None
        
        if api_key.expires_at and datetime.utcnow() > api_key.expires_at:
            return None
        
        if required_scope and required_scope not in api_key.scopes:
            return None
        
        return api_key
    
    def revoke(self, key: str) -> bool:
        """Revoke an API key."""
        if key in self._keys:
            self._keys[key].enabled = False
            logger.info(f"Revoked API key: {self._keys[key].name}")
            return True
        return False
    
    def list_keys(self) -> List[Dict[str, Any]]:
        """List all API keys (without the actual key values)."""
        return [
            {
                "name": k.name,
                "created_at": k.created_at.isoformat(),
                "expires_at": k.expires_at.isoformat() if k.expires_at else None,
                "rate_limit": k.rate_limit,
                "scopes": k.scopes,
                "enabled": k.enabled,
                "key_prefix": k.key[:8] + "..."
            }
            for k in self._keys.values()
        ]


# --- Rate Limiting ---

@dataclass
class RateLimitEntry:
    """Tracks request count for rate limiting."""
    count: int = 0
    window_start: float = 0.0


class RateLimiter:
    """Token bucket rate limiter."""
    
    def __init__(self, default_limit: int = 100, window_seconds: int = 60):
        """
        Args:
            default_limit: Default requests per window.
            window_seconds: Time window in seconds.
        """
        self.default_limit = default_limit
        self.window_seconds = window_seconds
        self._buckets: Dict[str, RateLimitEntry] = {}
    
    def _get_key(self, identifier: str) -> str:
        """Create rate limit key."""
        return hashlib.md5(identifier.encode()).hexdigest()
    
    def check(self, identifier: str, limit: Optional[int] = None) -> tuple[bool, Dict[str, Any]]:
        """
        Check if request is allowed.
        
        Args:
            identifier: Client identifier (API key, IP, etc.).
            limit: Custom limit override.
        
        Returns:
            Tuple of (allowed, info_dict).
        """
        key = self._get_key(identifier)
        limit = limit or self.default_limit
        now = time.time()
        
        entry = self._buckets.get(key)
        
        if entry is None or now - entry.window_start >= self.window_seconds:
            # New window
            self._buckets[key] = RateLimitEntry(count=1, window_start=now)
            return True, {
                "limit": limit,
                "remaining": limit - 1,
                "reset": int(now + self.window_seconds)
            }
        
        if entry.count >= limit:
            # Rate limited
            reset_time = entry.window_start + self.window_seconds
            return False, {
                "limit": limit,
                "remaining": 0,
                "reset": int(reset_time),
                "retry_after": int(reset_time - now)
            }
        
        # Increment counter
        entry.count += 1
        return True, {
            "limit": limit,
            "remaining": limit - entry.count,
            "reset": int(entry.window_start + self.window_seconds)
        }
    
    def reset(self, identifier: str):
        """Reset rate limit for an identifier."""
        key = self._get_key(identifier)
        self._buckets.pop(key, None)


# --- FastAPI Dependencies ---

# Global instances
api_key_manager = APIKeyManager()
rate_limiter = RateLimiter()

# Security schemes
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
api_key_query = APIKeyQuery(name="api_key", auto_error=False)


async def get_api_key(
    api_key_header: str = Security(api_key_header),
    api_key_query: str = Security(api_key_query)
) -> Optional[str]:
    """Extract API key from header or query parameter."""
    return api_key_header or api_key_query


async def require_auth(
    request: Request,
    api_key: str = Depends(get_api_key)
) -> APIKey:
    """Require valid API key authentication."""
    if api_key is None:
        raise HTTPException(
            status_code=401,
            detail="API key required",
            headers={"WWW-Authenticate": "ApiKey"}
        )
    
    validated = api_key_manager.validate(api_key)
    if validated is None:
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired API key"
        )
    
    # Check rate limit
    allowed, info = rate_limiter.check(api_key, validated.rate_limit)
    
    # Add rate limit headers to response
    request.state.rate_limit_info = info
    
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded",
            headers={
                "X-RateLimit-Limit": str(info["limit"]),
                "X-RateLimit-Remaining": str(info["remaining"]),
                "X-RateLimit-Reset": str(info["reset"]),
                "Retry-After": str(info.get("retry_after", 60))
            }
        )
    
    return validated


async def require_scope(scope: str):
    """Factory for scope-specific auth dependency."""
    async def check_scope(api_key: APIKey = Depends(require_auth)) -> APIKey:
        if scope not in api_key.scopes:
            raise HTTPException(
                status_code=403,
                detail=f"Scope '{scope}' required"
            )
        return api_key
    return check_scope


async def optional_auth(
    api_key: str = Depends(get_api_key)
) -> Optional[APIKey]:
    """Optional authentication - returns None if no key provided."""
    if api_key is None:
        return None
    return api_key_manager.validate(api_key)


# --- IP-based Rate Limiting (for unauthenticated requests) ---

async def rate_limit_by_ip(request: Request):
    """Rate limit by client IP address."""
    client_ip = request.client.host if request.client else "unknown"
    
    allowed, info = rate_limiter.check(f"ip:{client_ip}", limit=30)  # Lower limit for unauthenticated
    
    request.state.rate_limit_info = info
    
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded",
            headers={
                "X-RateLimit-Limit": str(info["limit"]),
                "X-RateLimit-Remaining": str(info["remaining"]),
                "X-RateLimit-Reset": str(info["reset"]),
                "Retry-After": str(info.get("retry_after", 60))
            }
        )


# --- Middleware for Rate Limit Headers ---

from starlette.middleware.base import BaseHTTPMiddleware


class RateLimitHeaderMiddleware(BaseHTTPMiddleware):
    """Add rate limit headers to all responses."""
    
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        
        # Add rate limit headers if available
        if hasattr(request.state, "rate_limit_info"):
            info = request.state.rate_limit_info
            response.headers["X-RateLimit-Limit"] = str(info["limit"])
            response.headers["X-RateLimit-Remaining"] = str(info["remaining"])
            response.headers["X-RateLimit-Reset"] = str(info["reset"])
        
        return response
