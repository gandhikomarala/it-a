# Middlewares.
import time
import uuid
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from packages.logging.logger import set_request_context, clear_request_context, get_logger

logger = get_logger("http.access")

class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        req_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        set_request_context(request_id=req_id)
        
        t0 = time.perf_counter()
        response: Response = await call_next(request)
        duration_ms = (time.perf_counter() - t0) * 1000.0
        
        response.headers["X-Request-ID"] = req_id
        response.headers["X-Response-Time-Ms"] = str(round(duration_ms, 2))
        
        logger.info(
            f"{request.method} {request.url.path} {response.status_code}",
            status_code=response.status_code,
            duration_ms=round(duration_ms, 2),
            method=request.method,
            path=request.url.path,
            client_ip=request.client.host if request.client else None
        )
        
        clear_request_context()
        return response

class AuditLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        return await call_next(request)

class RateLimitingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        return await call_next(request)
