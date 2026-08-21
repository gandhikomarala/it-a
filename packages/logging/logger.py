"""High-performance structured JSON logging implementation with ContextVars."""
import json
import logging
import sys
import time
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, Dict, Optional

# Context variable for request correlation
_REQUEST_ID_CTX: ContextVar[Optional[str]] = ContextVar("request_id", default=None)
_USER_ID_CTX: ContextVar[Optional[str]] = ContextVar("user_id", default=None)

def set_request_context(request_id: str, user_id: Optional[str] = None) -> None:
    _REQUEST_ID_CTX.set(request_id)
    if user_id:
        _USER_ID_CTX.set(user_id)

def clear_request_context() -> None:
    _REQUEST_ID_CTX.set(None)
    _USER_ID_CTX.set(None)

class StructuredJsonFormatter(logging.Formatter):
    """Custom JSON formatter emitting compliant structured log objects."""

    def format(self, record: logging.LogRecord) -> str:
        log_obj: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "line": record.lineno,
            "function": record.funcName,
        }

        # Inject context variables
        req_id = _REQUEST_ID_CTX.get()
        if req_id:
            log_obj["request_id"] = req_id
        
        user_id = _USER_ID_CTX.get()
        if user_id:
            log_obj["user_id"] = user_id

        # Attach extra dictionary attributes
        if hasattr(record, "extra_fields") and isinstance(record.extra_fields, dict):
            log_obj.update(record.extra_fields)

        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_obj)

class StructuredLogger:
    """Wrapper over standard logger supporting extra metadata and timing."""

    def __init__(self, logger: logging.Logger):
        self._logger = logger

    def _log(self, level: int, msg: str, extra: Optional[Dict[str, Any]] = None, exc_info=None):
        if self._logger.isEnabledFor(level):
            if exc_info is True:
                exc_info = sys.exc_info()
                if exc_info == (None, None, None):
                    exc_info = None
            record = self._logger.makeRecord(
                self._logger.name, level, "(unknown)", 0, msg, (), exc_info
            )
            if extra:
                record.extra_fields = extra
            self._logger.handle(record)

    def debug(self, msg: str, **kwargs):
        self._log(logging.DEBUG, msg, kwargs)

    def info(self, msg: str, **kwargs):
        self._log(logging.INFO, msg, kwargs)

    def warning(self, msg: str, **kwargs):
        self._log(logging.WARNING, msg, kwargs)

    def error(self, msg: str, exc_info=True, **kwargs):
        self._log(logging.ERROR, msg, kwargs, exc_info=exc_info)

    def critical(self, msg: str, exc_info=True, **kwargs):
        self._log(logging.CRITICAL, msg, kwargs, exc_info=exc_info)

def setup_logging(level: str = "INFO") -> None:
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(StructuredJsonFormatter())
    root_logger.addHandler(handler)

def get_logger(name: str) -> StructuredLogger:
    return StructuredLogger(logging.getLogger(name))

class LogContext:
    """Context manager for timing and tracing blocks of execution."""

    def __init__(self, logger: StructuredLogger, operation_name: str, **context_kwargs):
        self.logger = logger
        self.operation = operation_name
        self.context = context_kwargs
        self.start_time: float = 0

    def __enter__(self):
        self.start_time = time.perf_counter()
        self.logger.info(f"Starting {self.operation}", **self.context)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration_ms = (time.perf_counter() - self.start_time) * 1000.0
        if exc_type:
            self.logger.error(
                f"Failed {self.operation}",
                duration_ms=round(duration_ms, 2),
                error=str(exc_val),
                **self.context
            )
        else:
            self.logger.info(
                f"Completed {self.operation}",
                duration_ms=round(duration_ms, 2),
                **self.context
            )
