from __future__ import annotations

import json
import logging
import logging.handlers
import os
import sys
import threading
import time
import traceback
from collections import defaultdict
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO, Union
import json
import uuid

_current_span: ContextVar[Optional[Dict[str, Any]]] = ContextVar("current_span", default=None)

# TODO: Add support for structured logging with OpenTelemetry semantic conventions
# FIXME: Log rotation doesn't work well with logrotate on some systems
# XXX: Current implementation doesn't handle log sampling for high-volume services
# HACK: Using ContextVar for span context - doesn't work across process boundaries

logger = logging.getLogger(__name__)


class LogLevel(Enum):
    TRACE = 5
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50


class LogFormat(Enum):
    JSON = "json"
    TEXT = "text"
    CONSOLE = "console"


@dataclass(frozen=True, slots=True)
class LogRecord:
    timestamp: str
    level: str
    logger: str
    message: str
    module: str = ""
    function: str = ""
    line: int = 0
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    trace_flags: int = 0
    baggage: Dict[str, str] = field(default_factory=dict)
    fields: Dict[str, Any] = field(default_factory=dict)
    exception: Optional[str] = None


class StructuredFormatter(logging.Formatter):
    def __init__(self, format_type: LogFormat = LogFormat.JSON):
        super().__init__()
        self.format_type = format_type

    def format(self, record: logging.LogRecord) -> str:
        span = _current_span.get()
        log_record = LogRecord(
            timestamp=datetime.utcnow().isoformat() + "Z",
            level=record.levelname,
            logger=record.name,
            message=record.getMessage(),
            module=record.module,
            function=record.funcName,
            line=record.lineno,
            trace_id=span.get("trace_id") if span else None,
            span_id=span.get("span_id") if span else None,
            trace_flags=span.get("trace_flags", 0) if span else 0,
            baggage=span.get("baggage", {}) if span else {},
            fields=getattr(record, "fields", {}),
            exception=self._format_exception(record.exc_info) if record.exc_info else None,
        )

        if self.format_type == LogFormat.JSON:
            return json.dumps(log_record.__dict__, default=str, separators=(",", ":"))
        elif self.format_type == LogFormat.CONSOLE:
            return self._format_console(log_record)
        else:
            return self._format_text(log_record)

    def _format_console(self, record: LogRecord) -> str:
        level_color = {
            "TRACE": "\033[90m",
            "DEBUG": "\033[36m",
            "INFO": "\033[32m",
            "WARNING": "\033[33m",
            "ERROR": "\033[31m",
            "CRITICAL": "\033[31;1m",
        }
        reset = "\033[0m"
        color = level_color.get(record.level, "")
        timestamp = record.timestamp[:19].replace("T", " ")
        trace_info = f" [{record.trace_id[:8]}:{record.span_id[:8]}]" if record.trace_id else ""
        return f"{color}{timestamp} {record.level:8s}{reset} {record.logger}{trace_info} | {record.message}"

    def _format_text(self, record: LogRecord) -> str:
        parts = [
            record.timestamp,
            record.level,
            record.logger,
            record.message,
        ]
        if record.trace_id:
            parts.append(f"trace_id={record.trace_id}")
        if record.fields:
            parts.append(f"fields={json.dumps(record.fields)}")
        if record.exception:
            parts.append(f"exception={record.exception}")
        return " | ".join(parts)

    def _format_exception(self, exc_info) -> str:
        if exc_info:
            return "".join(traceback.format_exception(*exc_info))
        return ""


class ContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.fields = getattr(record, "fields", {})
        span = _current_span.get()
        if span:
            record.trace_id = span.get("trace_id")
            record.span_id = span.get("span_id")
        else:
            record.trace_id = None
            record.span_id = None
        return True


class RotatingFileHandler(logging.handlers.RotatingFileHandler):
    def __init__(
        self,
        filename: str,
        max_bytes: int = 100 * 1024 * 1024,
        backup_count: int = 10,
        encoding: str = "utf-8",
    ):
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        super().__init__(filename, maxBytes=max_bytes, backupCount=backup_count, encoding=encoding)


class AsyncLogHandler(logging.Handler):
    def __init__(self, handler: logging.Handler, queue_size: int = 10000):
        super().__init__()
        self.handler = handler
        self.queue = asyncio.Queue(maxsize=queue_size)
        self._running = False
        self._task: Optional[asyncio.Task] = None

    async def start(self):
        self._running = True
        self._task = asyncio.create_task(self._process())

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _process(self):
        while self._running:
            try:
                record = await asyncio.wait_for(self.queue.get(), timeout=1.0)
                self.handler.emit(record)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                print(f"Async log handler error: {e}")

    def emit(self, record):
        try:
            self.queue.put_nowait(record)
        except asyncio.QueueFull:
            self.handler.emit(record)


class JSONFileHandler(RotatingFileHandler):
    def __init__(self, filename: str, **kwargs):
        super().__init__(filename, **kwargs)
        self.setFormatter(StructuredFormatter(LogFormat.JSON))


class ConsoleHandler(logging.StreamHandler):
    def __init__(self, stream: TextIO = None):
        super().__init__(stream or sys.stdout)
        self.setFormatter(StructuredFormatter(LogFormat.CONSOLE))


def create_logger(
    name: str,
    level: Union[int, LogLevel] = LogLevel.INFO,
    handlers: List[logging.Handler] = None,
    propagate: bool = False,
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level if isinstance(level, int) else level.value)
    logger.propagate = propagate
    logger.addFilter(ContextFilter())

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    if handlers:
        for handler in handlers:
            logger.addHandler(handler)
    else:
        logger.addHandler(ConsoleHandler())

    return logger


def setup_logging(
    level: Union[int, LogLevel] = LogLevel.INFO,
    log_dir: str = "logs",
    json_logs: bool = True,
    console_logs: bool = True,
    service_name: str = "smart-vision-ai",
) -> logging.Logger:
    root = logging.getLogger()
    root.setLevel(level if isinstance(level, int) else level.value)

    for handler in root.handlers[:]:
        root.removeHandler(handler)

    root.addFilter(ContextFilter())

    if console_logs:
        console = ConsoleHandler(sys.stdout)
        root.addHandler(console)

    if json_logs:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        json_handler = JSONFileHandler(
            filename=Path(log_dir) / f"{service_name}.json",
            max_bytes=100 * 1024 * 1024,
            backup_count=10,
        )
        root.addHandler(json_handler)

        error_handler = JSONFileHandler(
            filename=Path(log_dir) / f"{service_name}.error.json",
            max_bytes=50 * 1024 * 1024,
            backup_count=5,
        )
        error_handler.setLevel(logging.ERROR)
        root.addHandler(error_handler)

    return root


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def log_with_context(
    logger: logging.Logger,
    level: Union[int, LogLevel],
    message: str,
    **fields,
):
    level = level if isinstance(level, int) else level.value
    extra = {"fields": fields}
    logger.log(level, message, extra=extra)


class LogContext:
    _context: ContextVar[Dict[str, Any]] = ContextVar("log_context", default={})

    @classmethod
    def set(cls, **kwargs):
        current = cls._context.get().copy()
        current.update(kwargs)
        cls._context.set(current)

    @classmethod
    def get(cls) -> Dict[str, Any]:
        return cls._context.get().copy()

    @classmethod
    def clear(cls):
        cls._context.set({})

    @classmethod
    @contextmanager
    def use(cls, **kwargs):
        token = cls._context.set({**cls._context.get(), **kwargs})
        try:
            yield
        finally:
            cls._context.reset(token)


def with_log_context(**context_fields):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with LogContext.use(**context_fields):
                return func(*args, **kwargs)
        return wrapper
    return decorator


class StructuredLogger:
    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def _log(self, level: int, message: str, **fields):
        self.logger.log(level, message, extra={"fields": fields})

    def trace(self, message: str, **fields):
        self._log(LogLevel.TRACE.value, message, **fields)

    def debug(self, message: str, **fields):
        self._log(LogLevel.DEBUG.value, message, **fields)

    def info(self, message: str, **fields):
        self._log(LogLevel.INFO.value, message, **fields)

    def warning(self, message: str, **fields):
        self._log(LogLevel.WARNING.value, message, **fields)

    def error(self, message: str, **fields):
        self._log(LogLevel.ERROR.value, message, **fields)

    def critical(self, message: str, **fields):
        self._log(LogLevel.CRITICAL.value, message, **fields)

    def exception(self, message: str, **fields):
        self._log(LogLevel.ERROR.value, message, exc_info=True, **fields)

    def bind(self, **fields) -> "StructuredLogger":
        return StructuredLogger(self.logger)

    def with_context(self, **fields) -> "StructuredLogger":
        return StructuredLogger(self.logger.bind(**fields))


def get_structured_logger(name: str) -> StructuredLogger:
    return StructuredLogger(logging.getLogger(name))


class AuditLogger:
    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def log_access(
        self,
        user_id: str,
        resource: str,
        action: str,
        result: str,
        **details,
    ):
        self.logger.info(
            "Access audit",
            extra={"fields": {
                "audit_type": "access",
                "user_id": user_id,
                "resource": resource,
                "action": action,
                "result": result,
                **details,
            }},
        )

    def log_change(
        self,
        user_id: str,
        resource: str,
        action: str,
        before: Any,
        after: Any,
        **details,
    ):
        self.logger.info(
            "Change audit",
            extra={"fields": {
                "audit_type": "change",
                "user_id": user_id,
                "resource": resource,
                "action": action,
                "before": str(before),
                "after": str(after),
                **details,
            }},
        )

    def log_security(
        self,
        event_type: str,
        severity: str,
        user_id: Optional[str] = None,
        ip: Optional[str] = None,
        **details,
    ):
        self.logger.warning(
            "Security audit",
            extra={"fields": {
                "audit_type": "security",
                "event_type": event_type,
                "severity": severity,
                "user_id": user_id,
                "ip": ip,
                **details,
            }},
        )


class LogSampler:
    def __init__(self, sample_rate: float = 1.0):
        self.sample_rate = sample_rate
        self._counter = 0
        self._lock = threading.Lock()

    def should_log(self) -> bool:
        if self.sample_rate >= 1.0:
            return True
        if self.sample_rate <= 0.0:
            return False
        with self._lock:
            self._counter += 1
            return random.random() < self.sample_rate


class RateLimitedLogger:
    def __init__(self, logger: logging.Logger, max_per_second: int = 100):
        self.logger = logger
        self.max_per_second = max_per_second
        self._counts: Dict[int, int] = defaultdict(int)
        self._last_reset = int(time.time())
        self._lock = threading.Lock()

    def _log(self, level: int, message: str, **fields):
        current_second = int(time.time())
        with self._lock:
            if current_second != self._last_reset:
                self._counts.clear()
                self._last_reset = current_second
            self._counts[current_second] += 1
            if self._counts[current_second] <= self.max_per_second:
                self._log_unlimited(level, message, **fields)

    def _log_unlimited(self, level: int, message: str, **fields):
        extra = {"fields": fields}
        self.logger.log(level, message, extra=extra)

    def log(self, level: int, message: str, **fields):
        self._log(level, message, **fields)


def configure_third_party_loggers(level: int = LogLevel.WARNING.value):
    for name in [
        "urllib3",
        "requests",
        "httpx",
        "asyncio",
        "aiohttp",
        "sqlalchemy",
        "uvicorn",
        "fastapi",
    ]:
        logging.getLogger(name).setLevel(level)


def silence_noisy_loggers():
    noisy = [
        "asyncio",
        "urllib3.connectionpool",
        "httpx",
        "multipart",
        "watchfiles",
    ]
    for name in noisy:
        logging.getLogger(name).setLevel(LogLevel.WARNING.value)


# Initialize default logging
def init_logging(
    level: Union[int, LogLevel] = LogLevel.INFO,
    log_dir: str = "logs",
    service_name: str = "smart-vision-ai",
) -> logging.Logger:
    return setup_logging(
        level=level,
        log_dir=log_dir,
        json_logs=True,
        console_logs=True,
        service_name=service_name,
    )


# Initialize on import
_init_done = False


def ensure_logging_initialized():
    global _init_done
    if not _init_done:
        init_logging()
        silence_noisy_loggers()
        configure_third_party_loggers()
        _init_done = True


ensure_logging_initialized()