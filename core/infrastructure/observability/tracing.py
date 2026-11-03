from __future__ import annotations

import asyncio
import contextvars
import functools
import hashlib
import json
import os
import random
import threading
import time
import uuid
from collections import defaultdict
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from contextlib import asynccontextmanager
import asyncio
import logging

logger = logging.getLogger(__name__)

# TODO: Add support for W3C TraceContext propagation headers
# FIXME: Sampling decision should be configurable per-service
# XXX: Current implementation doesn't support trace linking across services properly
# HACK: Using contextvars for span context - might not work in all async contexts


class SpanKind(Enum):
    INTERNAL = "internal"
    SERVER = "server"
    CLIENT = "client"
    PRODUCER = "producer"
    CONSUMER = "consumer"


class SpanStatus(Enum):
    OK = "ok"
    ERROR = "error"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class SpanContext:
    trace_id: str
    span_id: str
    trace_flags: int = 1
    trace_state: str = ""
    parent_span_id: Optional[str] = None


@dataclass(frozen=True, slots=True)
class Link:
    trace_id: str
    span_id: str
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Event:
    name: str
    timestamp: float = field(default_factory=time.time)
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Span:
    name: str
    kind: SpanKind = SpanKind.INTERNAL
    context: SpanContext = field(default_factory=lambda: SpanContext(
        trace_id=uuid.uuid4().hex,
        span_id=uuid.uuid4().hex[:16],
    ))
    parent_context: Optional[SpanContext] = None
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    status: SpanStatus = SpanStatus.OK
    status_message: str = ""
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[Event] = field(default_factory=list)
    links: List[Link] = field(default_factory=list)
    resource_attributes: Dict[str, Any] = field(default_factory=dict)

    def end(self, status: SpanStatus = SpanStatus.OK, message: str = ""):
        self.end_time = time.time()
        self.status = status
        self.status_message = message

    def add_event(self, name: str, attributes: Dict[str, Any] = None):
        self.events.append(Event(name=name, attributes=attributes or {}))

    def set_attribute(self, key: str, value: Any):
        self.attributes[key] = value

    def add_link(self, trace_id: str, span_id: str, attributes: Dict[str, Any] = None):
        self.links.append(Link(trace_id=trace_id, span_id=span_id, attributes=attributes or {}))

    def set_resource_attribute(self, key: str, value: Any):
        self.resource_attributes[key] = value

    @property
    def duration_ms(self) -> float:
        if self.end_time:
            return (self.end_time - self.start_time) * 1000
        return (time.time() - self.start_time) * 1000

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind.value,
            "context": {
                "trace_id": self.context.trace_id,
                "span_id": self.context.span_id,
                "trace_flags": self.context.trace_flags,
                "trace_state": self.context.trace_state,
                "parent_span_id": self.context.parent_span_id,
            },
            "parent_context": {
                "trace_id": self.parent_context.trace_id,
                "span_id": self.parent_context.span_id,
            } if self.parent_context else None,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "status": self.status.value,
            "status_message": self.status_message,
            "attributes": self.attributes,
            "events": [
                {"name": e.name, "timestamp": e.timestamp, "attributes": e.attributes}
                for e in self.events
            ],
            "links": [
                {"trace_id": l.trace_id, "span_id": l.span_id, "attributes": l.attributes}
                for l in self.links
            ],
            "resource_attributes": self.resource_attributes,
        }


class Tracer:
    def __init__(
        self,
        service_name: str,
        exporter: Optional[Any] = None,
        sampler: Optional[Callable[[SpanContext], bool]] = None,
        resource_attributes: Dict[str, Any] = None,
    ):
        self.service_name = service_name
        self.exporter = exporter
        self.sampler = sampler or (lambda ctx: True)
        self.resource_attributes = resource_attributes or {
            "service.name": service_name,
            "service.version": "1.0.0",
        }
        self._spans: List[Span] = []
        self._lock = asyncio.Lock()

    def start_span(
        self,
        name: str,
        kind: SpanKind = SpanKind.INTERNAL,
        parent: Optional[Span] = None,
        links: List[Link] = None,
        attributes: Dict[str, Any] = None,
    ) -> Span:
        parent_ctx = None
        if parent:
            parent_ctx = SpanContext(
                trace_id=parent.context.trace_id,
                span_id=parent.context.span_id,
                trace_flags=parent.context.trace_flags,
                trace_state=parent.context.trace_state,
                parent_span_id=parent.context.span_id,
            )

        context = SpanContext(
            trace_id=parent.context.trace_id if parent else uuid.uuid4().hex,
            span_id=uuid.uuid4().hex[:16],
            parent_span_id=parent.context.span_id if parent else None,
        )

        if not self.sampler(context):
            return Span(name=name, context=context, kind=kind, attributes={})

        span = Span(
            name=name,
            kind=kind,
            context=context,
            parent_context=parent_ctx,
            attributes=attributes or {},
            resource_attributes=self.resource_attributes,
        )
        if links:
            span.links.extend(links)

        self._spans.append(span)
        return span

    @contextmanager
    def trace(
        self,
        name: str,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Dict[str, Any] = None,
    ):
        span = self.start_span(name, kind=kind, attributes=attributes)
        try:
            yield span
            span.end()
        except Exception as e:
            span.end(SpanStatus.ERROR, str(e))
            raise

    @asynccontextmanager
    async def trace_async(
        self,
        name: str,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Dict[str, Any] = None,
    ):
        span = self.start_span(name, kind=kind, attributes=attributes)
        try:
            yield span
            span.end()
        except Exception as e:
            span.end(SpanStatus.ERROR, str(e))
            raise

    async def export(self):
        if self.exporter:
            spans_to_export = []
            async with self._lock:
                spans_to_export = self._spans.copy()
                self._spans.clear()
            await self.exporter.export(spans_to_export)


class CurrentSpan:
    _current: contextvars.ContextVar[Optional[Span]] = contextvars.ContextVar("_current", default=None)

    @classmethod
    def get(cls) -> Optional[Span]:
        return cls._current.get()

    @classmethod
    def set(cls, span: Optional[Span]):
        cls._current.set(span)

    @classmethod
    @contextmanager
    def use(cls, span: Span):
        token = cls._current.set(span)
        try:
            yield span
        finally:
            cls._current.reset(token)


def traced(
    name: Optional[str] = None,
    kind: SpanKind = SpanKind.INTERNAL,
    attributes: Dict[str, Any] = None,
    tracer: Optional[Tracer] = None,
):
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            nonlocal tracer
            if tracer is None:
                return func(*args, **kwargs)
            span_name = name or f"{func.__module__}.{func.__qualname__}"
            with tracer.trace(name=span_name, kind=kind, attributes=attributes) as span:
                span.set_attribute("function", func.__qualname__)
                span.set_attribute("module", func.__module__)
                try:
                    result = func(*args, **kwargs)
                    span.set_attribute("result_type", type(result).__name__)
                    return result
                except Exception as e:
                    span.set_attribute("error", str(e))
                    raise

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            nonlocal tracer
            if tracer is None:
                return await func(*args, **kwargs)
            span_name = name or f"{func.__module__}.{func.__qualname__}"
            async with tracer.trace_async(name=span_name, kind=kind, attributes=attributes) as span:
                span.set_attribute("function", func.__qualname__)
                span.set_attribute("module", func.__module__)
                try:
                    result = await func(*args, **kwargs)
                    span.set_attribute("result_type", type(result).__name__)
                    return result
                except Exception as e:
                    span.set_attribute("error", str(e))
                    raise

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    return decorator


class ConsoleExporter:
    def __init__(self, pretty: bool = False):
        self.pretty = pretty

    async def export(self, spans: List[Span]):
        for span in spans:
            data = span.to_dict()
            if self.pretty:
                print(json.dumps(data, indent=2, default=str))
            else:
                print(json.dumps(data, default=str))


class OTLPExporter:
    def __init__(
        self,
        endpoint: str = "http://localhost:4318/v1/traces",
        headers: Dict[str, str] = None,
        timeout: float = 10.0,
    ):
        self.endpoint = endpoint
        self.headers = headers or {"Content-Type": "application/json"}
        self.timeout = timeout
        self._session = None

    async def export(self, spans: List[Span]):
        import aiohttp
        if self._session is None:
            self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout))

        traces = [{
            "resource_spans": [{
                "resource": {
                    "attributes": [
                        {"key": k, "value": {"string_value": str(v)}}
                        for k, v in span.resource_attributes.items()
                    ]
                },
                "scope_spans": [{
                    "scope": {"name": "tracer", "version": "1.0.0"},
                    "spans": [span.to_dict() for span in spans],
                }],
            }]
        }]

        async with self._session.post(
            self.endpoint,
            json=traces,
            headers=self.headers,
        ) as resp:
            if resp.status >= 400:
                raise RuntimeError(f"OTLP export failed: {resp.status}")


class InMemorySpanStore:
    def __init__(self, max_spans: int = 10000):
        self.max_spans = max_spans
        self._spans: List[Span] = []
        self._by_trace: Dict[str, List[Span]] = defaultdict(list)
        self._lock = asyncio.Lock()

    async def add(self, span: Span):
        async with self._lock:
            self._spans.append(span)
            self._by_trace[span.context.trace_id].append(span)
            if len(self._spans) > self.max_spans:
                old = self._spans.pop(0)
                self._by_trace[old.context.trace_id].remove(old)

    async def get_trace(self, trace_id: str) -> List[Span]:
        async with self._lock:
            return self._by_trace.get(trace_id, []).copy()

    async def search(self, query: Dict[str, Any], limit: int = 100) -> List[Span]:
        async with self._lock:
            results = []
            for span in reversed(self._spans):
                if self._matches(span, query):
                    results.append(span)
                    if len(results) >= limit:
                        break
            return results

    def _matches(self, span: Span, query: Dict[str, Any]) -> bool:
        if "service" in query and query["service"] != span.resource_attributes.get("service.name"):
            return False
        if "name" in query and query["name"] not in span.name:
            return False
        if "status" in query and query["status"] != span.status.value:
            return False
        if "min_duration" in query and span.duration_ms < query["min_duration"]:
            return False
        if "max_duration" in query and span.duration_ms > query["max_duration"]:
            return False
        for k, v in query.get("attributes", {}).items():
            if span.attributes.get(k) != v:
                return False
        return True


current_trace: contextvars.ContextVar[Optional[Span]] = contextvars.ContextVar("current_trace", default=None)


def get_current_span() -> Optional[Span]:
    return current_trace.get()


def set_current_span(span: Optional[Span]):
    current_trace.set(span)


@contextmanager
def trace_context(span: Span):
    token = current_trace.set(span)
    try:
        yield
    finally:
        current_trace.reset(token)


class Baggage:
    _baggage: contextvars.ContextVar[Dict[str, str]] = contextvars.ContextVar("baggage", default={})

    @classmethod
    def get(cls, key: str) -> Optional[str]:
        return cls._baggage.get().get(key)

    @classmethod
    def set(cls, key: str, value: str):
        current = cls._baggage.get().copy()
        current[key] = value
        cls._baggage.set(current)

    @classmethod
    def get_all(cls) -> Dict[str, str]:
        return cls._baggage.get().copy()

    @classmethod
    def clear(cls):
        cls._baggage.set({})

    @classmethod
    @contextmanager
    def use(cls, items: Dict[str, str]):
        token = cls._baggage.set({**cls._baggage.get(), **items})
        try:
            yield
        finally:
            cls._baggage.reset(token)


def inject_carrier(carrier: Dict[str, str], span: Span = None):
    span = span or current_trace.get()
    if span:
        carrier["traceparent"] = f"00-{span.context.trace_id}-{span.context.span_id}-{span.context.trace_flags:02x}"
        if span.context.trace_state:
            carrier["tracestate"] = span.context.trace_state
    for key, value in Baggage.get_all().items():
        carrier[f"baggage-{key}"] = value


def extract_carrier(carrier: Dict[str, str]) -> Optional[SpanContext]:
    traceparent = carrier.get("traceparent") or carrier.get("trace-parent")
    if not traceparent:
        return None

    parts = traceparent.split("-")
    if len(parts) < 3:
        return None

    version, trace_id, span_id, flags = parts[0], parts[1], parts[2], parts[3] if len(parts) > 3 else "00"

    tracestate = carrier.get("tracestate") or carrier.get("trace-state")

    return SpanContext(
        trace_id=trace_id,
        span_id=span_id,
        trace_flags=int(flags, 16),
        trace_state=tracestate or "",
    )


def create_global_tracer(
    service_name: str,
    exporter_type: str = "console",
    endpoint: str = None,
) -> Tracer:
    if exporter_type == "console":
        exporter = ConsoleExporter(pretty=True)
    elif exporter_type == "otlp":
        exporter = OTLPExporter(endpoint=endpoint or "http://localhost:4318/v1/traces")
    else:
        exporter = None

    return Tracer(
        service_name=service_name,
        exporter=exporter,
        sampler=lambda ctx: True,
        resource_attributes={
            "service.name": service_name,
            "service.version": "1.0.0",
            "deployment.environment": os.getenv("APP_ENV", "development"),
        },
    )


class SpanProcessor:
    def __init__(self, exporter: Any, max_queue_size: int = 2048, schedule_delay_ms: int = 5000):
        self.exporter = exporter
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue_size)
        self.schedule_delay = schedule_delay_ms / 1000.0
        self._running = False
        self._task: Optional[asyncio.Task] = None

    async def start(self):
        self._running = True
        self._task = asyncio.create_task(self._process_loop())

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    def on_end(self, span: Span):
        try:
            self.queue.put_nowait(span)
        except asyncio.QueueFull:
            logger.warning("Span processor queue full, dropping span")

    async def _process_loop(self):
        batch = []
        while True:
            try:
                span = await asyncio.wait_for(self.queue.get(), timeout=self.schedule_delay)
                batch.append(span)
                while len(batch) < 512:
                    try:
                        batch.append(self.queue.get_nowait())
                    except asyncio.QueueEmpty:
                        break
            except asyncio.TimeoutError:
                pass

            if batch:
                try:
                    await self.exporter.export(batch)
                except Exception as e:
                    logger.error(f"Failed to export spans: {e}")
                batch.clear()


def create_tracing_pipeline(
    service_name: str,
    exporter_type: str = "console",
    endpoint: str = None,
) -> Tuple[Tracer, SpanProcessor]:
    tracer = create_global_tracer(service_name, "none")
    exporter = ConsoleExporter(pretty=True) if exporter_type == "console" else OTLPExporter(endpoint=endpoint)
    processor = SpanProcessor(exporter)
    return tracer, processor