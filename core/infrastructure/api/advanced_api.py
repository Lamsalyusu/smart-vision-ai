from __future__ import annotations

import asyncio
import time
import json
import uuid
from dataclasses import dataclass, field
from typing import Optional, Sequence, Callable, Dict, Any, List, TypeVar, Generic, Union
from enum import Enum
from abc import ABC, abstractmethod
from collections import defaultdict, deque
import inspect
import functools


T = TypeVar("T")


class HttpMethod(Enum):
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    PATCH = "PATCH"
    DELETE = "DELETE"
    HEAD = "HEAD"
    OPTIONS = "OPTIONS"


class ContentType(Enum):
    JSON = "application/json"
    FORM = "application/x-www-form-urlencoded"
    MULTIPART = "multipart/form-data"
    TEXT = "text/plain"
    XML = "application/xml"
    OCTET_STREAM = "application/octet-stream"


@dataclass
class Request:
    method: HttpMethod
    path: str
    headers: Dict[str, str] = field(default_factory=dict)
    query_params: Dict[str, str] = field(default_factory=dict)
    body: Optional[bytes] = None
    path_params: Dict[str, str] = field(default_factory=dict)
    state: Dict[str, Any] = field(default_factory=dict)

    @property
    def content_type(self) -> Optional[str]:
        return self.headers.get("content-type")

    def json(self) -> Any:
        if self.body:
            return json.loads(self.body.decode())
        return None

    def form(self) -> Dict[str, str]:
        if self.body and self.content_type == ContentType.FORM.value:
            return dict(pair.split("=") for pair in self.body.decode().split("&"))
        return {}

    def multipart(self) -> Dict[str, Any]:
        return {}


@dataclass
class Response:
    status_code: int = 200
    headers: Dict[str, str] = field(default_factory=dict)
    body: Optional[bytes] = None
    content_type: Optional[str] = None

    def set_json(self, data: Any):
        self.body = json.dumps(data).encode()
        self.content_type = ContentType.JSON.value
        self.headers["content-type"] = self.content_type

    def set_text(self, text: str):
        self.body = text.encode()
        self.content_type = ContentType.TEXT.value
        self.headers["content-type"] = self.content_type

    def set_html(self, html: str):
        self.body = html.encode()
        self.content_type = "text/html"
        self.headers["content-type"] = self.content_type

    def set_bytes(self, data: bytes, content_type: str = ContentType.OCTET_STREAM.value):
        self.body = data
        self.content_type = content_type
        self.headers["content-type"] = content_type

    def redirect(self, url: str, status_code: int = 302):
        self.status_code = status_code
        self.headers["location"] = url


class Router:
    def __init__(self):
        self.routes: Dict[str, Dict[HttpMethod, Callable]] = defaultdict(dict)
        self.middleware: List[Callable] = []
        self.exception_handlers: Dict[type, Callable] = {}

    def add_route(self, path: str, method: HttpMethod, handler: Callable):
        self.routes[path][method] = handler

    def get(self, path: str, handler: Callable):
        self.add_route(path, HttpMethod.GET, handler)

    def post(self, path: str, handler: Callable):
        self.add_route(path, HttpMethod.POST, handler)

    def put(self, path: str, handler: Callable):
        self.add_route(path, HttpMethod.PUT, handler)

    def patch(self, path: str, handler: Callable):
        self.add_route(path, HttpMethod.PATCH, handler)

    def delete(self, path: str, handler: Callable):
        self.add_route(path, HttpMethod.DELETE, handler)

    def add_middleware(self, middleware: Callable):
        self.middleware.append(middleware)

    def add_exception_handler(self, exc_type: type, handler: Callable):
        self.exception_handlers[exc_type] = handler

    def match(self, path: str, method: HttpMethod) -> Optional[Tuple[Callable, Dict[str, str]]]:
        for route_path, methods in self.routes.items():
            if method in methods:
                params = self._match_path(route_path, path)
                if params is not None:
                    return methods[method], params
        return None

    def _match_path(self, route: str, path: str) -> Optional[Dict[str, str]]:
        route_parts = route.strip("/").split("/")
        path_parts = path.strip("/").split("/")

        if len(route_parts) != len(path_parts):
            return None

        params = {}
        for rp, pp in zip(route_parts, path_parts):
            if rp.startswith("{") and rp.endswith("}"):
                params[rp[1:-1]] = pp
            elif rp != pp:
                return None

        return params


class Application:
    def __init__(self):
        self.router = Router()
        self.startup_handlers: List[Callable] = []
        self.shutdown_handlers: List[Callable] = []
        self.state: Dict[str, Any] = {}

    def on_startup(self, handler: Callable):
        self.startup_handlers.append(handler)

    def on_shutdown(self, handler: Callable):
        self.shutdown_handlers.append(handler)

    def include_router(self, router: Router, prefix: str = ""):
        for path, methods in router.routes.items():
            full_path = prefix + path
            for method, handler in methods.items():
                self.router.add_route(full_path, method, handler)

    async def handle_request(self, request: Request) -> Response:
        for middleware in self.router.middleware:
            request = await middleware(request) or request

        match = self.router.match(request.path, request.method)
        if not match:
            return Response(status_code=404, body=b"Not Found")

        handler, path_params = match
        request.path_params = path_params

        try:
            if inspect.iscoroutinefunction(handler):
                response = await handler(request)
            else:
                response = handler(request)
            return response if isinstance(response, Response) else Response(body=str(response).encode())
        except Exception as e:
            for exc_type, handler in self.router.exception_handlers.items():
                if isinstance(e, exc_type):
                    return await handler(request, e) if inspect.iscoroutinefunction(handler) else handler(request, e)
            return Response(status_code=500, body=b"Internal Server Error")

    async def startup(self):
        for handler in self.startup_handlers:
            if inspect.iscoroutinefunction(handler):
                await handler()
            else:
                handler()

    async def shutdown(self):
        for handler in self.shutdown_handlers:
            if inspect.iscoroutinefunction(handler):
                await handler()
            else:
                handler()


class RequestValidator:
    def __init__(self):
        self.validators: Dict[str, Callable] = {}

    def add_validator(self, name: str, validator: Callable[[Any], bool]):
        self.validators[name] = validator

    def validate(self, data: Any, rules: List[str]) -> Tuple[bool, List[str]]:
        errors = []
        for rule in rules:
            if rule in self.validators:
                if not self.validators[rule](data):
                    errors.append(f"Validation failed for rule: {rule}")
        return len(errors) == 0, errors


class RateLimitMiddleware:
    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.requests: Dict[str, List[float]] = defaultdict(list)
        self.lock = asyncio.Lock()

    async def __call__(self, request: Request) -> Optional[Request]:
        client_ip = request.headers.get("x-forwarded-for", request.headers.get("x-real-ip", "unknown"))
        now = time.time()

        async with self.lock:
            requests = self.requests[client_ip]
            requests[:] = [t for t in requests if now - t < 60]

            if len(requests) >= self.requests_per_minute:
                return None

            requests.append(now)
        return request


class CorsMiddleware:
    def __init__(
        self,
        allow_origins: List[str] = None,
        allow_methods: List[str] = None,
        allow_headers: List[str] = None,
        allow_credentials: bool = False,
    ):
        self.allow_origins = allow_origins or ["*"]
        self.allow_methods = allow_methods or ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
        self.allow_headers = allow_headers or ["*"]
        self.allow_credentials = allow_credentials

    async def __call__(self, request: Request) -> Optional[Request]:
        origin = request.headers.get("origin")
        if origin and (self.allow_origins == ["*"] or origin in self.allow_origins):
            request.state["cors_origin"] = origin
        return request

    def add_cors_headers(self, response: Response, request: Request):
        origin = request.state.get("cors_origin")
        if origin:
            response.headers["access-control-allow-origin"] = origin
            response.headers["access-control-allow-methods"] = ", ".join(self.allow_methods)
            response.headers["access-control-allow-headers"] = ", ".join(self.allow_headers)
            if self.allow_credentials:
                response.headers["access-control-allow-credentials"] = "true"


class AuthMiddleware:
    def __init__(self, secret_key: str, excluded_paths: List[str] = None):
        self.secret_key = secret_key
        self.excluded_paths = excluded_paths or ["/health", "/docs", "/openapi.json"]

    async def __call__(self, request: Request) -> Optional[Request]:
        if request.path in self.excluded_paths:
            return request

        auth_header = request.headers.get("authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return None

        token = auth_header[7:]
        try:
            import jwt
            payload = jwt.decode(token, self.secret_key, algorithms=["HS256"])
            request.state["user"] = payload
            return request
        except jwt.InvalidTokenError:
            return None


class CompressionMiddleware:
    def __init__(self, minimum_size: int = 500):
        self.minimum_size = minimum_size

    async def __call__(self, request: Request) -> Optional[Request]:
        accept_encoding = request.headers.get("accept-encoding", "")
        if "gzip" in accept_encoding:
            request.state["gzip"] = True
        return request

    def compress_response(self, response: Response, request: Request):
        if request.state.get("gzip") and response.body and len(response.body) >= self.minimum_size:
            import gzip
            response.body = gzip.compress(response.body)
            response.headers["content-encoding"] = "gzip"


class RequestIdMiddleware:
    async def __call__(self, request: Request) -> Optional[Request]:
        request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
        request.state["request_id"] = request_id
        return request

    def add_request_id_header(self, response: Response, request: Request):
        response.headers["x-request-id"] = request.state.get("request_id", "")


class LoggingMiddleware:
    def __init__(self, logger=None):
        self.logger = logger or print

    async def __call__(self, request: Request) -> Optional[Request]:
        start = time.time()
        request.state["start_time"] = start
        return request

    def log_response(self, request: Request, response: Response):
        duration = time.time() - request.state.get("start_time", time.time())
        self.logger(f"{request.method.value} {request.path} - {response.status_code} - {duration:.3f}s")


class WebSocketManager:
    def __init__(self):
        self.connections: Dict[str, Any] = {}
        self.rooms: Dict[str, Set[str]] = defaultdict(set)

    async def connect(self, websocket, client_id: str, room: str = "default"):
        await websocket.accept()
        self.connections[client_id] = websocket
        self.rooms[room].add(client_id)

    def disconnect(self, client_id: str):
        for room in self.rooms.values():
            room.discard(client_id)
        self.connections.pop(client_id, None)

    async def send_personal(self, client_id: str, message: Any):
        if client_id in self.connections:
            websocket = self.connections[client_id]
            await websocket.send_json(message) if isinstance(message, dict) else await websocket.send_text(str(message))

    async def broadcast(self, room: str, message: Any, exclude: str = None):
        for client_id in self.rooms.get(room, set()):
            if client_id != exclude and client_id in self.connections:
                websocket = self.connections[client_id]
                await websocket.send_json(message) if isinstance(message, dict) else await websocket.send_text(str(message))

    async def broadcast_all(self, message: Any, exclude: str = None):
        for client_id, websocket in self.connections.items():
            if client_id != exclude:
                await websocket.send_json(message) if isinstance(message, dict) else await websocket.send_text(str(message))


class SSEManager:
    def __init__(self):
        self.connections: Dict[str, asyncio.Queue] = {}

    async def connect(self, client_id: str) -> asyncio.Queue:
        queue = asyncio.Queue()
        self.connections[client_id] = queue
        return queue

    def disconnect(self, client_id: str):
        self.connections.pop(client_id, None)

    async def send(self, client_id: str, event: str, data: Any):
        if client_id in self.connections:
            await self.connections[client_id].put({"event": event, "data": data})

    async def broadcast(self, event: str, data: Any):
        for queue in self.connections.values():
            await queue.put({"event": event, "data": data})


class GraphQLResolver:
    def __init__(self):
        self.queries: Dict[str, Callable] = {}
        self.mutations: Dict[str, Callable] = {}
        self.subscriptions: Dict[str, Callable] = {}

    def query(self, name: str):
        def decorator(func: Callable):
            self.queries[name] = func
            return func
        return decorator

    def mutation(self, name: str):
        def decorator(func: Callable):
            self.mutations[name] = func
            return func
        return decorator

    def subscription(self, name: str):
        def decorator(func: Callable):
            self.subscriptions[name] = func
            return func
        return decorator

    async def execute(self, query: str, variables: Dict = None) -> Dict:
        return {"data": {}, "errors": []}


class OpenAPISpec:
    def __init__(self, title: str = "API", version: str = "1.0.0"):
        self.title = title
        self.version = version
        self.paths: Dict[str, Dict] = {}
        self.components: Dict[str, Any] = {"schemas": {}, "securitySchemes": {}}
        self.servers: List[Dict] = [{"url": "/"}]

    def add_path(self, path: str, method: HttpMethod, handler: Callable, summary: str = "", description: str = ""):
        if path not in self.paths:
            self.paths[path] = {}

        self.paths[path][method.value.lower()] = {
            "summary": summary,
            "description": description,
            "responses": {"200": {"description": "Successful response"}},
        }

    def add_schema(self, name: str, schema: Dict):
        self.components["schemas"][name] = schema

    def add_security_scheme(self, name: str, scheme: Dict):
        self.components["securitySchemes"][name] = scheme

    def to_dict(self) -> Dict:
        return {
            "openapi": "3.0.0",
            "info": {"title": self.title, "version": self.version},
            "servers": self.servers,
            "paths": self.paths,
            "components": self.components,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


class APIGateway:
    def __init__(self):
        self.services: Dict[str, str] = {}
        self.routes: Dict[str, str] = {}
        self.load_balancer = LoadBalancer()

    def register_service(self, name: str, url: str):
        self.services[name] = url

    def add_route(self, path: str, service: str):
        self.routes[path] = service

    async def route_request(self, request: Request) -> Response:
        for path, service in self.routes.items():
            if request.path.startswith(path):
                target_url = self.services.get(service)
                if not target_url:
                    return Response(status_code=503, body=b"Service unavailable")

                return await self._proxy_request(request, target_url)

        return Response(status_code=404, body=b"Not Found")

    async def _proxy_request(self, request: Request, target_url: str) -> Response:
        import httpx
        url = target_url.rstrip("/") + request.path
        async with httpx.AsyncClient() as client:
            response = await client.request(
                method=request.method.value,
                url=url,
                headers=dict(request.headers),
                params=request.query_params,
                content=request.body,
            )
            return Response(
                status_code=response.status_code,
                headers=dict(response.headers),
                body=response.content,
            )


class LoadBalancer:
    def __init__(self):
        self.backends: List[str] = []
        self.current = 0
        self.lock = asyncio.Lock()

    def add_backend(self, url: str):
        self.backends.append(url)

    async def get_backend(self) -> Optional[str]:
        async with self.lock:
            if not self.backends:
                return None
            backend = self.backends[self.current]
            self.current = (self.current + 1) % len(self.backends)
            return backend
