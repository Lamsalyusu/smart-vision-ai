from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Sequence, Callable, Dict, Any, List, Tuple, Set
from enum import Enum
from abc import ABC, abstractmethod
import math
import time
import threading
import json
import asyncio
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
import uuid


class HTTPMethod(Enum):
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"
    HEAD = "HEAD"
    OPTIONS = "OPTIONS"


class APIVersion(Enum):
    V1 = "v1"
    V2 = "v2"


@dataclass(frozen=True, slots=True)
class APIRequest:
    method: HTTPMethod
    path: str
    headers: Dict[str, str] = field(default_factory=dict)
    query_params: Dict[str, str] = field(default_factory=dict)
    body: Any = None
    path_params: Dict[str, str] = field(default_factory=dict)
    client_ip: str = ""
    user_agent: str = ""
    timestamp: float = field(default_factory=time.time)
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass(frozen=True, slots=True)
class APIResponse:
    status_code: int
    body: Any = None
    headers: Dict[str, str] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    request_id: str = ""


class Route:
    def __init__(
        self,
        method: HTTPMethod,
        path: str,
        handler: Callable,
        middleware: List[Callable] = None,
        auth_required: bool = False,
        rate_limit: Optional[int] = None,
        version: APIVersion = APIVersion.V1
    ):
        self.method = method
        self.path = path
        self.handler = handler
        self.middleware = middleware or []
        self.auth_required = auth_required
        self.rate_limit = rate_limit
        self.version = version

    def matches(self, method: HTTPMethod, path: str) -> bool:
        if self.method != method:
            return False

        path_parts = self.path.strip('/').split('/')
        request_parts = path.strip('/').split('/')

        if len(path_parts) != len(request_parts):
            return False

        for pattern, request_part in zip(path_parts, request_parts):
            if pattern.startswith('{') and pattern.endswith('}'):
                continue
            if pattern != request_part:
                return False

        return True

    def extract_params(self, path: str) -> Dict[str, str]:
        path_parts = self.path.strip('/').split('/')
        request_parts = path.strip('/').split('/')

        params = {}
        for pattern, request_part in zip(path_parts, request_parts):
            if pattern.startswith('{') and pattern.endswith('}'):
                param_name = pattern[1:-1]
                params[param_name] = request_part

        return params


class Router:
    def __init__(self):
        self.routes: List[Route] = []
        self.middleware: List[Callable] = []

    def add_route(self, route: Route):
        self.routes.append(route)
        self.routes.sort(key=lambda r: len(r.path.split('/')), reverse=True)

    def add_middleware(self, middleware: Callable):
        self.middleware.append(middleware)

    def route(self, method: HTTPMethod, path: str) -> Tuple[Optional[Route], Dict[str, str]]:
        for route in self.routes:
            if route.matches(method, path):
                params = route.extract_params(path)
                return route, params

        return None, {}

    def add_route(
        self,
        method: HTTPMethod,
        path: str,
        handler: Callable,
        middleware: List[Callable] = None,
        auth_required: bool = False,
        rate_limit: Optional[int] = None,
        version: APIVersion = APIVersion.V1
    ) -> Route:
        route = Route(method, path, handler, middleware, auth_required, rate_limit, version)
        self.add_route(route)
        return route

    def get(self, path: str, handler: Callable, **kwargs) -> Route:
        return self.add_route(HTTPMethod.GET, path, handler, **kwargs)

    def post(self, path: str, handler: Callable, **kwargs) -> Route:
        return self.add_route(HTTPMethod.POST, path, handler, **kwargs)

    def put(self, path: str, handler: Callable, **kwargs) -> Route:
        return self.add_route(HTTPMethod.PUT, path, handler, **kwargs)

    def delete(self, path: str, handler: Callable, **kwargs) -> Route:
        return self.add_route(HTTPMethod.DELETE, path, handler, **kwargs)

    def patch(self, path: str, handler: Callable, **kwargs) -> Route:
        return self.add_route(HTTPMethod.PATCH, path, handler, **kwargs)


class Middleware:
    def __init__(self, name: str):
        self.name = name

    async def __call__(self, request: APIRequest, call_next: Callable) -> APIResponse:
        return await call_next(request)


class RateLimitMiddleware(Middleware):
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        super().__init__("rate_limit")
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: Dict[str, List[float]] = defaultdict(list)
        self.lock = threading.RLock()

    async def __call__(self, request: APIRequest, call_next: Callable) -> APIResponse:
        client_ip = request.client_ip
        now = time.time()

        with self.lock:
            self.requests[client_ip] = [
                t for t in self.requests[client_ip] if now - t < self.window_seconds
            ]

            if len(self.requests[client_ip]) >= self.max_requests:
                return APIResponse(
                    status_code=429,
                    body={"error": "Rate limit exceeded"},
                    headers={"Retry-After": str(self.window_seconds)}
                )

            self.requests[client_ip].append(now)

        return await call_next(request)


class AuthMiddleware(Middleware):
    def __init__(self, secret_key: str, algorithm: str = "HS256"):
        super().__init__("auth")
        self.secret_key = secret_key
        self.algorithm = algorithm

    async def __call__(self, request: APIRequest, call_next: Callable) -> APIResponse:
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return APIResponse(
                status_code=401,
                body={"error": "Missing or invalid authorization header"}
            )

        token = auth_header[7:]
        try:
            import jwt
            payload = jwt.decode(request, self.secret_key, algorithms=[self.algorithm])
            request.metadata["user"] = payload
        except Exception:
            return APIResponse(
                status_code=401,
                body={"error": "Invalid or expired token"}
            )

        return await call_next(request)


class CORSMiddleware(Middleware):
    def __init__(
        self,
        allow_origins: List[str] = None,
        allow_methods: List[str] = None,
        allow_headers: List[str] = None,
        allow_credentials: bool = True
    ):
        super().__init__("cors")
        self.allow_origins = allow_origins or ["*"]
        self.allow_methods = allow_methods or ["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"]
        self.allow_headers = allow_headers or ["Content-Type", "Authorization"]
        self.allow_credentials = allow_credentials

    async def __call__(self, request: APIRequest, call_next: Callable) -> APIResponse:
        if request.method == HTTPMethod.OPTIONS:
            return APIResponse(
                status_code=200,
                headers=self._get_cors_headers(request)
            )

        response = await call_next(request)
        response.headers.update(self._get_cors_headers(request))
        return response

    def _get_cors_headers(self, request: APIRequest) -> Dict[str, str]:
        origin = request.headers.get("Origin", "")
        allow_origin = "*"
        if origin in self.allow_origins:
            allow_origin = origin

        return {
            "Access-Control-Allow-Origin": allow_origin,
            "Access-Control-Allow-Methods": ", ".join(self.allow_methods),
            "Access-Control-Allow-Headers": ", ".join(self.allow_headers),
            "Access-Control-Allow-Credentials": str(self.allow_credentials).lower()
        }


class LoggingMiddleware(Middleware):
    def __init__(self):
        super().__init__("logging")

    async def __call__(self, request: APIRequest, call_next: Callable) -> APIResponse:
        start_time = time.time()

        response = await call_next(request)

        duration = time.time() - start_time
        print(f"{request.method.value} {request.path} - {response.status_code} - {duration:.3f}s")

        return response


class ErrorHandlingMiddleware(Middleware):
    def __init__(self):
        super().__init__("error_handling")

    async def __call__(self, request: APIRequest, call_next: Callable) -> APIResponse:
        try:
            return await call_next(request)
        except Exception as e:
            return APIResponse(
                status_code=500,
                body={"error": "Internal server error", "message": str(e)},
                headers={"Content-Type": "application/json"}
            )


class APIServer:
    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8080,
        title: str = "Smart Vision AI API",
        version: str = "1.0.0"
    ):
        self.host = host
        self.port = port
        self.title = title
        self.version = version

        self.router = Router()
        self.middleware: List[Middleware] = []
        self.running = False
        self.server = None

        self._setup_default_routes()

    def _setup_default_routes(self):
        self.router.get("/health", self._health_check)
        self.router.get("/version", self._version_info)

    async def _health_check(self, request: APIRequest) -> APIResponse:
        return APIResponse(
            status_code=200,
            body={"status": "healthy", "timestamp": time.time()}
        )

    async def _version_info(self, request: APIRequest) -> APIResponse:
        return APIResponse(
            status_code=200,
            body={
                "title": self.title,
                "version": self.version,
                "timestamp": time.time()
            }
        )

    def add_middleware(self, middleware: Middleware):
        self.middleware.append(middleware)

    def add_route(
        self,
        method: HTTPMethod,
        path: str,
        handler: Callable,
        middleware: List[Callable] = None,
        auth_required: bool = False,
        rate_limit: Optional[int] = None,
        version: APIVersion = APIVersion.V1
    ) -> Route:
        return self.router.add_route(method, path, handler, middleware, auth_required, rate_limit, version)

    def get(self, path: str, handler: Callable, **kwargs) -> Route:
        return self.router.get(path, handler, **kwargs)

    def post(self, path: str, handler: Callable, **kwargs) -> Route:
        return self.router.post(path, handler, **kwargs)

    def put(self, path: str, handler: Callable, **kwargs) -> Route:
        return self.router.put(path, handler, **kwargs)

    def delete(self, path: str, handler: Callable, **kwargs) -> Route:
        return self.router.delete(path, handler, **kwargs)

    def patch(self, path: str, handler: Callable, **kwargs) -> Route:
        return self.router.patch(path, handler, **kwargs)

    async def handle_request(self, request: APIRequest) -> APIResponse:
        route, params = self.router.route(request.method, request.path)

        if not route:
            return APIResponse(
                status_code=404,
                body={"error": "Not found"}
            )

        request.path_params = params

        for middleware in self.middleware:
            request = await middleware(request, lambda r: r)

        if route.auth_required:
            auth_result = self._check_auth(request)
            if not auth_result:
                return APIResponse(
                    status_code=401,
                    body={"error": "Unauthorized"}
                )

        try:
            result = await route.handler(request)
            if isinstance(result, APIResponse):
                return result
            return APIResponse(status_code=200, body=result)
        except Exception as e:
            return APIResponse(
                status_code=500,
                body={"error": str(e)}
            )

    def _check_auth(self, request: APIRequest) -> bool:
        auth_header = request.headers.get("Authorization", "")
        return auth_header.startswith("Bearer ")

    def start(self):
        self.running = True
        print(f"Starting API server on {self.host}:{self.port}")

    def stop(self):
        self.running = False


class APIClient:
    def __init__(self, base_url: str, api_key: str = None, timeout: float = 30.0):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout

    def _request(
        self,
        method: HTTPMethod,
        path: str,
        params: Dict = None,
        data: Any = None,
        headers: Dict = None
    ) -> Dict:
        import requests

        url = f"{self.base_url}{path}"
        headers = headers or {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        response = requests.request(
            method=method.value,
            url=url,
            params=params,
            json=data,
            headers=headers,
            timeout=self.timeout
        )

        if response.status_code >= 400:
            raise Exception(f"API Error: {response.status_code} - {response.text}")

        return response.json()

    def get(self, path: str, params: Dict = None) -> Dict:
        return self._request(HTTPMethod.GET, path, params=params)

    def post(self, path: str, data: Any = None) -> Dict:
        return self._request(HTTPMethod.POST, path, data=data)

    def put(self, path: str, data: Any = None) -> Dict:
        return self._request(HTTPMethod.PUT, path, data=data)

    def delete(self, path: str) -> Dict:
        return self._request(HTTPMethod.DELETE, path)

    def patch(self, path: str, data: Any = None) -> Dict:
        return self._request(HTTPMethod.PATCH, path, data=data)


class APIGateway:
    def __init__(self):
        self.services: Dict[str, APIClient] = {}
        self.routes: Dict[str, str] = {}
        self.load_balancer = None

    def register_service(self, name: str, client: APIClient, routes: List[str]):
        self.services[name] = client
        for route in routes:
            self.routes[route] = name

    def route_request(self, method: HTTPMethod, path: str, **kwargs) -> Dict:
        service_name = self.routes.get(path)
        if not service_name:
            for route, service in self.routes.items():
                if path.startswith(route):
                    service_name = service
                    break

        if not service_name or service_name not in self.services:
            raise ValueError(f"No service found for path: {path}")

        client = self.services[service_name]
        return client._request(method, path, **kwargs)

    def get(self, path: str, **kwargs) -> Dict:
        return self.route_request(HTTPMethod.GET, path, **kwargs)

    def post(self, path: str, data: Any = None) -> Dict:
        return self.route_request(HTTPMethod.POST, path, data=data)

    def put(self, path: str, data: Any = None) -> Dict:
        return self.route_request(HTTPMethod.PUT, path, data=data)

    def delete(self, path: str) -> Dict:
        return self.route_request(HTTPMethod.DELETE, path)


class RateLimiter:
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: Dict[str, List[float]] = defaultdict(list)
        self.lock = threading.RLock()

    def is_allowed(self, key: str) -> bool:
        now = time.time()
        with self.lock:
            self.requests[key] = [
                t for t in self.requests[key] if now - t < self.window_seconds
            ]

            if len(self.requests[key]) >= self.max_requests:
                return False

            self.requests[key].append(now)
            return True

    def get_remaining(self, key: str) -> int:
        now = time.time()
        with self.lock:
            self.requests[key] = [
                t for t in self.requests[key] if now - t < self.window_seconds
            ]
            return max(0, self.max_requests - len(self.requests[key]))


class APIVersionManager:
    def __init__(self):
        self.versions: Dict[str, Router] = {}
        self.default_version = "v1"

    def register_version(self, version: str, router: Router):
        self.versions[version] = router

    def get_router(self, version: str) -> Optional[Router]:
        return self.versions.get(version)

    def set_default_version(self, version: str):
        if version in self.versions:
            self.default_version = version

    def route_request(self, request: APIRequest) -> Optional[Route]:
        version = request.headers.get("API-Version", self.default_version)
        router = self.versions.get(version)
        if router:
            return router.route(request.method, request.path)
        return None


class APIDocumentation:
    def __init__(self, title: str = "API Documentation", version: str = "1.0.0"):
        self.title = title
        self.version = version
        self.endpoints: List[Dict] = []

    def add_endpoint(
        self,
        method: HTTPMethod,
        path: str,
        summary: str,
        description: str = "",
        parameters: List[Dict] = None,
        responses: Dict[int, Dict] = None,
        tags: List[str] = None
    ):
        self.endpoints.append({
            "method": method.value,
            "path": path,
            "summary": summary,
            "description": description,
            "parameters": parameters or [],
            "responses": responses or {},
            "tags": tags or []
        })

    def generate_openapi_spec(self) -> Dict:
        return {
            "openapi": "3.0.0",
            "info": {
                "title": self.title,
                "version": self.version
            },
            "paths": self._generate_paths(),
            "components": {
                "schemas": {},
                "securitySchemes": {
                    "BearerAuth": {
                        "type": "http",
                        "scheme": "bearer"
                    }
                }
            }
        }

    def _generate_paths(self) -> Dict:
        paths = {}
        for endpoint in self.endpoints:
            if endpoint["path"] not in paths:
                paths[endpoint["path"]] = {}

            paths[endpoint["path"]][endpoint["method"].lower()] = {
                "summary": endpoint["summary"],
                "description": endpoint["description"],
                "parameters": endpoint["parameters"],
                "responses": endpoint["responses"],
                "tags": endpoint["tags"]
            }
        return paths

    def to_json(self) -> str:
        return json.dumps(self.generate_openapi_spec(), indent=2)

    def to_yaml(self) -> str:
        import yaml
        return yaml.dump(self.generate_openapi_spec())


class APIGateway:
    def __init__(self):
        self.services: Dict[str, Any] = {}
        self.routes: Dict[str, str] = {}
        self.load_balancer = None

    def register_service(self, name: str, base_url: str, routes: List[str]):
        from urllib.parse import urljoin
        self.services[name] = {
            "base_url": base_url.rstrip('/'),
            "routes": routes
        }
        for route in routes:
            self.routes[route] = name

    def route_request(self, method: HTTPMethod, path: str, **kwargs) -> Any:
        service_name = self.routes.get(path)
        if not service_name:
            for route, service in self.routes.items():
                if path.startswith(route):
                    service_name = service
                    break

        if not service_name or service_name not in self.services:
            raise ValueError(f"No service found for path: {path}")

        service = self.services[service_name]
        url = f"{service['base_url']}{path}"

        import requests
        response = requests.request(
            method=method.value,
            url=url,
            **kwargs
        )

        return response.json()


class OpenAPISpecGenerator:
    @staticmethod
    def generate_from_routes(routes: List[Route]) -> Dict:
        paths = {}
        for route in routes:
            if route.path not in paths:
                paths[route.path] = {}

            paths[route.path][route.method.value.lower()] = {
                "summary": getattr(route.handler, "__doc__", ""),
                "responses": {
                    "200": {"description": "Successful response"},
                    "400": {"description": "Bad request"},
                    "401": {"description": "Unauthorized"},
                    "404": {"description": "Not found"},
                    "500": {"description": "Internal server error"}
                }
            }

        return {
            "openapi": "3.0.0",
            "info": {
                "title": "Smart Vision AI API",
                "version": "1.0.0"
            },
            "paths": paths
        }

    @staticmethod
    def save_spec(spec: Dict, path: str, format: str = "json"):
        if format == "yaml":
            import yaml
            with open(path, 'w') as f:
                yaml.dump(spec, f)
        else:
            with open(path, 'w') as f:
                json.dump(spec, f, indent=2)


from core.domain.vision.feature_extraction import FeatureType, Keypoint, FeatureMatch, FeatureSet
