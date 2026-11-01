from __future__ import annotations

from typing import Any

import httpx

from .exceptions import (
    AuthenticationError,
    RateLimitExceededError,
    SmartVisionException,
)
from .models import (
    MeshReconstructionResult,
)


class SmartVisionClient:
    """Synchronous and asynchronous enterprise client for Smart Vision AI platform."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_token: str | None = None,
        timeout: float = 10.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_token = api_token
        self.timeout = timeout
        self._headers = {"Authorization": f"Bearer {api_token}"} if api_token else {}

    async def get_health(self) -> dict[str, Any]:
        """Check API service health status."""
        async with httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout) as client:
            resp = await client.get("/health")
            if resp.status_code != 200:
                raise SmartVisionException(f"Health check failed with status {resp.status_code}")
            return resp.json()

    async def reconstruct_3d_mesh(
        self,
        object_class: str = "chair",
        points: list[list[float]] | None = None,
        voxel_size: float = 0.08,
        smoothing_iterations: int = 2,
    ) -> MeshReconstructionResult:
        """Request high-fidelity Marching Cubes 3D surface reconstruction."""
        async with httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout, headers=self._headers) as client:
            payload = {
                "object_class": object_class,
                "points": points,
                "voxel_size": voxel_size,
                "smoothing_iterations": smoothing_iterations,
                "output_format": "threejs",
            }
            resp = await client.post("/mesh/reconstruct", json=payload)
            if resp.status_code == 401:
                raise AuthenticationError("Invalid or expired API token")
            elif resp.status_code == 429:
                raise RateLimitExceededError("Rate limit exceeded")
            elif resp.status_code != 200:
                raise SmartVisionException(f"Mesh reconstruction failed: {resp.text}")

            data = resp.json()
            stats = data.get("statistics", {})
            return MeshReconstructionResult(
                success=data.get("success", True),
                object_class=object_class,
                vertex_count=stats.get("vertex_count", 0),
                face_count=stats.get("face_count", 0),
                surface_area_sq_m=stats.get("surface_area_sq_m", 0.0),
                threejs_geometry=data.get("threejs_geometry"),
                obj_data=data.get("obj_data"),
            )
