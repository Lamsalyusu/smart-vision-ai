from __future__ import annotations


class SmartVisionException(Exception):
    """Base exception for all Smart Vision AI SDK errors."""


class AuthenticationError(SmartVisionException):
    """Raised when authentication credentials or token are invalid."""


class RateLimitExceededError(SmartVisionException):
    """Raised when API rate limits are exceeded."""


class ModelInferenceError(SmartVisionException):
    """Raised when deep learning inference or 3D reconstruction fails."""


class StreamConnectionError(SmartVisionException):
    """Raised when WebSocket telemetry connection drops or fails to connect."""
