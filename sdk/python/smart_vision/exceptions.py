from __future__ import annotations


class SmartVisionException(Exception):
    """Base exception for all Smart Vision AI SDK errors."""
    pass


class AuthenticationError(SmartVisionException):
    """Raised when authentication credentials or token are invalid."""
    pass


class RateLimitExceededError(SmartVisionException):
    """Raised when API rate limits are exceeded."""
    pass


class ModelInferenceError(SmartVisionException):
    """Raised when deep learning inference or 3D reconstruction fails."""
    pass


class StreamConnectionError(SmartVisionException):
    """Raised when WebSocket telemetry connection drops or fails to connect."""
    pass
