"""Outbound email over SMTP.

Configured with the same variables as the Django service:
EMAIL_HOST (default smtp.gmail.com), EMAIL_PORT (default 587, STARTTLS),
EMAIL_HOST_USER, EMAIL_HOST_PASSWORD, and optionally EMAIL_FROM and
EMAIL_USE_TLS (default true; set false only for a local test relay).
"""

from __future__ import annotations

import os
import smtplib
from email.message import EmailMessage


class EmailNotConfiguredError(RuntimeError):
    """Raised when SMTP credentials are missing."""


def email_configured() -> bool:
    return bool(os.getenv("EMAIL_HOST_USER") and os.getenv("EMAIL_HOST_PASSWORD"))


def send_email(to: str, subject: str, body: str) -> None:
    user = os.getenv("EMAIL_HOST_USER")
    password = os.getenv("EMAIL_HOST_PASSWORD")
    if not user or not password:
        raise EmailNotConfiguredError("EMAIL_HOST_USER and EMAIL_HOST_PASSWORD are not set")

    message = EmailMessage()
    message["From"] = os.getenv("EMAIL_FROM") or user
    message["To"] = to
    message["Subject"] = subject
    message.set_content(body)

    host = os.getenv("EMAIL_HOST", "smtp.gmail.com")
    port = int(os.getenv("EMAIL_PORT", "587"))
    use_tls = os.getenv("EMAIL_USE_TLS", "true").lower() != "false"
    with smtplib.SMTP(host, port, timeout=10) as smtp:
        if use_tls:
            smtp.starttls()
            smtp.login(user, password)
        smtp.send_message(message)
