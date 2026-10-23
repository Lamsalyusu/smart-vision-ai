from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import threading
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Union
import asyncio

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AlertStatus(Enum):
    FIRING = "firing"
    RESOLVED = "resolved"
    ACKNOWLEDGED = "acknowledged"
    SUPPRESSED = "suppressed"


class ConditionOperator(Enum):
    GT = ">"
    GTE = ">="
    LT = "<"
    LTE = "<="
    EQ = "=="
    NE = "!="
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    REGEX = "regex"


@dataclass(frozen=True, slots=True)
class AlertRule:
    name: str
    description: str
    metric: str
    operator: ConditionOperator
    threshold: float
    severity: AlertSeverity
    duration: float = 60.0
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)
    runbook_url: str = ""


@dataclass
class Alert:
    id: str
    rule_name: str
    severity: AlertSeverity
    status: AlertStatus
    started_at: datetime
    resolved_at: Optional[datetime] = None
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    value: float = 0.0
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)
    fingerprint: str = ""
    generator_url: str = ""


class AlertManager:
    def __init__(self):
        self.rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self._lock = asyncio.Lock()
        self._evaluation_task: Optional[asyncio.Task] = None
        self._running = False
        self.notification_handlers: List[Callable] = []
        self.silences: Dict[str, Silence] = {}

    def add_rule(self, rule: AlertRule):
        self.rules[rule.name] = rule

    def remove_rule(self, name: str):
        self.rules.pop(name, None)

    def add_notification_handler(self, handler: Callable):
        self.notification_handlers.append(handler)

    async def start(self):
        self._running = True
        self._evaluation_task = asyncio.create_task(self._evaluation_loop())

    async def stop(self):
        self._running = False
        if self._evaluation_task:
            self._evaluation_task.cancel()
            try:
                await self._evaluation_task
            except asyncio.CancelledError:
                pass

    async def _evaluation_loop(self):
        while self._running:
            await asyncio.sleep(10)
            await self._evaluate_rules()

    async def _evaluate_rules(self):
        for rule in self.rules.values():
            try:
                await self._evaluate_rule(rule)
            except Exception as e:
                logger.error(f"Rule evaluation failed for {rule.name}: {e}")

    async def _evaluate_rule(self, rule: AlertRule):
        pass

    def _generate_fingerprint(self, rule_name: str, labels: Dict[str, str]) -> str:
        data = f"{rule_name}:{json.dumps(labels, sort_keys=True)}"
        return hashlib.md5(data.encode()).hexdigest()[:16]

    async def fire_alert(self, rule: AlertRule, value: float, labels: Dict[str, str]):
        fingerprint = self._generate_fingerprint(rule.name, labels)

        async with self._lock:
            existing = self.active_alerts.get(fingerprint)
            if existing and existing.status == AlertStatus.FIRING:
                existing.value = value
                return

        alert = Alert(
            id=str(uuid.uuid4()),
            rule_name=rule.name,
            severity=rule.severity,
            status=AlertStatus.FIRING,
            started_at=datetime.utcnow(),
            value=value,
            labels=labels,
            annotations=rule.annotations,
            fingerprint=fingerprint,
        )

        async with self._lock:
            self.active_alerts[fingerprint] = alert
            self.alert_history.append(alert)

        for handler in self.notification_handlers:
            try:
                await handler(alert)
            except Exception as e:
                logger.error(f"Notification handler failed: {e}")

    async def resolve_alert(self, fingerprint: str):
        async with self._lock:
            alert = self.active_alerts.get(fingerprint)
            if alert and alert.status == AlertStatus.FIRING:
                alert.status = AlertStatus.RESOLVED
                alert.resolved_at = datetime.utcnow()

    async def acknowledge(self, fingerprint: str, user: str):
        async with self._lock:
            alert = self.active_alerts.get(fingerprint)
            if alert:
                alert.status = AlertStatus.ACKNOWLEDGED
                alert.acknowledged_at = datetime.utcnow()
                alert.acknowledged_by = user

    def add_silence(self, silence: Silence):
        self.silences[silence.id] = silence

    def remove_silence(self, silence_id: str):
        self.silences.pop(silence_id, None)

    def is_silenced(self, labels: Dict[str, str]) -> bool:
        for silence in self.silences.values():
            if silence.active and silence.matches(labels):
                return True
        return False


@dataclass(frozen=True, slots=True)
class Silence:
    id: str
    matchers: Dict[str, str]
    starts_at: datetime
    ends_at: datetime
    created_by: str
    comment: str = ""

    def active(self) -> bool:
        now = datetime.utcnow()
        return self.starts_at <= now <= self.ends_at

    def matches(self, labels: Dict[str, str]) -> bool:
        for key, pattern in self.matchers.items():
            if key not in labels:
                return False
            if pattern.endswith("*"):
                if not labels[key].startswith(pattern[:-1]):
                    return False
            elif labels[key] != pattern:
                return False
        return True


class NotificationHandler:
    def __init__(self):
        self.handlers: Dict[str, Callable] = {}

    def register(self, name: str, handler: Callable):
        self.handlers[name] = handler

    async def send(self, alert: Alert, channels: List[str] = None):
        channels = channels or list(self.handlers.keys())
        for channel in channels:
            if channel in self.handlers:
                try:
                    await self.handlers[channel](alert)
                except Exception as e:
                    logger.error(f"Notification handler {channel} failed: {e}")


class EmailNotifier:
    def __init__(self, smtp_host: str, smtp_port: int, username: str, password: str, from_addr: str):
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.from_addr = from_addr

    async def send(self, alert: Alert, to: List[str]):
        import smtplib
        from email.mime.text import MIMEText
        from email.mime.multipart import MIMEMultipart

        msg = MIMEMultipart()
        msg['From'] = self.from_addr
        msg['To'] = ", ".join(to)
        msg['Subject'] = f"[{alert.severity.value.upper()}] {alert.rule_name}"

        body = f"""
        Alert: {alert.rule_name}
        Severity: {alert.severity.value}
        Value: {alert.value}
        Started: {alert.started_at}
        Labels: {json.dumps(alert.labels, indent=2)}
        Annotations: {json.dumps(alert.annotations, indent=2)}
        """
        msg.attach(MIMEText(body, 'plain'))

        try:
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                server.login(self.username, self.password)
                server.sendmail(self.from_addr, to, msg.as_string())
        except Exception as e:
            logger.error(f"Email notification failed: {e}")


class SlackNotifier:
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url

    async def send(self, alert: Alert, channel: str = None):
        import aiohttp

        color = {
            "info": "#36a64f",
            "warning": "#ff9900",
            "critical": "#ff0000",
            "emergency": "#8b0000",
        }.get(alert.severity.value, "#cccccc")

        payload = {
            "attachments": [{
                "color": color,
                "title": f"{alert.severity.value.upper()}: {alert.rule_name}",
                "fields": [
                    {"title": "Value", "value": str(alert.value), "short": True},
                    {"title": "Severity", "value": alert.severity.value, "short": True},
                    {"title": "Started", "value": alert.started_at.isoformat(), "short": True},
                ],
                "footer": "Smart Vision AI Alerting",
                "ts": int(time.time()),
            }
        }
        if channel:
            payload["channel"] = channel

        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(self.webhook_url, json=payload) as resp:
                    if resp.status >= 400:
                        logger.error(f"Slack notification failed: {resp.status}")
            except Exception as e:
                logger.error(f"Slack notification failed: {e}")


class PagerDutyNotifier:
    def __init__(self, integration_key: str):
        self.integration_key = integration_key

    async def send(self, alert: Alert):
        import aiohttp

        payload = {
            "routing_key": self.integration_key,
            "event_action": "trigger" if alert.status == AlertStatus.FIRING else "resolve",
            "dedup_key": alert.fingerprint,
            "payload": {
                "summary": alert.rule_name,
                "severity": alert.severity.value,
                "source": "smart-vision-ai",
                "custom_details": {
                    "value": alert.value,
                    "labels": alert.labels,
                    "annotations": alert.annotations,
                },
            }
        }

        if alert.status == AlertStatus.RESOLVED:
            payload["event_action"] = "resolve"

        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    "https://events.pagerduty.com/v2/enqueue",
                    json=payload,
                ) as resp:
                    if resp.status >= 400:
                        logger.error(f"PagerDuty notification failed: {resp.status}")
            except Exception as e:
                logger.error(f"PagerDuty notification failed: {e}")


class WebhookNotifier:
    def __init__(self, url: str, headers: Dict[str, str] = None):
        self.url = url
        self.headers = headers or {"Content-Type": "application/json"}

    async def send(self, alert: Alert):
        import aiohttp

        payload = {
            "alert": {
                "id": alert.id,
                "rule_name": alert.rule_name,
                "severity": alert.severity.value,
                "status": alert.status.value,
                "value": alert.value,
                "labels": alert.labels,
                "annotations": alert.annotations,
                "started_at": alert.started_at.isoformat(),
                "resolved_at": alert.resolved_at.isoformat() if alert.resolved_at else None,
            }
        }

        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    self.url,
                    json={"alert": alert.__dict__},
                    headers=self.headers,
                ) as resp:
                    if resp.status >= 400:
                        logger.error(f"Webhook notification failed: {resp.status}")
            except Exception as e:
                logger.error(f"Webhook notification failed: {e}")


class Dashboard:
    def __init__(self, name: str):
        self.name = name
        self.panels: List["Panel"] = []

    def add_panel(self, panel: "Panel"):
        self.panels.append(panel)

    def to_json(self) -> str:
        return json.dumps({
            "name": self.name,
            "panels": [p.to_dict() for p in self.panels],
        }, indent=2)


class Panel:
    def __init__(
        self,
        title: str,
        type: str,
        metrics: List[str],
        visualization: str = "graph",
        time_range: str = "1h",
    ):
        self.title = title
        self.type = type
        self.metrics = metrics
        self.visualization = visualization
        self.time_range = time_range

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "type": self.type,
            "metrics": self.metrics,
            "visualization": self.visualization,
            "time_range": self.time_range,
        }


class GrafanaDashboard:
    def __init__(self, title: str, uid: str = None):
        self.title = title
        self.uid = uid or str(uuid.uuid4())[:8]
        self.panels: List[Dict] = []
        self.timezone = "utc"
        self.refresh = "10s"
        self.time = {"from": "now-1h", "to": "now"}

    def add_graph_panel(
        self,
        title: str,
        targets: List[Dict],
        grid_pos: Dict[str, int] = None,
    ):
        panel = {
            "type": "graph",
            "title": title,
            "targets": targets,
            "gridPos": grid_pos or {"x": 0, "y": 0, "w": 12, "h": 8},
        }
        self.panels.append(panel)

    def add_stat_panel(self, title: str, target: Dict, grid_pos: Dict = None):
        panel = {
            "type": "stat",
            "title": title,
            "targets": [target],
            "gridPos": grid_pos or {"x": 0, "y": 0, "w": 6, "h": 4},
        }
        self.panels.append(panel)

    def add_table_panel(self, title: str, target: Dict, grid_pos: Dict = None):
        panel = {
            "type": "table",
            "title": title,
            "targets": [target],
            "gridPos": grid_pos or {"x": 0, "y": 0, "w": 12, "h": 8},
        }
        self.panels.append(panel)

    def to_json(self) -> Dict:
        return {
            "dashboard": {
                "title": self.title,
                "uid": self.uid,
                "timezone": self.timezone,
                "refresh": self.refresh,
                "time": self.time,
                "panels": self.panels,
            }
        }


class PrometheusExporter:
    def __init__(self):
        self.metrics: Dict[str, List[Dict]] = defaultdict(list)

    def add_metric(
        self,
        name: str,
        value: float,
        labels: Dict[str, str] = None,
        metric_type: str = "gauge",
    ):
        self.metrics[name].append({
            "value": value,
            "labels": labels or {},
            "type": metric_type,
            "timestamp": time.time(),
        })

    def generate_metrics(self) -> str:
        lines = []
        for name, samples in self.metrics.items():
            for sample in samples:
                label_str = ""
                if sample["labels"]:
                    label_str = "{" + ",".join(f'{k}="{v}"' for k, v in sample["labels"].items()) + "}"
                lines.append(f"{name}{label_str} {sample['value']}")
        return "\n".join(lines) + "\n"