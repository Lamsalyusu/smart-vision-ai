from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class RuleOperator(str, Enum):
    EQUALS = "eq"
    NOT_EQUALS = "neq"
    GREATER_THAN = "gt"
    LESS_THAN = "lt"
    IN_LIST = "in"
    CONTAINS = "contains"


class RuleLogical(str, Enum):
    AND = "AND"
    OR = "OR"


@dataclass(frozen=True, slots=True)
class RuleCondition:
    field_name: str
    operator: RuleOperator
    expected_value: Any

    def evaluate(self, context: dict[str, Any]) -> bool:
        actual_val = context.get(self.field_name)
        if actual_val is None:
            return False

        if self.operator == RuleOperator.EQUALS:
            return actual_val == self.expected_value
        elif self.operator == RuleOperator.NOT_EQUALS:
            return actual_val != self.expected_value
        elif self.operator == RuleOperator.GREATER_THAN:
            return float(actual_val) > float(self.expected_value)
        elif self.operator == RuleOperator.LESS_THAN:
            return float(actual_val) < float(self.expected_value)
        elif self.operator == RuleOperator.IN_LIST:
            return actual_val in self.expected_value
        elif self.operator == RuleOperator.CONTAINS:
            return self.expected_value in actual_val
        return False


@dataclass(frozen=True, slots=True)
class SecurityPolicyRule:
    rule_id: str
    name: str
    conditions: list[RuleCondition]
    logic: RuleLogical = RuleLogical.AND
    severity: str = "HIGH"
    action_type: str = "TRIGGER_ALARM"

    def is_triggered(self, context: dict[str, Any]) -> bool:
        if not self.conditions:
            return False
        evaluations = [c.evaluate(context) for c in self.conditions]
        if self.logic == RuleLogical.AND:
            return all(evaluations)
        elif self.logic == RuleLogical.OR:
            return any(evaluations)
        return False


class PolicyEngine:
    """Enterprise Policy Evaluation Engine evaluating business security rules in real-time."""

    def __init__(self) -> None:
        self._rules: dict[str, SecurityPolicyRule] = {}

    def register_rule(self, rule: SecurityPolicyRule) -> None:
        self._rules[rule.rule_id] = rule

    def unregister_rule(self, rule_id: str) -> None:
        self._rules.pop(rule_id, None)

    def evaluate(self, context: dict[str, Any]) -> list[dict[str, Any]]:
        """Evaluate context against all active policies and return triggered security actions."""
        triggered = []
        for rule in self._rules.values():
            if rule.is_triggered(context):
                triggered.append({
                    "rule_id": rule.rule_id,
                    "rule_name": rule.name,
                    "severity": rule.severity,
                    "action_type": rule.action_type,
                    "context": context,
                })
        return triggered
