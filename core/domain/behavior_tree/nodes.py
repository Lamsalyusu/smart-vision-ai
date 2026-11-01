from __future__ import annotations

from collections.abc import Callable
from enum import Enum
from typing import Any


class NodeStatus(Enum):
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    RUNNING = "RUNNING"


class Blackboard:
    """Shared thread-safe key-value memory for Behavior Tree nodes."""

    def __init__(self) -> None:
        self._data: dict[str, Any] = {}

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self._data[key] = value

    def has(self, key: str) -> bool:
        return key in self._data


class BehaviorNode:
    """Base Behavior Tree execution node."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.status = NodeStatus.FAILURE

    def tick(self, bb: Blackboard) -> NodeStatus:
        raise NotImplementedError


class ActionNode(BehaviorNode):
    """Executes a leaf action function."""

    def __init__(self, name: str, action_fn: Callable[[Blackboard], NodeStatus]) -> None:
        super().__init__(name)
        self.action_fn = action_fn

    def tick(self, bb: Blackboard) -> NodeStatus:
        self.status = self.action_fn(bb)
        return self.status


class ConditionNode(BehaviorNode):
    """Evaluates a boolean condition predicate."""

    def __init__(self, name: str, predicate: Callable[[Blackboard], bool]) -> None:
        super().__init__(name)
        self.predicate = predicate

    def tick(self, bb: Blackboard) -> NodeStatus:
        res = self.predicate(bb)
        self.status = NodeStatus.SUCCESS if res else NodeStatus.FAILURE
        return self.status


class Sequence(BehaviorNode):
    """Composite node: ticks children in order. Returns SUCCESS only if all children succeed."""

    def __init__(self, name: str, children: list[BehaviorNode]) -> None:
        super().__init__(name)
        self.children = children

    def tick(self, bb: Blackboard) -> NodeStatus:
        for child in self.children:
            status = child.tick(bb)
            if status != NodeStatus.SUCCESS:
                self.status = status
                return status
        self.status = NodeStatus.SUCCESS
        return NodeStatus.SUCCESS


class Selector(BehaviorNode):
    """Fallback composite node: ticks children in order. Returns SUCCESS on first child success."""

    def __init__(self, name: str, children: list[BehaviorNode]) -> None:
        super().__init__(name)
        self.children = children

    def tick(self, bb: Blackboard) -> NodeStatus:
        for child in self.children:
            status = child.tick(bb)
            if status != NodeStatus.FAILURE:
                self.status = status
                return status
        self.status = NodeStatus.FAILURE
        return NodeStatus.FAILURE


class Inverter(BehaviorNode):
    """Decorator node: inverts SUCCESS to FAILURE and vice-versa."""

    def __init__(self, name: str, child: BehaviorNode) -> None:
        super().__init__(name)
        self.child = child

    def tick(self, bb: Blackboard) -> NodeStatus:
        status = self.child.tick(bb)
        if status == NodeStatus.SUCCESS:
            self.status = NodeStatus.FAILURE
        elif status == NodeStatus.FAILURE:
            self.status = NodeStatus.SUCCESS
        else:
            self.status = NodeStatus.RUNNING
        return self.status
