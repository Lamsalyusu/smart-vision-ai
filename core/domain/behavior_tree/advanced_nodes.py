from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Sequence, Callable, Dict, Any, List, Tuple, Set
from enum import Enum
from abc import ABC, abstractmethod
import math
import time
from collections import defaultdict


class NodeStatus(Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    RUNNING = "running"
    ERROR = "error"
    SKIPPED = "skipped"


class NodeType(Enum):
    SEQUENCE = "sequence"
    SELECTOR = "selector"
    PARALLEL = "parallel"
    DECORATOR = "decorator"
    ACTION = "action"
    CONDITION = "condition"
    SUBTREE = "subtree"


class ParallelPolicy(Enum):
    SUCCESS_ON_ONE = "success_on_one"
    SUCCESS_ON_ALL = "success_on_all"
    FAILURE_ON_ONE = "failure_on_one"
    FAILURE_ON_ALL = "failure_on_all"


@dataclass(frozen=True, slots=True)
class BlackboardEntry:
    key: str
    value: Any
    timestamp: float = field(default_factory=time.time)
    ttl: Optional[float] = None


@dataclass(frozen=True, slots=True)
class NodeExecution:
    node_id: str
    status: NodeStatus
    start_time: float
    end_time: Optional[float] = None
    result: Any = None
    error: Optional[str] = None


class BehaviorNode(ABC):
    def __init__(
        self,
        node_id: str,
        name: str,
        node_type: NodeType,
        blackboard: Optional[Dict[str, Any]] = None
    ):
        self.node_id = node_id
        self.name = name
        self.node_type = node_type
        self.blackboard = blackboard or {}
        self.children: List[BehaviorNode] = []
        self.parent: Optional[BehaviorNode] = None
        self.preconditions: List[Callable[[], bool]] = []
        self.postconditions: List[Callable[[], bool]] = []
        self.decorators: List['DecoratorNode'] = []

    def add_child(self, child: 'BehaviorNode'):
        child.parent = self
        self.children.append(child)

    def add_precondition(self, condition: Callable[[], bool]):
        self.preconditions.append(condition)

    def add_postcondition(self, condition: Callable[[], bool]):
        self.postconditions.append(condition)

    def add_decorator(self, decorator: 'DecoratorNode'):
        self.decorators.append(decorator)

    def check_preconditions(self) -> bool:
        return all(cond() for cond in self.preconditions)

    def check_postconditions(self) -> bool:
        return all(cond() for cond in self.postconditions)

    def set_blackboard(self, key: str, value: Any, ttl: Optional[float] = None):
        self.blackboard[key] = BlackboardEntry(key, value, ttl=ttl)

    def get_blackboard(self, key: str, default: Any = None) -> Any:
        entry = self.blackboard.get(key)
        if entry is None:
            return default
        if entry.ttl and time.time() - entry.timestamp > entry.ttl:
            del self.blackboard[key]
            return default
        return entry.value

    @abstractmethod
    def tick(self, dt: float) -> NodeStatus:
        pass

    def reset(self):
        for child in self.children:
            child.reset()


class CompositeNode(BehaviorNode):
    def __init__(
        self,
        node_id: str,
        name: str,
        node_type: NodeType,
        blackboard: Optional[Dict[str, Any]] = None
    ):
        super().__init__(node_id, name, node_type, blackboard)
        self.current_child = 0

    def reset(self):
        super().reset()
        self.current_child = 0


class SequenceNode(CompositeNode):
    def __init__(
        self,
        node_id: str,
        name: str,
        blackboard: Optional[Dict[str, Any]] = None,
        stop_on_failure: bool = True
    ):
        super().__init__(node_id, name, NodeType.SEQUENCE, blackboard)
        self.stop_on_failure = stop_on_failure

    def tick(self, dt: float) -> NodeStatus:
        if not self.check_preconditions():
            return NodeStatus.FAILURE

        while self.current_child < len(self.children):
            child = self.children[self.current_child]
            status = child.tick(dt)

            if status == NodeStatus.RUNNING:
                return NodeStatus.RUNNING
            elif status == NodeStatus.FAILURE:
                if self.stop_on_failure:
                    self.current_child = 0
                    return NodeStatus.FAILURE
            elif status == NodeStatus.SUCCESS:
                pass

            self.current_child += 1

        if self.current_child >= len(self.children):
            self.current_child = 0
            if self.check_postconditions():
                return NodeStatus.SUCCESS
            return NodeStatus.FAILURE

        return NodeStatus.RUNNING


class SelectorNode(CompositeNode):
    def __init__(
        self,
        node_id: str,
        name: str,
        blackboard: Optional[Dict[str, Any]] = None,
        stop_on_success: bool = True
    ):
        super().__init__(node_id, name, NodeType.SELECTOR, blackboard)
        self.stop_on_success = stop_on_success

    def tick(self, dt: float) -> NodeStatus:
        if not self.check_preconditions():
            return NodeStatus.FAILURE

        while self.current_child < len(self.children):
            child = self.children[self.current_child]
            status = child.tick(dt)

            if status == NodeStatus.RUNNING:
                return NodeStatus.RUNNING
            elif status == NodeStatus.SUCCESS:
                if self.stop_on_success:
                    self.current_child = 0
                    if self.check_postconditions():
                        return NodeStatus.SUCCESS
                    return NodeStatus.FAILURE
            elif status == NodeStatus.FAILURE:
                pass

            self.current_child += 1

        if self.current_child >= len(self.children):
            self.current_child = 0
            return NodeStatus.FAILURE

        return NodeStatus.RUNNING


class ParallelNode(CompositeNode):
    def __init__(
        self,
        node_id: str,
        name: str,
        blackboard: Optional[Dict[str, Any]] = None,
        policy: ParallelPolicy = ParallelPolicy.SUCCESS_ON_ALL
    ):
        super().__init__(node_id, name, NodeType.PARALLEL, blackboard)
        self.policy = policy
        self.child_statuses: Dict[str, NodeStatus] = {}

    def tick(self, dt: float) -> NodeStatus:
        if not self.check_preconditions():
            return NodeStatus.FAILURE

        running_count = 0
        success_count = 0
        failure_count = 0

        for child in self.children:
            status = child.tick(dt)
            self.child_statuses[child.node_id] = status

            if status == NodeStatus.RUNNING:
                running_count += 1
            elif status == NodeStatus.SUCCESS:
                success_count += 1
            elif status == NodeStatus.FAILURE:
                failure_count += 1

        if self.policy == ParallelPolicy.SUCCESS_ON_ONE:
            if success_count > 0:
                return NodeStatus.SUCCESS
            elif failure_count == len(self.children):
                return NodeStatus.FAILURE
        elif self.policy == ParallelPolicy.SUCCESS_ON_ALL:
            if failure_count > 0:
                return NodeStatus.FAILURE
            elif success_count == len(self.children):
                return NodeStatus.SUCCESS
        elif self.policy == ParallelPolicy.FAILURE_ON_ONE:
            if failure_count > 0:
                return NodeStatus.FAILURE
            elif success_count == len(self.children):
                return NodeStatus.SUCCESS
        elif self.policy == ParallelPolicy.FAILURE_ON_ALL:
            if success_count > 0:
                return NodeStatus.SUCCESS
            elif failure_count == len(self.children):
                return NodeStatus.FAILURE

        return NodeStatus.RUNNING


class DecoratorNode(BehaviorNode):
    def __init__(
        self,
        node_id: str,
        name: str,
        child: Optional[BehaviorNode] = None,
        blackboard: Optional[Dict[str, Any]] = None
    ):
        super().__init__(node_id, name, NodeType.DECORATOR, blackboard)
        self.child = child
        if child:
            child.parent = self

    def tick(self, dt: float) -> NodeStatus:
        if self.child is None:
            return NodeStatus.ERROR
        return self._decorate(self.child.tick(dt))

    @abstractmethod
    def _decorate(self, child_status: NodeStatus) -> NodeStatus:
        pass


class InverterDecorator(DecoratorNode):
    def _decorate(self, child_status: NodeStatus) -> NodeStatus:
        if child_status == NodeStatus.SUCCESS:
            return NodeStatus.FAILURE
        elif child_status == NodeStatus.FAILURE:
            return NodeStatus.SUCCESS
        return child_status


class RepeaterDecorator(DecoratorNode):
    def __init__(
        self,
        node_id: str,
        name: str,
        child: Optional[BehaviorNode] = None,
        blackboard: Optional[Dict[str, Any]] = None,
        count: int = -1,
        until_success: bool = False
    ):
        super().__init__(node_id, name, child, blackboard)
        self.count = count
        self.until_success = until_success
        self.current_count = 0

    def _decorate(self, child_status: NodeStatus) -> NodeStatus:
        if self.until_success:
            if child_status == NodeStatus.SUCCESS:
                return NodeStatus.SUCCESS
            elif child_status == NodeStatus.FAILURE:
                return NodeStatus.RUNNING
            return NodeStatus.RUNNING
        else:
            self.current_count += 1
            if self.count > 0 and self.current_count >= self.count:
                return NodeStatus.SUCCESS
            return child_status

    def reset(self):
        super().reset()
        self.current_count = 0


class TimeoutDecorator(DecoratorNode):
    def __init__(
        self,
        node_id: str,
        name: str,
        child: Optional[BehaviorNode] = None,
        blackboard: Optional[Dict[str, Any]] = None,
        timeout: float = 10.0
    ):
        super().__init__(node_id, name, child, blackboard)
        self.timeout = timeout
        self.start_time = 0.0

    def _decorate(self, child_status: NodeStatus) -> NodeStatus:
        if self.start_time == 0.0:
            self.start_time = time.time()

        if time.time() - self.start_time > self.timeout:
            return NodeStatus.FAILURE

        return child_status

    def reset(self):
        super().reset()
        self.start_time = 0.0


class RetryDecorator(DecoratorNode):
    def __init__(
        self,
        node_id: str,
        name: str,
        child: Optional[BehaviorNode] = None,
        blackboard: Optional[Dict[str, Any]] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        super().__init__(node_id, name, child, blackboard)
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.retry_count = 0
        self.last_retry_time = 0.0

    def _decorate(self, child_status: NodeStatus) -> NodeStatus:
        if child_status == NodeStatus.SUCCESS:
            self.retry_count = 0
            return NodeStatus.SUCCESS

        if child_status == NodeStatus.FAILURE:
            if self.retry_count < self.max_retries:
                if time.time() - self.last_retry_time >= self.retry_delay:
                    self.retry_count += 1
                    self.last_retry_time = time.time()
                    return NodeStatus.RUNNING
            return NodeStatus.FAILURE

        return child_status

    def reset(self):
        super().reset()
        self.retry_count = 0
        self.last_retry_time = 0.0


class CooldownDecorator(DecoratorNode):
    def __init__(
        self,
        node_id: str,
        name: str,
        child: Optional[BehaviorNode] = None,
        blackboard: Optional[Dict[str, Any]] = None,
        cooldown: float = 1.0
    ):
        super().__init__(node_id, name, child, blackboard)
        self.cooldown = cooldown
        self.last_execution = 0.0

    def _decorate(self, child_status: NodeStatus) -> NodeStatus:
        if time.time() - self.last_execution < self.cooldown:
            return NodeStatus.FAILURE

        if child_status == NodeStatus.SUCCESS:
            self.last_execution = time.time()
            return NodeStatus.SUCCESS

        return child_status


class ConditionNode(BehaviorNode):
    def __init__(
        self,
        node_id: str,
        name: str,
        condition: Callable[[], bool],
        blackboard: Optional[Dict[str, Any]] = None
    ):
        super().__init__(node_id, name, NodeType.CONDITION, blackboard)
        self.condition = condition

    def tick(self, dt: float) -> NodeStatus:
        if not self.check_preconditions():
            return NodeStatus.FAILURE

        if self.condition():
            if self.check_postconditions():
                return NodeStatus.SUCCESS
        return NodeStatus.FAILURE


class ActionNode(BehaviorNode):
    def __init__(
        self,
        node_id: str,
        name: str,
        action: Callable[[float], NodeStatus],
        blackboard: Optional[Dict[str, Any]] = None
    ):
        super().__init__(node_id, name, NodeType.ACTION, blackboard)
        self.action = action
        self.status = NodeStatus.RUNNING

    def tick(self, dt: float) -> NodeStatus:
        if not self.check_preconditions():
            return NodeStatus.FAILURE

        self.status = self.action(dt)

        if self.status == NodeStatus.SUCCESS and self.check_postconditions():
            return NodeStatus.SUCCESS

        return self.status


class SubtreeNode(BehaviorNode):
    def __init__(
        self,
        node_id: str,
        name: str,
        subtree_root: BehaviorNode,
        blackboard: Optional[Dict[str, Any]] = None
    ):
        super().__init__(node_id, name, NodeType.SUBTREE, blackboard)
        self.subtree_root = subtree_root

    def tick(self, dt: float) -> NodeStatus:
        if not self.check_preconditions():
            return NodeStatus.FAILURE

        return self.subtree_root.tick(dt)

    def reset(self):
        self.subtree_root.reset()


class BehaviorTree:
    def __init__(self, root: BehaviorNode, blackboard: Optional[Dict[str, Any]] = None):
        self.root = root
        self.blackboard = blackboard or {}
        self.execution_history: List[NodeExecution] = []
        self.tick_count = 0
        self.running = False

    def tick(self, dt: float = 0.016) -> NodeStatus:
        self.tick_count += 1
        start_time = time.time()
        status = self.root.tick(dt)
        end_time = time.time()

        self.execution_history.append(NodeExecution(
            node_id=self.root.node_id,
            status=status,
            start_time=start_time,
            end_time=end_time
        ))

        return status

    def run(self, max_ticks: int = 1000, dt: float = 0.016) -> NodeStatus:
        self.running = True
        for _ in range(max_ticks):
            if not self.running:
                break
            status = self.tick(dt)
            if status in [NodeStatus.SUCCESS, NodeStatus.FAILURE, NodeStatus.ERROR]:
                self.running = False
                return status
        return NodeStatus.RUNNING

    def stop(self):
        self.running = False

    def reset(self):
        self.root.reset()
        self.execution_history.clear()
        self.tick_count = 0


class BehaviorTreeBuilder:
    def __init__(self):
        self.nodes: Dict[str, BehaviorNode] = {}
        self.root: Optional[BehaviorNode] = None
        self.blackboard: Dict[str, Any] = {}

    def sequence(self, node_id: str, name: str, **kwargs) -> 'BehaviorTreeBuilder':
        node = SequenceNode(node_id, name, self.blackboard, **kwargs)
        self.nodes[node_id] = node
        if self.root is None:
            self.root = node
        return self

    def selector(self, node_id: str, name: str, **kwargs) -> 'BehaviorTreeBuilder':
        node = SelectorNode(node_id, name, self.blackboard, **kwargs)
        self.nodes[node_id] = node
        if self.root is None:
            self.root = node
        return self

    def parallel(self, node_id: str, name: str, policy: ParallelPolicy = ParallelPolicy.SUCCESS_ON_ALL, **kwargs) -> 'BehaviorTreeBuilder':
        node = ParallelNode(node_id, name, self.blackboard, policy, **kwargs)
        self.nodes[node_id] = node
        if self.root is None:
            self.root = node
        return self

    def action(self, node_id: str, name: str, action: Callable[[float], NodeStatus], **kwargs) -> 'BehaviorTreeBuilder':
        node = ActionNode(node_id, name, action, self.blackboard)
        self.nodes[node_id] = node
        if self.root is None:
            self.root = node
        return self

    def condition(self, node_id: str, name: str, condition: Callable[[], bool], **kwargs) -> 'BehaviorTreeBuilder':
        node = ConditionNode(node_id, name, condition, self.blackboard)
        self.nodes[node_id] = node
        if self.root is None:
            self.root = node
        return self

    def inverter(self, node_id: str, name: str, child_id: str, **kwargs) -> 'BehaviorTreeBuilder':
        child = self.nodes.get(child_id)
        node = InverterDecorator(node_id, name, child, self.blackboard, **kwargs)
        self.nodes[node_id] = node
        if self.root is None:
            self.root = node
        return self

    def repeater(self, node_id: str, name: str, child_id: str, count: int = -1, until_success: bool = False, **kwargs) -> 'BehaviorTreeBuilder':
        child = self.nodes.get(child_id)
        node = RepeaterDecorator(node_id, name, child, self.blackboard, count, until_success, **kwargs)
        self.nodes[node_id] = node
        if self.root is None:
            self.root = node
        return self

    def timeout(self, node_id: str, name: str, child_id: str, timeout: float = 10.0, **kwargs) -> 'BehaviorTreeBuilder':
        child = self.nodes.get(child_id)
        node = TimeoutDecorator(node_id, name, child, self.blackboard, timeout, **kwargs)
        self.nodes[node_id] = node
        if self.root is None:
            self.root = node
        return self

    def retry(self, node_id: str, name: str, child_id: str, max_retries: int = 3, retry_delay: float = 1.0, **kwargs) -> 'BehaviorTreeBuilder':
        child = self.nodes.get(child_id)
        node = RetryDecorator(node_id, name, child, self.blackboard, max_retries, retry_delay, **kwargs)
        self.nodes[node_id] = node
        if self.root is None:
            self.root = node
        return self

    def cooldown(self, node_id: str, name: str, child_id: str, cooldown: float = 1.0, **kwargs) -> 'BehaviorTreeBuilder':
        child = self.nodes.get(child_id)
        node = CooldownDecorator(node_id, name, child, self.blackboard, cooldown, **kwargs)
        self.nodes[node_id] = node
        if self.root is None:
            self.root = node
        return self

    def child(self, parent_id: str, child_id: str) -> 'BehaviorTreeBuilder':
        parent = self.nodes.get(parent_id)
        child = self.nodes.get(child_id)
        if parent and child:
            parent.add_child(child)
        return self

    def precondition(self, node_id: str, condition: Callable[[], bool]) -> 'BehaviorTreeBuilder':
        node = self.nodes.get(node_id)
        if node:
            node.add_precondition(condition)
        return self

    def postcondition(self, node_id: str, condition: Callable[[], bool]) -> 'BehaviorTreeBuilder':
        node = self.nodes.get(node_id)
        if node:
            node.add_postcondition(condition)
        return self

    def build(self) -> BehaviorTree:
        return BehaviorTree(self.root, self.blackboard)


class BehaviorTreeVisualizer:
    @staticmethod
    def to_dot(tree: BehaviorTree) -> str:
        lines = ["digraph BehaviorTree {"]
        lines.append('  node [shape=box, style="filled,rounded"];')
        lines.append('  edge [fontsize=10];')

        def add_node(node: BehaviorNode, depth: int = 0):
            color = {
                NodeType.SEQUENCE: "#4CAF50",
                NodeType.SELECTOR: "#2196F3",
                NodeType.PARALLEL: "#FF9800",
                NodeType.DECORATOR: "#9C27B0",
                NodeType.ACTION: "#F44336",
                NodeType.CONDITION: "#00BCD4",
                NodeType.SUBTREE: "#795548"
            }.get(node.node_type, "#9E9E9E")

            label = f"{node.name}\\n({node.node_id})"
            lines.append(f'  "{node.node_id}" [label="{label}", fillcolor="{color}"];')

            for child in node.children:
                lines.append(f'  "{node.node_id}" -> "{child.node_id}";')
                add_node(child, depth + 1)

        add_node(tree.root)
        lines.append("}")
        return "\n".join(lines)

    @staticmethod
    def draw_execution_history(tree: BehaviorTree, image_size: Tuple[int, int] = (1200, 800)) -> np.ndarray:
        import cv2
        vis = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)

        if not tree.execution_history:
            return vis

        bar_height = 30
        y_start = 50
        max_ticks = min(len(tree.execution_history), (image_size[1] - 100) // bar_height)

        for i, exec_record in enumerate(tree.execution_history[-max_ticks:]):
            y = y_start + i * bar_height
            color = {
                NodeStatus.SUCCESS: (0, 255, 0),
                NodeStatus.FAILURE: (0, 0, 255),
                NodeStatus.RUNNING: (255, 255, 0),
                NodeStatus.ERROR: (255, 0, 255),
                NodeStatus.SKIPPED: (128, 128, 128)
            }.get(exec_record.status, (255, 255, 255))

            cv2.rectangle(vis, (50, y), (50 + int((exec_record.end_time - exec_record.start_time) * 1000), y + bar_height - 2), color, -1)
            cv2.putText(vis, f"{exec_record.node_id}: {exec_record.status.value}", (60, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return vis


class BehaviorTreeSerializer:
    @staticmethod
    def to_dict(tree: BehaviorTree) -> Dict[str, Any]:
        def serialize_node(node: BehaviorNode) -> Dict[str, Any]:
            data = {
                'node_id': node.node_id,
                'name': node.name,
                'type': node.node_type.value,
                'blackboard': {k: v.value for k, v in node.blackboard.items()},
                'children': [serialize_node(c) for c in node.children],
                'decorators': []
            }

            for dec in node.decorators:
                data['decorators'].append({
                    'node_id': dec.node_id,
                    'name': dec.name,
                    'type': dec.__class__.__name__,
                    'params': {k: v for k, v in dec.__dict__.items() if not k.startswith('_') and k not in ['node_id', 'name', 'node_type', 'blackboard', 'children', 'parent', 'preconditions', 'postconditions', 'decorators', 'child']}
                })

            return data

        return {
            'root': serialize_node(tree.root),
            'blackboard': {k: v.value for k, v in tree.blackboard.items()},
            'tick_count': tree.tick_count
        }

    @staticmethod
    def from_dict(data: Dict[str, Any], action_registry: Dict[str, Callable], condition_registry: Dict[str, Callable]) -> BehaviorTree:
        def deserialize_node(node_data: Dict[str, Any]) -> BehaviorNode:
            node_type = NodeType(node_data['type'])

            if node_type == NodeType.SEQUENCE:
                node = SequenceNode(node_data['node_id'], node_data['name'])
            elif node_type == NodeType.SELECTOR:
                node = SelectorNode(node_data['node_id'], node_data['name'])
            elif node_type == NodeType.PARALLEL:
                node = ParallelNode(node_data['node_id'], node_data['name'])
            elif node_type == NodeType.ACTION:
                action_name = node_data.get('action')
                action = action_registry.get(action_name) if action_name else lambda dt: NodeStatus.SUCCESS
                node = ActionNode(node_data['node_id'], node_data['name'], action)
            elif node_type == NodeType.CONDITION:
                cond_name = node_data.get('condition')
                condition = condition_registry.get(cond_name) if cond_name else lambda: True
                node = ConditionNode(node_data['node_id'], node_data['name'], condition)
            elif node_type == NodeType.SUBTREE:
                node = SubtreeNode(node_data['node_id'], node_data['name'], None)
            else:
                node = BehaviorNode(node_data['node_id'], node_data['name'], node_type)

            node.blackboard = {k: BlackboardEntry(k, v) for k, v in node_data.get('blackboard', {}).items()}

            for child_data in node_data.get('children', []):
                child = deserialize_node(child_data)
                node.add_child(child)

            for dec_data in node_data.get('decorators', []):
                dec_type = dec_data['type']
                if dec_type == 'InverterDecorator':
                    dec = InverterDecorator(dec_data['node_id'], dec_data['name'], node)
                elif dec_type == 'RepeaterDecorator':
                    dec = RepeaterDecorator(dec_data['node_id'], dec_data['name'], node, **dec_data.get('params', {}))
                elif dec_type == 'TimeoutDecorator':
                    dec = TimeoutDecorator(dec_data['node_id'], dec_data['name'], node, **dec_data.get('params', {}))
                elif dec_type == 'RetryDecorator':
                    dec = RetryDecorator(dec_data['node_id'], dec_data['name'], node, **dec_data.get('params', {}))
                elif dec_type == 'CooldownDecorator':
                    dec = CooldownDecorator(dec_data['node_id'], dec_data['name'], node, **dec_data.get('params', {}))
                else:
                    continue
                node.add_decorator(dec)

            return node

        root = deserialize_node(data['root'])
        blackboard = {k: BlackboardEntry(k, v) for k, v in data.get('blackboard', {}).items()}
        tree = BehaviorTree(root, blackboard)
        tree.tick_count = data.get('tick_count', 0)
        return tree


from core.domain.vision.feature_extraction import FeatureType, Keypoint, FeatureMatch, FeatureSet
