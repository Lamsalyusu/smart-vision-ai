import unittest

from core.domain.behavior_tree.nodes import (
    ActionNode,
    BehaviorNode,
    Blackboard,
    ConditionNode,
    Inverter,
    NodeStatus,
    Selector,
    Sequence,
)


class NodeStatusTests(unittest.TestCase):
    def test_values(self):
        self.assertEqual(NodeStatus.SUCCESS.value, "SUCCESS")
        self.assertEqual(NodeStatus.FAILURE.value, "FAILURE")
        self.assertEqual(NodeStatus.RUNNING.value, "RUNNING")


class BlackboardTests(unittest.TestCase):
    def test_get_set_has(self):
        bb = Blackboard()
        self.assertFalse(bb.has("key"))
        self.assertEqual(bb.get("key", "default"), "default")
        bb.set("key", 42)
        self.assertTrue(bb.has("key"))
        self.assertEqual(bb.get("key"), 42)
        self.assertEqual(bb.get("missing", "default"), "default")


class ActionNodeTests(unittest.TestCase):
    def test_tick_runs_action(self):
        called = []

        def action(bb):
            called.append(True)
            return NodeStatus.SUCCESS

        node = ActionNode("act", action)
        bb = Blackboard()
        status = node.tick(bb)
        self.assertEqual(status, NodeStatus.SUCCESS)
        self.assertEqual(len(called), 1)
        self.assertEqual(node.status, NodeStatus.SUCCESS)


class ConditionNodeTests(unittest.TestCase):
    def test_tick_true(self):
        node = ConditionNode("cond", lambda bb: True)
        bb = Blackboard()
        status = node.tick(bb)
        self.assertEqual(status, NodeStatus.SUCCESS)

    def test_tick_false(self):
        node = ConditionNode("cond", lambda bb: False)
        bb = Blackboard()
        status = node.tick(bb)
        self.assertEqual(status, NodeStatus.FAILURE)


class SequenceTests(unittest.TestCase):
    def test_all_succeed(self):
        bb = Blackboard()
        n1 = ActionNode("a", lambda bb: NodeStatus.SUCCESS)
        n2 = ActionNode("b", lambda bb: NodeStatus.SUCCESS)
        seq = Sequence("seq", [n1, n2])
        status = seq.tick(bb)
        self.assertEqual(status, NodeStatus.SUCCESS)

    def test_stops_on_failure(self):
        call_count = [0]

        def fail(bb):
            call_count[0] += 1
            return NodeStatus.FAILURE

        def never(bb):
            return NodeStatus.SUCCESS

        bb = Blackboard()
        n1 = ActionNode("a", fail)
        n2 = ActionNode("b", never)
        seq = Sequence("seq", [n1, n2])
        status = seq.tick(bb)
        self.assertEqual(status, NodeStatus.FAILURE)
        self.assertEqual(call_count[0], 1)  # n2 should not run


class SelectorTests(unittest.TestCase):
    def test_first_success(self):
        call_count = [0]

        def fail(bb):
            call_count[0] += 1
            return NodeStatus.FAILURE

        def succeed(bb):
            return NodeStatus.SUCCESS

        bb = Blackboard()
        n1 = ActionNode("a", fail)
        n2 = ActionNode("b", succeed)
        sel = Selector("sel", [n1, n2])
        status = sel.tick(bb)
        self.assertEqual(status, NodeStatus.SUCCESS)
        self.assertEqual(call_count[0], 1)

    def test_all_fail(self):
        bb = Blackboard()
        n1 = ActionNode("a", lambda bb: NodeStatus.FAILURE)
        n2 = ActionNode("b", lambda bb: NodeStatus.FAILURE)
        sel = Selector("sel", [n1, n2])
        status = sel.tick(bb)
        self.assertEqual(status, NodeStatus.FAILURE)


class InverterTests(unittest.TestCase):
    def test_inverts_success(self):
        bb = Blackboard()
        child = ActionNode("child", lambda bb: NodeStatus.SUCCESS)
        inv = Inverter("inv", child)
        status = inv.tick(bb)
        self.assertEqual(status, NodeStatus.FAILURE)

    def test_inverts_failure(self):
        bb = Blackboard()
        child = ActionNode("child", lambda bb: NodeStatus.FAILURE)
        inv = Inverter("inv", child)
        status = inv.tick(bb)
        self.assertEqual(status, NodeStatus.SUCCESS)

    def test_passes_running(self):
        bb = Blackboard()
        child = ActionNode("child", lambda bb: NodeStatus.RUNNING)
        inv = Inverter("inv", child)
        status = inv.tick(bb)
        self.assertEqual(status, NodeStatus.RUNNING)


class BehaviorNodeBaseTests(unittest.TestCase):
    def test_default_status(self):
        node = BehaviorNode("base")
        self.assertEqual(node.status, NodeStatus.FAILURE)
        self.assertEqual(node.name, "base")

    def test_tick_raises_not_implemented(self):
        node = BehaviorNode("base")
        bb = Blackboard()
        with self.assertRaises(NotImplementedError):
            node.tick(bb)


if __name__ == "__main__":
    unittest.main()
