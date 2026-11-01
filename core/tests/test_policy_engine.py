import unittest

from core.domain.analytics.policy_engine import (
    PolicyEngine,
    RuleCondition,
    RuleLogical,
    RuleOperator,
    SecurityPolicyRule,
)


class PolicyEngineTests(unittest.TestCase):
    def setUp(self):
        self.engine = PolicyEngine()

    def test_rule_evaluation_with_and_logic(self):
        rule = SecurityPolicyRule(
            rule_id="rule_restricted_knife",
            name="Restricted Weapon in Vault",
            conditions=[
                RuleCondition(field_name="object_class", operator=RuleOperator.EQUALS, expected_value="knife"),
                RuleCondition(field_name="confidence", operator=RuleOperator.GREATER_THAN, expected_value=0.8),
                RuleCondition(field_name="zone_type", operator=RuleOperator.EQUALS, expected_value="restricted"),
            ],
            logic=RuleLogical.AND,
            severity="CRITICAL",
            action_type="DISPATCH_SECURITY",
        )
        self.engine.register_rule(rule)

        # Context matching all conditions
        matching_context = {
            "object_class": "knife",
            "confidence": 0.95,
            "zone_type": "restricted",
            "track_id": 105,
        }
        triggered = self.engine.evaluate(matching_context)
        self.assertEqual(len(triggered), 1)
        self.assertEqual(triggered[0]["rule_id"], "rule_restricted_knife")
        self.assertEqual(triggered[0]["severity"], "CRITICAL")

        # Context with low confidence -> does not trigger
        non_matching_context = {
            "object_class": "knife",
            "confidence": 0.65,
            "zone_type": "restricted",
            "track_id": 105,
        }
        triggered_none = self.engine.evaluate(non_matching_context)
        self.assertEqual(len(triggered_none), 0)

    def test_rule_evaluation_with_or_logic(self):
        rule = SecurityPolicyRule(
            rule_id="rule_hazard_alert",
            name="Fire or Weapon Detected",
            conditions=[
                RuleCondition(field_name="object_class", operator=RuleOperator.EQUALS, expected_value="fire"),
                RuleCondition(field_name="object_class", operator=RuleOperator.EQUALS, expected_value="knife"),
            ],
            logic=RuleLogical.OR,
            severity="HIGH",
        )
        self.engine.register_rule(rule)

        triggered_fire = self.engine.evaluate({"object_class": "fire"})
        self.assertEqual(len(triggered_fire), 1)

        triggered_knife = self.engine.evaluate({"object_class": "knife"})
        self.assertEqual(len(triggered_knife), 1)

        triggered_bottle = self.engine.evaluate({"object_class": "water_bottle"})
        self.assertEqual(len(triggered_bottle), 0)


if __name__ == "__main__":
    unittest.main()
