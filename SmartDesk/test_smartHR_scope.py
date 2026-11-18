"""Tests for SmartHR request-scope and ticket-offer behavior."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import smartHR


class SmartHRScopeTests(unittest.TestCase):
    def test_out_of_scope_question_skips_agent_and_ticket_offer(self):
        with (
            patch.object(smartHR, "classify_question_scope", return_value="out_of_scope"),
            patch.object(smartHR, "initialize_agent") as initialize_agent,
        ):
            result = smartHR.run_agent("What is the capital of India?")

        initialize_agent.assert_not_called()
        self.assertEqual(result["scope"], "out_of_scope")
        self.assertFalse(result["needs_ticket"])
        self.assertEqual(result["tools_used"], [])
        self.assertTrue(result["response"].startswith("[NO_ANSWER]"))

    def test_unresolved_in_scope_question_can_offer_ticket(self):
        class NoAnswerModel:
            def invoke(self, _messages):
                return SimpleNamespace(
                    content="[NO_ANSWER] I don't have enough information.",
                    tool_calls=[],
                )

        with (
            patch.object(smartHR, "classify_question_scope", return_value="it_support"),
            patch.object(smartHR, "llm_with_tools", NoAnswerModel()),
        ):
            result = smartHR.run_agent("My company VPN is not connecting.")

        self.assertEqual(result["scope"], "it_support")
        self.assertTrue(result["needs_ticket"])

    def test_unclear_question_does_not_offer_ticket(self):
        with patch.object(smartHR, "classify_question_scope", return_value="unclear"):
            result = smartHR.run_agent("Can you help me with something?")

        self.assertEqual(result["scope"], "unclear")
        self.assertFalse(result["needs_ticket"])
        self.assertEqual(result["tools_used"], [])


if __name__ == "__main__":
    unittest.main()
