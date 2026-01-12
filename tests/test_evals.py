"""PR Agent Evaluation.

Automated benchmark (evals) for the agent.
"""

import unittest
from pr_agent.llm import LLMClient

class TestPRAgent(unittest.TestCase):
    def setUp(self):
        self.client = LLMClient(provider="openai", temperature=0)

    def test_syntax_error_detection(self):
        """Ensure agent catches basic syntax errors."""
        bad_code = "def foo() return 5" # Missing colon
        
        # Test just the review logic
        result = self.client.review_code_chunk(bad_code, "python")
        self.assertIn("syntax", result.lower())

    def test_security_leak_detection(self):
        """Ensure agent catches hardcoded secrets."""
        bad_code = "api_key = 'sk-12345'"
        
        result = self.client.review_code_chunk(bad_code, "python")
        self.assertIn("secret", result.lower())

if __name__ == "__main__":
    unittest.main()
