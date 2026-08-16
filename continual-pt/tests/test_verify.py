"""
Quick unit tests for the verifier.
Run with: python -m pytest tests/test_verify.py -v
Or just: python tests/test_verify.py
"""
import sys
import os

# Add parent dir to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from continual_pt.verify import run_in_sandbox, parse_verdict, parse_reason, extract_code
from continual_pt import Verdict


def test_parse_verdict():
    assert parse_verdict("CLAIM_VERIFIED\nREASON: works") == Verdict.PASS
    assert parse_verdict("CLAIM_FALSIFIED\nREASON: broken") == Verdict.FAIL
    assert parse_verdict("CLAIM_UNCLEAR\nREASON: cant tell") == Verdict.CONFUSED
    assert parse_verdict("no marker") == Verdict.CONFUSED


def test_parse_reason():
    assert parse_reason("CLAIM_VERIFIED\nREASON: it works") == "it works"
    assert parse_reason("CLAIM_FALSIFIED\nno reason line") == ""


def test_extract_code():
    text = "Here's the code:\n```python\nprint('hello')\n```\nDone."
    assert extract_code(text) == "print('hello')"

    text = "```\nprint('hello')\n```"
    assert extract_code(text) == "print('hello')"

    text = "print('raw code')"
    assert extract_code(text) == "print('raw code')"


def test_run_in_sandbox():
    # Test a simple passing script
    result = run_in_sandbox("print('CLAIM_VERIFIED')\nprint('REASON: test passed')")
    assert "CLAIM_VERIFIED" in result["stdout"]
    assert result["exit_code"] == 0

    # Test a failing script
    result = run_in_sandbox("raise ValueError('test error')")
    assert result["exit_code"] != 0
    assert "ValueError" in result["stderr"]


def test_run_in_sandbox_timeout():
    # Test timeout
    result = run_in_sandbox("import time\ntime.sleep(100)", timeout=2)
    assert result["exit_code"] == -1
    assert "Timeout" in result["stderr"]


if __name__ == "__main__":
    test_parse_verdict()
    print("✓ test_parse_verdict")
    test_parse_reason()
    print("✓ test_parse_reason")
    test_extract_code()
    print("✓ test_extract_code")
    test_run_in_sandbox()
    print("✓ test_run_in_sandbox")
    test_run_in_sandbox_timeout()
    print("✓ test_run_in_sandbox_timeout")
    print("\nAll tests passed!")
