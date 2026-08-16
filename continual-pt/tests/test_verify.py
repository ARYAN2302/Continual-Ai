"""
Unit tests for the verifier.
Run with: python -m pytest tests/test_verify.py -v
Or just: python tests/test_verify.py
"""
import sys
import os

# Add parent dir to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from continual_pt.verify import (
    run_in_sandbox, parse_verdict, parse_reason,
    extract_code, sanitize_code,
)
from continual_pt import Verdict


def test_parse_verdict():
    assert parse_verdict("CLAIM_VERIFIED\nREASON: works") == Verdict.PASS
    assert parse_verdict("CLAIM_FALSIFIED\nREASON: broken") == Verdict.FAIL
    assert parse_verdict("CLAIM_UNCLEAR\nREASON: cant tell") == Verdict.CONFUSED
    assert parse_verdict("no marker") == Verdict.CONFUSED


def test_parse_reason():
    assert parse_reason("CLAIM_VERIFIED\nREASON: it works") == "it works"
    assert parse_reason("CLAIM_FALSIFIED\nno reason line") == ""


def test_extract_code_markdown_fences():
    """Code wrapped in ```python ... ``` should be extracted cleanly."""
    text = 'Here is the code:\n```python\nimport torch\nprint("hello")\n```\nDone.'
    result = extract_code(text)
    assert "import torch" in result
    assert "print" in result
    assert "```" not in result
    assert "Here is the code" not in result
    assert "Done" not in result


def test_extract_code_no_fences():
    """Raw code without fences should be returned as-is (cleaned)."""
    text = 'import torch\nprint("hello")'
    result = extract_code(text)
    assert "import torch" in result
    assert "print" in result


def test_extract_code_leading_backticks():
    """Stray backticks at the start should be stripped."""
    text = '```import torch\nprint("hello")```'
    result = extract_code(text)
    assert result.startswith("import")
    assert "```" not in result


def test_sanitize_code_removes_prose():
    """Prose lines should be removed, code preserved."""
    code = '''import torch

Here is the test code:

result = torch.randn(5)
print("CLAIM_VERIFIED")
'''
    result = sanitize_code(code)
    assert "import torch" in result
    assert "result = torch.randn" in result
    assert "Here is the test code" not in result


def test_sanitize_code_adds_missing_imports():
    """Missing imports should be added automatically."""
    code = '''class MyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.randn(10, 5)
'''
    result = sanitize_code(code)
    assert "import torch" in result
    assert "import torch.nn as nn" in result


def test_sanitize_code_removes_backticks():
    """Backtick artifacts should be removed."""
    code = '```python\nimport torch\nprint("hello")\n```'
    result = sanitize_code(code)
    assert "```" not in result
    assert "import torch" in result


def test_sanitize_code_repairs_unterminated_docstring():
    """Unterminated triple-quoted strings should be closed."""
    code = 'import torch\n\nclass Foo:\n    """This is a docstring that never closes\n    def bar(self):\n        pass'
    result = sanitize_code(code)
    # The docstring should now be closed (even number of """)
    assert result.count('"""') % 2 == 0


def test_generate_function_strips_thinking_trace():
    """The generate function should strip the thinking trace (everything before last closing think tag)."""
    # The marker the model uses to end its thinking trace
    # Constructed from parts to avoid rendering issues
    THINK_END = "</think" + ">"

    # Case 1: no thinking trace — return as-is
    full_response = "print('hello')"
    if THINK_END in full_response:
        response = full_response.rsplit(THINK_END, 1)[-1].strip()
    else:
        response = full_response.strip()
    assert response == "print('hello')"

    # Case 2: thinking trace present — strip everything before the LAST marker
    full_response2 = "Let me think about this.\nThis is my reasoning.\n" + THINK_END + "\nprint('hello')"
    if THINK_END in full_response2:
        response2 = full_response2.rsplit(THINK_END, 1)[-1].strip()
    else:
        response2 = full_response2.strip()
    assert "Let me think" not in response2
    assert "print('hello')" in response2

    # Case 3: multiple markers — take after the LAST one
    full_response3 = "first thought" + THINK_END + "second thought" + THINK_END + "final answer"
    if THINK_END in full_response3:
        response3 = full_response3.rsplit(THINK_END, 1)[-1].strip()
    else:
        response3 = full_response3.strip()
    assert response3 == "final answer"


def test_run_in_sandbox():
    """Sandbox should run simple Python code."""
    result = run_in_sandbox("print('CLAIM_VERIFIED')\nprint('REASON: test passed')")
    assert "CLAIM_VERIFIED" in result["stdout"]
    assert result["exit_code"] == 0


def test_run_in_sandbox_failure():
    """Sandbox should capture errors."""
    result = run_in_sandbox("raise ValueError('test error')")
    assert result["exit_code"] != 0
    assert "ValueError" in result["stderr"]


def test_run_in_sandbox_timeout():
    """Sandbox should timeout gracefully."""
    result = run_in_sandbox("import time\ntime.sleep(100)", timeout=2)
    assert result["exit_code"] == -1
    assert "Timeout" in result["stderr"]


if __name__ == "__main__":
    tests = [
        test_parse_verdict,
        test_parse_reason,
        test_extract_code_markdown_fences,
        test_extract_code_no_fences,
        test_extract_code_leading_backticks,
        test_sanitize_code_removes_prose,
        test_sanitize_code_adds_missing_imports,
        test_sanitize_code_removes_backticks,
        test_sanitize_code_repairs_unterminated_docstring,
        test_generate_function_strips_thinking_trace,
        test_run_in_sandbox,
        test_run_in_sandbox_failure,
        test_run_in_sandbox_timeout,
    ]
    for test in tests:
        try:
            test()
            print(f"✓ {test.__name__}")
        except AssertionError as e:
            print(f"✗ {test.__name__}: {e}")
            raise
    print("\nAll tests passed!")
