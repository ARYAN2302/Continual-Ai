"""
The verify phase.

Two paths:
1. Executable: model implements X, writes tests, runtime runs them in a sandbox.
2. Non-executable: the 5-step process (decompose → reconstruct → search → trace lineage → verdict).

The verifier is the load-bearing component. It does triple duty:
- Write gate: only PASS/RECONSTRUCTED items become training data
- Retention check: re-run the original verifier on previously committed items
- Improvement gate: after weight update, re-verify to check if X was actually learned
"""

import subprocess
import sys
import os
import json
import time
import tempfile
import traceback
import re
from typing import List, Dict, Optional
from continual_pt import Verdict, VerifierType


# ============================================================
# EXECUTABLE VERIFIER
# ============================================================

IMPL_PROMPT = """You are an expert ML engineer. Implement {x} in PyTorch.

Output a COMPLETE, RUNNABLE Python file. The very first line MUST be an import statement.

Requirements:
- Start with imports: import torch, import torch.nn as nn, import math, etc.
- Implement {x} from scratch (no external libraries except torch)
- Include a __main__ block that creates an instance and runs it
- Print "IMPL_READY" at the end if successful
- ALL imports at the top. Do not reference any undefined module.

CRITICAL RULES:
- Output ONLY Python code. No markdown. No backticks. No explanation.
- Do NOT wrap code in ```python ``` fences.
- Do NOT write "Here is the implementation:" or any prose.
- The FIRST character of your response must be 'i' (from "import").

Example output format:
import torch
import torch.nn as nn

class MyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        # ...

if __name__ == "__main__":
    # ...
    print("IMPL_READY")

Now write the implementation. Begin with "import".
"""

TEST_CLAIM_PROMPT = """You are verifying a claim about {x}.

Claim: "{claim}"

Here is the implementation that will be tested:

{implementation}

Write a MINIMAL Python test that verifies THIS SPECIFIC claim against the implementation above.

CRITICAL RULES:
- DO NOT include the implementation. It will be prepended automatically.
- DO NOT define the class/function again. It already exists.
- Write ONLY the test code that USES the implementation.
- The implementation above will be available as-is when your test runs.

Output rules:
- Output ONLY Python code. No markdown. No backticks. No explanation.
- Do NOT wrap code in ```python ``` fences.
- Start with "import" or "#".
- Create an instance of the implementation and test the SPECIFIC claim.
- Print exactly "CLAIM_VERIFIED" if the claim is true.
- Print exactly "CLAIM_FALSIFIED" if the claim is false.
- Print exactly "CLAIM_UNCLEAR" if the test cannot determine.
- Print "REASON: <brief reason>" on the next line after the verdict.

Example (for a claim about a LinearLayer class):
# The implementation (LinearLayer) is already defined above.
# Write only the test:
import torch
layer = SimpleLinearLayer(out_features=5, in_features=3)
# Check the claim:
if layer.weight.shape == (5, 3):
    print("CLAIM_VERIFIED")
    print("REASON: weight shape is (out_features, in_features) = (5, 3)")
else:
    print("CLAIM_FALSIFIED")
    print("REASON: weight shape is", layer.weight.shape, "expected (5, 3)")

Now write the test. DO NOT include the implementation. Begin with "import" or "#".
"""


def extract_code(text: str) -> str:
    """Extract Python code from model output. Handles:
    - Markdown fences (```python ... ``` or ``` ... ```)
    - Leading backticks/whitespace
    - Prose before/after code
    - Dedents indented code blocks

    Strategy: find the longest contiguous block of lines that looks like Python
    (starts with import, from, class, def, or # comment), and return it.
    """
    import re

    # Step 1: Try to extract from markdown fences first
    code = text

    # Match ```python ... ``` or ``` ... ```
    fence_pattern = re.compile(r'```(?:python)?\s*\n(.*?)```', re.DOTALL)
    matches = fence_pattern.findall(text)
    if matches:
        # Take the longest match (most likely the actual code)
        code = max(matches, key=len)
    else:
        # No fences — try to find code by looking for Python-like lines
        lines = text.split("\n")
        code_lines = []
        in_code = False
        for line in lines:
            stripped = line.strip()
            # Start capturing when we see a Python-like line
            if not in_code:
                if (stripped.startswith(("import ", "from ", "class ", "def ", "#", "@"))
                    or stripped == ""
                    or (stripped and not stripped[0].isalpha())):
                    # Heuristic: if line starts with a Python keyword, treat as code
                    if any(stripped.startswith(kw) for kw in
                           ["import ", "from ", "class ", "def ", "#", "@", "if ", "for ",
                            "while ", "try:", "with ", "return ", "print(", "raise "]):
                        in_code = True
            if in_code:
                # Stop if we hit a line that's clearly prose (long sentence with no code chars)
                if stripped and not any(c in stripped for c in "=:()[]{}#"):
                    # Check if it's prose (starts with capital, ends with period, no code syntax)
                    if (stripped[0].isupper() and stripped.endswith(".")
                        and "=" not in stripped and "(" not in stripped):
                        break
                code_lines.append(line)
        if code_lines:
            code = "\n".join(code_lines)

    # Step 2: Clean up the extracted code
    lines = code.split("\n")

    # Remove leading empty lines
    while lines and not lines[0].strip():
        lines.pop(0)
    # Remove trailing empty lines
    while lines and not lines[-1].strip():
        lines.pop()

    # Step 3: Dedent (find min indentation, strip it)
    min_indent = float("inf")
    for line in lines:
        if line.strip():
            indent = len(line) - len(line.lstrip())
            min_indent = min(min_indent, indent)
    if 0 < min_indent != float("inf"):
        lines = [line[min_indent:] if line.strip() else line for line in lines]

    # Step 4: Strip any remaining leading/trailing backticks or quotes
    result = "\n".join(lines).strip()
    # Remove stray backticks at start
    while result.startswith("`"):
        result = result[1:].strip()
    # Remove stray backticks at end
    while result.endswith("`"):
        result = result[:-1].strip()

    return result


def sanitize_code(code: str) -> str:
    """Final sanitization pass before running code in sandbox.
    Handles common model output issues:
    - Stray markdown artifacts
    - Prose lines mixed with code
    - Missing imports
    - Unescaped string literals
    """
    import re

    # Remove any remaining backtick sequences
    code = code.replace("```python", "").replace("```", "")

    # Remove lines that are clearly prose (not Python)
    lines = code.split("\n")
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        # Skip empty lines (keep them for formatting)
        if not stripped:
            cleaned_lines.append(line)
            continue
        # Skip lines that look like prose explanations
        # Prose indicators: starts with "Here", "This", "Note:", "Now", "Let", etc.
        # and doesn't contain Python syntax
        prose_starters = [
            "Here is", "Here's", "This is", "This will", "Note:", "Now ",
            "Let me", "Let's", "The following", "Below is", "First,",
            "Second,", "Finally,", "In this", "To verify", "We need",
            "The test", "The code", "The implementation", "I'll",
            "We'll", "This code", "This test", "This implementation",
            "That concludes", "This concludes", "The above",
        ]
        is_prose = any(stripped.startswith(p) for p in prose_starters)
        # A line is prose if it starts with a prose starter AND doesn't look like an assignment/call
        # (i.e., no '=' that's not inside a string, no '(' that indicates a function call)
        looks_like_assignment = bool(re.match(r'^\w+\s*=', stripped))
        looks_like_call = '(' in stripped and not stripped.startswith(("Here", "This", "Note", "Now", "Let", "The", "Below", "First", "Second", "Finally", "In ", "To ", "We ", "I'll", "That", "The above"))
        if is_prose and not (looks_like_assignment or looks_like_call):
            continue
        # Skip lines that are just "Python code:" or similar labels
        if stripped in ["Python code:", "Python:", "Code:", "Test:", "Test code:",
                       "Implementation:", "Implementation code:"]:
            continue
        cleaned_lines.append(line)

    code = "\n".join(cleaned_lines)

    # Dedent again in case sanitization introduced indentation
    lines = code.split("\n")
    min_indent = float("inf")
    for line in lines:
        if line.strip():
            indent = len(line) - len(line.lstrip())
            min_indent = min(min_indent, indent)
    if 0 < min_indent != float("inf"):
        lines = [line[min_indent:] if line.strip() else line for line in lines]
    code = "\n".join(lines)

    # Ensure there's at least an import if the code references torch/nn
    if "torch" in code and "import torch" not in code:
        code = "import torch\n" + code
    if "nn." in code and "import torch.nn" not in code and "from torch import nn" not in code:
        code = "import torch.nn as nn\n" + code
    if "math." in code and "import math" not in code:
        code = "import math\n" + code
    if "F." in code and "torch.nn.functional" not in code and "import torch.nn.functional" not in code:
        code = "import torch.nn.functional as F\n" + code

    # Repair unterminated string literals (common when model runs out of tokens mid-docstring)
    code = repair_unterminated_strings(code)

    return code.strip()


def repair_unterminated_strings(code: str) -> str:
    """Fix unterminated triple-quoted strings, single quotes, and double quotes.
    Common when the model hits max_new_tokens mid-generation.

    For triple-quoted strings (docstrings): if there's an odd number of \"\"\", append a closing one.
    For single/double quotes: only fix if they're clearly unbalanced at the end of a line.
    """
    import re

    # Count triple-quoted strings
    triple_count = code.count('"""')
    if triple_count % 2 == 1:
        # Odd number — the last one is unterminated. Append a closing """.
        code = code.rstrip() + '\n"""'
        # Add a pass statement in case the docstring was inside a function/class body
        # that expected more indented code
        code += '\npass  # auto-added: repair_unterminated_strings'

    # Also handle ''' triple-quoted strings
    triple_single_count = code.count("'''")
    if triple_single_count % 2 == 1:
        code = code.rstrip() + "\n'''"
        code += '\npass  # auto-added: repair_unterminated_strings'

    return code


def run_in_sandbox(code: str, timeout: int = 60) -> Dict:
    """Run Python code in a subprocess sandbox. Returns stdout, stderr, exit_code."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        temp_path = f.name

    try:
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}  # CPU only for sandbox
        result = subprocess.run(
            [sys.executable, temp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "exit_code": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"stdout": "", "stderr": f"Timeout after {timeout}s", "exit_code": -1}
    except Exception as e:
        return {"stdout": "", "stderr": str(e), "exit_code": -1}
    finally:
        os.unlink(temp_path)


def parse_verdict(stdout: str) -> Verdict:
    """Parse the verdict from sandbox output."""
    if "CLAIM_VERIFIED" in stdout:
        return Verdict.PASS
    elif "CLAIM_FALSIFIED" in stdout:
        return Verdict.FAIL
    elif "CLAIM_UNCLEAR" in stdout:
        return Verdict.CONFUSED
    return Verdict.CONFUSED


def parse_reason(stdout: str) -> str:
    """Extract the REASON line from output."""
    for line in stdout.split("\n"):
        if line.startswith("REASON:"):
            return line[len("REASON:"):].strip()
    return ""


def verify_executable(model, tokenizer, x: str, claims: List[str], config) -> Dict:
    """
    Executable verifier path.

    1. Model implements X
    2. For each claim, model writes a test
    3. Runtime runs implementation + test in sandbox
    4. Returns per-claim verdicts

    Returns: {
        "implementation": str,
        "claims": [{"claim": str, "verdict": Verdict, "reason": str, "stdout": str, "stderr": str}],
        "all_pass": bool,
    }
    """
    from continual_pt.model import generate

    print(f"\n[verify:exec] Verifying {x} ({len(claims)} claims)")

    # Step 1: Generate implementation
    print("[verify:exec] Generating implementation...")
    impl_response = generate(model, tokenizer,
                             IMPL_PROMPT.format(x=x),
                             max_new_tokens=config.max_new_tokens_impl,
                             temperature=config.temperature)
    implementation = extract_code(impl_response)
    implementation = sanitize_code(implementation)
    print(f"[verify:exec] Implementation: {len(implementation)} chars")

    # Quick sanity: does it run?
    sanity_code = sanitize_code(implementation + '\nprint("CLAIM_VERIFIED")\nprint("REASON: runs")')
    sanity = run_in_sandbox(sanity_code, timeout=config.verification_timeout)
    if sanity["exit_code"] != 0:
        print(f"[verify:exec] Implementation doesn't run: {sanity['stderr'][:200]}")
        # Still proceed — tests might work if they include their own fixed version
    else:
        print(f"[verify:exec] Implementation runs OK")

    # Step 2: Verify each claim
    results = []
    for i, claim in enumerate(claims):
        print(f"\n  [{i+1}/{len(claims)}] {claim[:80]}...")
        t0 = time.time()

        # Generate test
        test_response = generate(model, tokenizer,
                                TEST_CLAIM_PROMPT.format(x=x, claim=claim,
                                                         implementation=implementation),
                                max_new_tokens=config.max_new_tokens_test,
                                temperature=config.temperature)
        test_code = extract_code(test_response)
        test_code = sanitize_code(test_code)

        # Run test in sandbox
        full_code = sanitize_code(implementation + "\n\n# === TEST ===\n" + test_code)
        sandbox_result = run_in_sandbox(full_code, timeout=config.verification_timeout)

        verdict = parse_verdict(sandbox_result["stdout"])
        reason = parse_reason(sandbox_result["stdout"])
        elapsed = time.time() - t0

        print(f"    → {verdict.value} ({elapsed:.1f}s)")
        if reason:
            print(f"      {reason[:100]}")

        results.append({
            "claim": claim,
            "verdict": verdict.value,
            "reason": reason,
            "test_code": test_code,
            "stdout": sandbox_result["stdout"][-2000:],
            "stderr": sandbox_result["stderr"][-1000:] if sandbox_result["stderr"] else "",
            "time": elapsed,
        })

    all_pass = all(r["verdict"] == Verdict.PASS.value for r in results)

    return {
        "verifier_type": "executable",
        "implementation": implementation,
        "claims": results,
        "all_pass": all_pass,
        "pass_count": sum(1 for r in results if r["verdict"] == Verdict.PASS.value),
        "fail_count": sum(1 for r in results if r["verdict"] == Verdict.FAIL.value),
        "confused_count": sum(1 for r in results if r["verdict"] == Verdict.CONFUSED.value),
    }


# ============================================================
# NON-EXECUTABLE VERIFIER (the 5-step process)
# ============================================================

DECOMPOSE_PROMPT = """You are analyzing a claim about: {x}

Claim: "{claim}"

Decompose this claim into independent, separately-checkable sub-claims.
Each sub-claim should be a single, testable assertion that can be verified independently.

If the claim is already a single assertion, return it as the only sub-claim.

Output as JSON array of strings. Each string is one sub-claim.
Example: ["sub-claim 1", "sub-claim 2"]

Output ONLY the JSON array. No explanation.
"""

RECONSTRUCT_PROMPT = """You are reconstructing the mechanism behind a claim.

Claim: "{sub_claim}"

Try to derive, step by step, WHY this claim would be true.
- Show the reasoning chain
- If there's a gap you can't explain, mark it explicitly with "GAP: ..."
- If you cannot reconstruct the mechanism at all, say "CANNOT_RECONSTRUCT"

This reconstruction is necessary but NOT sufficient for verification.
It qualifies the claim for external checking — it is never itself the verification.

Output your reconstruction as a clear step-by-step explanation.
"""

SEARCH_QUERY_PROMPT = """You need to find external evidence for this sub-claim:

"{sub_claim}"

Generate 2 web search queries to find independent, authoritative sources.
Focus on:
- Primary sources (papers, official docs, experiments)
- Independent analyses (not sources that cite other sources)

Output as JSON array of 2 search query strings.
Output ONLY the JSON array.
"""

LINEAGE_PROMPT = """You are classifying the lineage of a source.

Sub-claim being verified: "{sub_claim}"

Source URL: {url}
Source title: {title}
Source content (first 1500 chars):
{content}

Classify this source as:
- INDEPENDENT: It runs its own experiment, derivation, or ablation. It provides original data or analysis.
- DERIVATIVE: It cites, agrees with, or summarizes another source without independent testing.

If INDEPENDENT, also state what original evidence it provides.
If DERIVATIVE, state what source it appears to derive from (if mentioned).

Output as JSON:
{{
  "classification": "INDEPENDENT" or "DERIVATIVE",
  "evidence": "what original evidence it provides, or what it derives from",
  "supports": true/false (does it support the sub-claim?)
}}

Output ONLY the JSON.
"""

VERDICT_PROMPT = """You are making a final verdict on a sub-claim.

Sub-claim: "{sub_claim}"

Reconstruction:
{reconstruction}

Independent sources found:
{independent_sources}

Contradicting independent sources:
{contradicting_sources}

Make a verdict:
- RECONSTRUCTED: The mechanism holds (no gaps), AND at least one independent source confirms, AND no independent source contradicts.
- OPEN: The mechanism holds, but there is insufficient independent evidence either way.
- DISCARD: The mechanism has a gap, OR independent evidence contradicts.

Output as JSON:
{{
  "verdict": "RECONSTRUCTED" or "OPEN" or "DISCARD",
  "reasoning": "brief explanation"
}}

Output ONLY the JSON.
"""


def verify_non_executable(model, tokenizer, x: str, claim: str, config) -> Dict:
    """
    Non-executable verifier: the 5-step process.

    1. Decompose — split into sub-claims
    2. Reconstruct — derive why (necessary, not sufficient)
    3. Search — external evidence
    4. Trace lineage — INDEPENDENT vs DERIVATIVE
    5. Verdict — per sub-claim

    Returns: {
        "claim": str,
        "sub_claims": [{"sub_claim": str, "reconstruction": str, "verdict": str, ...}],
        "overall_verdict": Verdict,
    }
    """
    from continual_pt.model import generate
    from continual_pt.absorb import duckduckgo_search, fetch_page

    print(f"\n[verify:non-exec] Verifying claim about {x}")
    print(f"  Claim: {claim[:100]}...")

    # Step 1: Decompose
    print("\n  [Step 1] Decomposing into sub-claims...")
    response = generate(model, tokenizer,
                        DECOMPOSE_PROMPT.format(x=x, claim=claim),
                        max_new_tokens=1024,
                        temperature=config.temperature)
    try:
        sub_claims = json.loads(response.strip())
        if not isinstance(sub_claims, list):
            sub_claims = [claim]
    except json.JSONDecodeError:
        match = re.search(r'\[.*\]', response, re.DOTALL)
        if match:
            try:
                sub_claims = json.loads(match.group())
            except:
                sub_claims = [claim]
        else:
            sub_claims = [claim]

    print(f"  Decomposed into {len(sub_claims)} sub-claims")
    for i, sc in enumerate(sub_claims):
        print(f"    {i+1}. {sc[:80]}...")

    # Steps 2-5 per sub-claim
    sub_results = []
    for i, sc in enumerate(sub_claims):
        print(f"\n  [Sub-claim {i+1}] {sc[:80]}...")

        # Step 2: Reconstruct
        print("    [Step 2] Reconstructing mechanism...")
        reconstruction = generate(model, tokenizer,
                                  RECONSTRUCT_PROMPT.format(sub_claim=sc),
                                  max_new_tokens=2048,
                                  temperature=config.temperature)
        has_gap = "GAP:" in reconstruction or "CANNOT_RECONSTRUCT" in reconstruction
        print(f"    Reconstruction: {'has GAP' if has_gap else 'no gaps'} ({len(reconstruction)} chars)")

        if has_gap:
            sub_results.append({
                "sub_claim": sc,
                "reconstruction": reconstruction,
                "verdict": Verdict.DISCARD.value,
                "reasoning": "Mechanism has a gap — cannot reconstruct",
                "sources": [],
            })
            print(f"    → {Verdict.DISCARD.value} (gap in mechanism)")
            continue

        # Step 3: Search for external evidence
        print("    [Step 3] Searching for external evidence...")
        search_query_response = generate(model, tokenizer,
                                         SEARCH_QUERY_PROMPT.format(sub_claim=sc),
                                         max_new_tokens=512,
                                         temperature=config.temperature)
        try:
            search_queries = json.loads(search_query_response.strip())
        except:
            search_queries = [sc[:100]]

        sources = []
        for sq in search_queries[:2]:
            results = duckduckgo_search(sq, max_results=3, delay=config.search_delay)
            for r in results[:2]:
                page = fetch_page(r["url"])
                if page.get("text"):
                    sources.append(page)

        print(f"    Found {len(sources)} sources")

        # Step 4: Trace lineage
        print("    [Step 4] Tracing lineage...")
        independent_sources = []
        contradicting_sources = []

        for src in sources:
            lineage_response = generate(model, tokenizer,
                                        LINEAGE_PROMPT.format(
                                            sub_claim=sc,
                                            url=src["url"],
                                            title=src["title"],
                                            content=src["text"][:1500]
                                        ),
                                        max_new_tokens=512,
                                        temperature=config.temperature)
            try:
                lineage = json.loads(lineage_response.strip())
                # Try to extract JSON from response
                if not isinstance(lineage, dict):
                    match = re.search(r'\{.*\}', lineage_response, re.DOTALL)
                    if match:
                        lineage = json.loads(match.group())
            except json.JSONDecodeError:
                match = re.search(r'\{.*\}', lineage_response, re.DOTALL)
                if match:
                    try:
                        lineage = json.loads(match.group())
                    except:
                        lineage = {"classification": "DERIVATIVE", "evidence": "parse error", "supports": False}
                else:
                    lineage = {"classification": "DERIVATIVE", "evidence": "parse error", "supports": False}

            src["lineage"] = lineage

            if lineage.get("classification") == "INDEPENDENT":
                if lineage.get("supports"):
                    independent_sources.append(src)
                else:
                    contradicting_sources.append(src)

        print(f"    Independent supporting: {len(independent_sources)}")
        print(f"    Independent contradicting: {len(contradicting_sources)}")

        # Step 5: Verdict
        print("    [Step 5] Making verdict...")
        verdict_response = generate(model, tokenizer,
                                    VERDICT_PROMPT.format(
                                        sub_claim=sc,
                                        reconstruction=reconstruction,
                                        independent_sources=json.dumps([
                                            {"url": s["url"], "title": s["title"],
                                             "evidence": s.get("lineage", {}).get("evidence", "")}
                                            for s in independent_sources
                                        ], indent=2),
                                        contradicting_sources=json.dumps([
                                            {"url": s["url"], "title": s["title"]}
                                            for s in contradicting_sources
                                        ], indent=2),
                                    ),
                                    max_new_tokens=512,
                                    temperature=config.temperature)
        try:
            verdict_data = json.loads(verdict_response.strip())
        except json.JSONDecodeError:
            match = re.search(r'\{.*\}', verdict_response, re.DOTALL)
            if match:
                try:
                    verdict_data = json.loads(match.group())
                except:
                    verdict_data = {"verdict": "OPEN", "reasoning": "parse error"}
            else:
                verdict_data = {"verdict": "OPEN", "reasoning": "parse error"}

        verdict_str = verdict_data.get("verdict", "OPEN").upper()
        if verdict_str == "RECONSTRUCTED":
            verdict = Verdict.RECONSTRUCTED
        elif verdict_str == "DISCARD":
            verdict = Verdict.DISCARD
        else:
            verdict = Verdict.OPEN

        print(f"    → {verdict.value}: {verdict_data.get('reasoning', '')[:100]}")

        sub_results.append({
            "sub_claim": sc,
            "reconstruction": reconstruction,
            "verdict": verdict.value,
            "reasoning": verdict_data.get("reasoning", ""),
            "sources": sources,
            "independent_count": len(independent_sources),
            "contradicting_count": len(contradicting_sources),
        })

    # Overall verdict: worst sub-claim wins
    # DISCARD > OPEN > RECONSTRUCTED (worst to best)
    if any(r["verdict"] == Verdict.DISCARD.value for r in sub_results):
        overall = Verdict.DISCARD
    elif any(r["verdict"] == Verdict.OPEN.value for r in sub_results):
        overall = Verdict.OPEN
    else:
        overall = Verdict.RECONSTRUCTED

    print(f"\n  Overall verdict: {overall.value}")

    return {
        "verifier_type": "non_executable",
        "claim": claim,
        "sub_claims": sub_results,
        "overall_verdict": overall.value,
        "pass": overall == Verdict.RECONSTRUCTED,
    }


# ============================================================
# VERIFIER ROUTER
# ============================================================

def verify(model, tokenizer, x: str, config,
           verifier_type: VerifierType = VerifierType.AUTO,
           claims: Optional[List[str]] = None,
           claim: Optional[str] = None) -> Dict:
    """
    Route to the appropriate verifier.

    For executable X: pass claims (list of testable assertions).
    For non-executable X: pass a single claim (the assertion to verify).
    For AUTO: try executable first; if no claims provided, use non-executable.
    """
    if verifier_type == VerifierType.EXECUTABLE or (
        verifier_type == VerifierType.AUTO and claims is not None
    ):
        if claims is None:
            claims = [f"{x} works as described"]
        return verify_executable(model, tokenizer, x, claims, config)
    else:
        if claim is None:
            claim = x  # treat X itself as the claim
        return verify_non_executable(model, tokenizer, x, claim, config)
