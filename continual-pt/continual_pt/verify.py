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

Write a COMPLETE, RUNNABLE Python file that:
1. Implements {x} from scratch (no external libraries except torch)
2. Includes a __main__ block that creates an instance and runs it
3. Prints "IMPL_READY" at the end if successful

Write ONLY Python code. No explanation. No markdown formatting.
"""

TEST_CLAIM_PROMPT = """You are verifying a claim about {x}.

Claim: "{claim}"

Here is an implementation:

```python
{implementation}
```

Write a Python test that verifies THIS SPECIFIC claim. The test must:
1. Be self-contained (include the implementation inline if needed)
2. Use only torch (no other external libraries)
3. Print exactly "CLAIM_VERIFIED" if the claim is true
4. Print exactly "CLAIM_FALSIFIED" if the claim is false
5. Print exactly "CLAIM_UNCLEAR" if the test cannot determine
6. Print "REASON: <brief reason>" on the next line

Write ONLY Python code. No explanation.
"""


def extract_code(text: str) -> str:
    """Extract Python code from markdown fences or plain text.
    Strips leading whitespace from each line to prevent indentation SyntaxErrors."""
    code = text
    if "```python" in text:
        start = text.index("```python") + len("```python")
        end = text.find("```", start)
        if end != -1:
            code = text[start:end]
    elif "```" in text:
        parts = text.split("```")
        if len(parts) >= 3:
            code = parts[1]
            if code.startswith("python\n"):
                code = code[7:]
    # Strip leading/trailing whitespace and dedent
    lines = code.strip().split("\n")
    # Find minimum indentation (excluding empty lines)
    min_indent = float("inf")
    for line in lines:
        if line.strip():
            indent = len(line) - len(line.lstrip())
            min_indent = min(min_indent, indent)
    if min_indent > 0 and min_indent != float("inf"):
        lines = [line[min_indent:] if line.strip() else line for line in lines]
    return "\n".join(lines).strip()


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
    print(f"[verify:exec] Implementation: {len(implementation)} chars")

    # Quick sanity: does it run?
    sanity = run_in_sandbox(implementation + '\nprint("CLAIM_VERIFIED")\nprint("REASON: runs")',
                            timeout=config.verification_timeout)
    if sanity["exit_code"] != 0:
        print(f"[verify:exec] Implementation doesn't run: {sanity['stderr'][:200]}")
        # Still proceed — tests might work if they include their own fixed version

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

        # Run test in sandbox
        full_code = implementation + "\n\n# === TEST ===\n" + test_code
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
