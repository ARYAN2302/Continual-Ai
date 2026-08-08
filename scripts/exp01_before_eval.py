"""
=============================================================================
Experiment 01: Cold Start + Verify (BEFORE eval)
=============================================================================

Loads LFM2.5-2.6B cold (no adapter). Asks it to implement LoRA.
For each claim about LoRA, asks the model to write a test.
Runs the implementation + test in a sandboxed subprocess.
Records the pass rate as the BEFORE baseline.

This is the "before" half of the before/after comparison.

Run on Kaggle T4x2 (or any CUDA GPU with >=8GB VRAM).
Self-contained: installs all deps, downloads model, runs, saves results.

Usage:
    python exp01_before_eval.py

Output:
    {output_dir}/implementation.py   — the model's LoRA implementation
    {output_dir}/tests/claim_N.py    — the model's test for each claim
    {output_dir}/results.json        — full results with verdicts
    {output_dir}/summary.txt         — human-readable summary
=============================================================================
"""

import subprocess
import sys
import os
import json
import time
import tempfile
import traceback

# === Dependency Installation (Kaggle-safe) ===
def install(pkgs):
    """Install packages quietly."""
    for pkg in pkgs:
        try:
            __import__(pkg.split("[")[0].replace("-", "_"))
        except ImportError:
            print(f"  Installing {pkg}...")
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q", pkg],
                check=True,
            )

print("=== Checking dependencies ===")
install(["torch", "transformers>=4.46", "peft>=0.13", "bitsandbytes", "accelerate"])

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# === Configuration ===
MODEL_ID = "LiquidAI/LFM2.5-2.6B"
OUTPUT_DIR = os.environ.get("EXP_OUTPUT_DIR",
    "/kaggle/working/exp01" if os.path.exists("/kaggle") else os.path.join(os.path.dirname(__file__), "..", "download", "exp_results", "exp01")
)
OUTPUT_DIR = os.path.abspath(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "tests"), exist_ok=True)

METHOD = "LoRA (Low-Rank Adaptation)"

# The 5 claims we will verify. Each is specific enough to operationalize.
CLAIMS = [
    "LoRA adds two low-rank matrices A (of shape r x d_in) and B (of shape d_out x r) to an existing weight matrix W (of shape d_out x d_in).",
    "The rank r is strictly smaller than both d_in and d_out (e.g., r=8 when d_in=d_out=4096).",
    "The original weight matrix W is frozen — its requires_grad is set to False.",
    "The forward pass computes output = W @ x + B @ A @ x (scaled by alpha/r), where x is the input.",
    "LoRA uses fewer trainable parameters than full fine-tuning. Specifically, trainable params = r * (d_in + d_out) instead of d_in * d_out.",
]

MAX_NEW_TOKENS_IMPL = 2048
MAX_NEW_TOKENS_TEST = 1024
VERIFICATION_TIMEOUT = 30  # seconds per test
TEMPERATURE = 0.7

# === Model Loading ===
def load_model():
    """Load LFM2.5-2.6B in 4-bit NF4 on GPU."""
    print(f"Loading {MODEL_ID} in 4-bit NF4...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.eval()
    print(f"  Model loaded. Device map: {model.hf_device_map if hasattr(model, 'hf_device_map') else 'auto'}")
    return model, tokenizer

# === Generation ===
def generate(model, tokenizer, user_prompt, max_new_tokens=2048, temperature=0.7):
    """Generate a response from the model."""
    messages = [{"role": "user", "content": user_prompt}]
    try:
        input_ids = tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(model.device)
    except Exception:
        # Fallback: no chat template
        input_ids = tokenizer(user_prompt, return_tensors="pt").input_ids.to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id or tokenizer.pad_token_id,
        )
    response = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
    return response.strip()

# === Code Extraction ===
def extract_code(text):
    """Extract Python code from markdown fences or plain text."""
    # Try ```python ... ```
    if "```python" in text:
        start = text.index("```python") + len("```python")
        end = text.find("```", start)
        if end != -1:
            return text[start:end].strip()
    # Try ``` ... ```
    if "```" in text:
        parts = text.split("```")
        if len(parts) >= 3:
            # Take the second block (first code block)
            code = parts[1]
            if code.startswith("python\n"):
                code = code[7:]
            return code.strip()
    # No fences — return as-is (might be raw code)
    return text.strip()

# === Implementation Generation ===
IMPL_PROMPT = """You are an expert ML engineer. Implement {method} in PyTorch.

Write a COMPLETE, RUNNABLE Python file that:
1. Defines a LoRALayer class (or equivalent) that wraps a linear layer
2. The class must accept parameters: in_features, out_features, rank (r), alpha
3. Include a __main__ block that creates an instance and prints the shapes of A and B
4. Print "IMPL_READY" at the end if no errors

Requirements:
- Use torch.nn.Module
- Initialize A with kaiming_uniform and B with zeros (standard LoRA init)
- The forward method must accept a tensor x and return the modified output
- Do NOT use the peft library — implement from scratch

Write ONLY Python code. No explanation, no markdown formatting.
"""

def generate_implementation(model, tokenizer, method):
    """Ask the model to implement LoRA from scratch."""
    print("  Generating implementation...")
    response = generate(model, tokenizer, IMPL_PROMPT.format(method=method),
                        max_new_tokens=MAX_NEW_TOKENS_IMPL, temperature=TEMPERATURE)
    code = extract_code(response)
    return code

# === Test Generation ===
TEST_PROMPT = """You are an ML engineer verifying a claim about {method}.

Claim to verify: "{claim}"

Here is an implementation of {method}:

```python
{implementation}
```

Write a Python test that verifies THIS SPECIFIC claim. The test must:
1. Import nothing external except torch (and copy/use the implementation above)
2. Run a concrete check that tests the claim
3. Print exactly "CLAIM_VERIFIED" if the claim is true
4. Print exactly "CLAIM_FALSIFIED" if the claim is false
5. Print exactly "CLAIM_UNCLEAR" if the test cannot determine the answer
6. Print a brief reason after the verdict on a new line starting with "REASON:"

The test code must be self-contained — include the implementation inline if needed.

Write ONLY Python code. No explanation.
"""

def generate_test(model, tokenizer, method, claim, implementation):
    """Ask the model to write a test for a specific claim."""
    response = generate(model, tokenizer, TEST_PROMPT.format(
        method=method, claim=claim, implementation=implementation
    ), max_new_tokens=MAX_NEW_TOKENS_TEST, temperature=TEMPERATURE)
    return extract_code(response)

# === The Verifier (the core primitive) ===
def verify_claim(implementation_code, test_code, timeout=30):
    """
    The verifier. Runs implementation + test in a subprocess.
    Returns a dict with verdict, stdout, stderr.

    This is the load-bearing component. It does triple duty:
    - Write gate: only PASS claims become training data
    - Contradiction check: (used in later experiments)
    - Improvement gate: (used in after-eval to check drift)
    """
    # Combine implementation and test
    full_code = implementation_code + "\n\n# === TEST ===\n" + test_code

    # Write to a temp file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(full_code)
        temp_path = f.name

    try:
        # Run in subprocess with CPU-only (GPU is busy with the model)
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
        result = subprocess.run(
            [sys.executable, temp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
        stdout = result.stdout
        stderr = result.stderr

        # Parse verdict from stdout
        if "CLAIM_VERIFIED" in stdout:
            verdict = "PASS"
        elif "CLAIM_FALSIFIED" in stdout:
            verdict = "FAIL"
        elif "CLAIM_UNCLEAR" in stdout:
            verdict = "CONFUSED"
        else:
            # No marker found — probably crashed or didn't print
            verdict = "CONFUSED"

        # Extract reason if present
        reason = ""
        for line in stdout.split("\n"):
            if line.startswith("REASON:"):
                reason = line[len("REASON:"):].strip()
                break

        return {
            "verdict": verdict,
            "reason": reason,
            "stdout": stdout[-2000:],
            "stderr": stderr[-1000:] if stderr else "",
            "exit_code": result.returncode,
        }

    except subprocess.TimeoutExpired:
        return {
            "verdict": "CONFUSED",
            "reason": "Timeout",
            "stdout": "",
            "stderr": f"Process timed out after {timeout}s",
            "exit_code": -1,
        }
    except Exception as e:
        return {
            "verdict": "CONFUSED",
            "reason": str(e),
            "stdout": "",
            "stderr": traceback.format_exc()[-1000:],
            "exit_code": -1,
        }
    finally:
        os.unlink(temp_path)

# === Main Experiment ===
def main():
    print("=" * 70)
    print("EXPERIMENT 01: COLD START + VERIFY (BEFORE EVAL)")
    print("=" * 70)
    print(f"Model: {MODEL_ID}")
    print(f"Method: {METHOD}")
    print(f"Claims to verify: {len(CLAIMS)}")
    print(f"Output: {OUTPUT_DIR}")
    print()

    # Step 1: Load model
    print("--- Step 1: Loading model ---")
    t0 = time.time()
    model, tokenizer = load_model()
    print(f"  Loaded in {time.time()-t0:.1f}s")
    print()

    # Step 2: Generate implementation
    print("--- Step 2: Generating implementation ---")
    t0 = time.time()
    implementation = generate_implementation(model, tokenizer, METHOD)
    print(f"  Generated in {time.time()-t0:.1f}s")
    print(f"  Code length: {len(implementation)} chars, {len(implementation.splitlines())} lines")

    # Save implementation
    impl_path = os.path.join(OUTPUT_DIR, "implementation.py")
    with open(impl_path, "w") as f:
        f.write(implementation)
    print(f"  Saved to {impl_path}")
    print()

    # Quick sanity check: does the implementation run at all?
    print("--- Step 2b: Sanity check (does implementation run?) ---")
    sanity = verify_claim(implementation, "print('CLAIM_VERIFIED')\nprint('REASON: implementation runs')", timeout=15)
    print(f"  Sanity: {sanity['verdict']} (exit code {sanity['exit_code']})")
    if sanity["stderr"]:
        print(f"  stderr: {sanity['stderr'][:300]}")
    print()

    # Step 3: Verify each claim
    print("--- Step 3: Verifying claims ---")
    results = []
    for i, claim in enumerate(CLAIMS):
        print(f"\n  [{i+1}/{len(CLAIMS)}] {claim[:90]}...")
        t0 = time.time()

        # Generate test for this claim
        test_code = generate_test(model, tokenizer, METHOD, claim, implementation)
        print(f"    Test generated: {len(test_code)} chars")

        # Save test
        test_path = os.path.join(OUTPUT_DIR, "tests", f"claim_{i+1}.py")
        with open(test_path, "w") as f:
            f.write(implementation + "\n\n# === TEST ===\n" + test_code)

        # Verify
        result = verify_claim(implementation, test_code, timeout=VERIFICATION_TIMEOUT)
        result["claim"] = claim
        result["claim_index"] = i + 1
        result["test_code"] = test_code
        result["time"] = time.time() - t0
        results.append(result)

        print(f"    Verdict: {result['verdict']} ({result['time']:.1f}s)")
        if result["reason"]:
            print(f"    Reason: {result['reason']}")
        if result["verdict"] != "PASS" and result["stderr"]:
            print(f"    stderr: {result['stderr'][:200]}")

    # Step 4: Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    pass_count = sum(1 for r in results if r["verdict"] == "PASS")
    fail_count = sum(1 for r in results if r["verdict"] == "FAIL")
    confused_count = sum(1 for r in results if r["verdict"] == "CONFUSED")
    pass_rate = pass_count / len(results) if results else 0

    print(f"Model: {MODEL_ID}")
    print(f"Method: {METHOD}")
    print(f"Claims: {len(CLAIMS)}")
    print(f"  PASS:     {pass_count}")
    print(f"  FAIL:     {fail_count}")
    print(f"  CONFUSED: {confused_count}")
    print(f"Pass rate: {pass_rate:.1%}")
    print()

    # Per-claim breakdown
    print("Per-claim breakdown:")
    for r in results:
        marker = {"PASS": "✓", "FAIL": "✗", "CONFUSED": "?"}[r["verdict"]]
        print(f"  {marker} [{r['claim_index']}] {r['claim'][:80]}...")
        if r["reason"]:
            print(f"      → {r['reason'][:100]}")
    print()

    # Save full results
    summary = {
        "experiment": "exp01_cold_start",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": MODEL_ID,
        "method": METHOD,
        "num_claims": len(CLAIMS),
        "pass": pass_count,
        "fail": fail_count,
        "confused": confused_count,
        "pass_rate": pass_rate,
        "implementation": implementation,
        "results": results,
    }

    results_path = os.path.join(OUTPUT_DIR, "results.json")
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Full results saved to {results_path}")

    # Save human-readable summary
    summary_path = os.path.join(OUTPUT_DIR, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Experiment 01: Cold Start + Verify\n")
        f.write(f"{'='*50}\n\n")
        f.write(f"Model: {MODEL_ID}\n")
        f.write(f"Method: {METHOD}\n")
        f.write(f"Date: {summary['timestamp']}\n\n")
        f.write(f"Results:\n")
        f.write(f"  PASS:     {pass_count}/{len(CLAIMS)}\n")
        f.write(f"  FAIL:     {fail_count}/{len(CLAIMS)}\n")
        f.write(f"  CONFUSED: {confused_count}/{len(CLAIMS)}\n")
        f.write(f"  Pass rate: {pass_rate:.1%}\n\n")
        f.write(f"Per-claim:\n")
        for r in results:
            f.write(f"  [{r['verdict']}] {r['claim']}\n")
            if r["reason"]:
                f.write(f"    → {r['reason']}\n")
    print(f"Summary saved to {summary_path}")

    print(f"\n{'='*70}")
    print(f"BEFORE EVAL COMPLETE. Pass rate: {pass_rate:.1%}")
    print(f"{'='*70}")

    return summary

if __name__ == "__main__":
    main()
