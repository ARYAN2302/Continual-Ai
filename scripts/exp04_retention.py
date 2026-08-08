"""
=============================================================================
Experiment 04: Retention Check — Re-verify LoRA After GRPO Training
=============================================================================

The critical test: after training the adapter on LoRA (exp02) and testing on
GRPO (exp03), does the model STILL know LoRA?

This is the avr-cl retention check. If the adapter caused the model to forget
LoRA while learning GRPO, that's catastrophic forgetting — the exact failure
mode the avr-cl gate is designed to prevent.

Run on Kaggle T4x2.
Self-contained: loads adapter, re-runs the LoRA verification from exp01.

Usage:
    python exp04_retention.py

Inputs:
    {exp01_dir}/results.json  — original LoRA results (baseline)
    {exp02_dir}/adapter/      — trained adapter

Outputs:
    {output_dir}/retention_results.json — re-verification results
    {output_dir}/comparison.json        — before vs after comparison
    {output_dir}/summary.txt            — human-readable summary
=============================================================================
"""

import subprocess
import sys
import os
import json
import time
import tempfile
import traceback

# === Dependencies ===
def install(pkgs):
    for pkg in pkgs:
        try:
            __import__(pkg.split("[")[0].replace("-", "_"))
        except ImportError:
            print(f"  Installing {pkg}...")
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=True)

print("=== Checking dependencies ===")
install(["torch", "transformers>=4.46", "peft>=0.13", "bitsandbytes", "accelerate"])

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# === Config ===
MODEL_ID = "LiquidAI/LFM2.5-2.6B"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
EXP01_DIR = os.environ.get("EXP01_DIR",
    os.path.join(PROJECT_ROOT, "download", "exp_results", "exp01"))
EXP02_DIR = os.environ.get("EXP02_DIR",
    os.path.join(PROJECT_ROOT, "download", "exp_results", "exp02"))
EXP01_DIR = os.path.abspath(EXP01_DIR)
EXP02_DIR = os.path.abspath(EXP02_DIR)

OUTPUT_DIR = os.environ.get("EXP_OUTPUT_DIR",
    "/kaggle/working/exp04" if os.path.exists("/kaggle") else os.path.join(PROJECT_ROOT, "download", "exp_results", "exp04"))
OUTPUT_DIR = os.path.abspath(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "tests"), exist_ok=True)

METHOD = "LoRA (Low-Rank Adaptation)"

# Same claims as exp01 — must match exactly
CLAIMS = [
    "LoRA adds two low-rank matrices A (of shape r x d_in) and B (of shape d_out x r) to an existing weight matrix W (of shape d_out x d_in).",
    "The rank r is strictly smaller than both d_in and d_out (e.g., r=8 when d_in=d_out=4096).",
    "The original weight matrix W is frozen — its requires_grad is set to False.",
    "The forward pass computes output = W @ x + B @ A @ x (scaled by alpha/r), where x is the input.",
    "LoRA uses fewer trainable parameters than full fine-tuning. Specifically, trainable params = r * (d_in + d_out) instead of d_in * d_out.",
]

MAX_NEW_TOKENS_IMPL = 2048
MAX_NEW_TOKENS_TEST = 1024
VERIFICATION_TIMEOUT = 30
TEMPERATURE = 0.7

# === Model Loading ===
def load_model(with_adapter=True, adapter_path=None):
    print(f"Loading {MODEL_ID} (adapter={with_adapter})...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=bnb_config,
        device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True,
    )
    model.eval()

    if with_adapter and adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
        model.eval()
        print(f"  Adapter loaded from {adapter_path}")

    return model, tokenizer

# === Generation ===
def generate(model, tokenizer, user_prompt, max_new_tokens=2048, temperature=0.7):
    messages = [{"role": "user", "content": user_prompt}]
    try:
        input_ids = tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True
        ).to(model.device)
    except Exception:
        input_ids = tokenizer(user_prompt, return_tensors="pt").input_ids.to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            input_ids, max_new_tokens=max_new_tokens, do_sample=True,
            temperature=temperature, top_p=0.9, pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True).strip()

def extract_code(text):
    if "```python" in text:
        start = text.index("```python") + len("```python")
        end = text.find("```", start)
        if end != -1:
            return text[start:end].strip()
    if "```" in text:
        parts = text.split("```")
        if len(parts) >= 3:
            code = parts[1]
            if code.startswith("python\n"):
                code = code[7:]
            return code.strip()
    return text.strip()

# === Prompts (must match exp01 exactly) ===
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

def generate_implementation(model, tokenizer, method):
    response = generate(model, tokenizer, IMPL_PROMPT.format(method=method),
                        max_new_tokens=MAX_NEW_TOKENS_IMPL, temperature=TEMPERATURE)
    return extract_code(response)

def generate_test(model, tokenizer, method, claim, implementation):
    response = generate(model, tokenizer, TEST_PROMPT.format(
        method=method, claim=claim, implementation=implementation
    ), max_new_tokens=MAX_NEW_TOKENS_TEST, temperature=TEMPERATURE)
    return extract_code(response)

# === Verifier ===
def verify_claim(implementation_code, test_code, timeout=30):
    full_code = implementation_code + "\n\n# === TEST ===\n" + test_code
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(full_code)
        temp_path = f.name
    try:
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
        result = subprocess.run([sys.executable, temp_path], capture_output=True,
                               text=True, timeout=timeout, env=env)
        stdout, stderr = result.stdout, result.stderr
        if "CLAIM_VERIFIED" in stdout:
            verdict = "PASS"
        elif "CLAIM_FALSIFIED" in stdout:
            verdict = "FAIL"
        elif "CLAIM_UNCLEAR" in stdout:
            verdict = "CONFUSED"
        else:
            verdict = "CONFUSED"
        reason = ""
        for line in stdout.split("\n"):
            if line.startswith("REASON:"):
                reason = line[len("REASON:"):].strip()
                break
        return {"verdict": verdict, "reason": reason,
                "stdout": stdout[-2000:], "stderr": stderr[-1000:] if stderr else "",
                "exit_code": result.returncode}
    except subprocess.TimeoutExpired:
        return {"verdict": "CONFUSED", "reason": "Timeout", "stdout": "", "stderr": f"Timeout {timeout}s", "exit_code": -1}
    except Exception as e:
        return {"verdict": "CONFUSED", "reason": str(e), "stdout": "", "stderr": traceback.format_exc()[-1000:], "exit_code": -1}
    finally:
        os.unlink(temp_path)

# === Main ===
def main():
    print("=" * 70)
    print("EXPERIMENT 04: RETENTION CHECK — Re-verify LoRA After GRPO Training")
    print("=" * 70)
    print(f"Model: {MODEL_ID}")
    print(f"Method: {METHOD} (re-testing after GRPO experiment)")
    print()

    # Load exp01 baseline
    exp01_path = os.path.join(EXP01_DIR, "results.json")
    if not os.path.exists(exp01_path):
        print(f"ERROR: {exp01_path} not found. Run exp01 first.")
        return
    with open(exp01_path) as f:
        exp01_data = json.load(f)
    baseline_pass_rate = exp01_data.get("pass_rate", 0)
    print(f"Baseline (exp01 cold): {baseline_pass_rate:.1%}")

    # Load adapter
    adapter_path = os.path.join(EXP02_DIR, "adapter")
    if not os.path.exists(adapter_path):
        print(f"ERROR: Adapter not found at {adapter_path}")
        return

    # Load model WITH adapter
    print("\n--- Loading model with adapter ---")
    model, tokenizer = load_model(with_adapter=True, adapter_path=adapter_path)

    # Re-run LoRA verification
    print(f"\n--- Re-verifying {METHOD} claims (with adapter) ---")
    impl = generate_implementation(model, tokenizer, METHOD)
    print(f"  Implementation: {len(impl)} chars")

    impl_path = os.path.join(OUTPUT_DIR, "retention_implementation.py")
    with open(impl_path, "w") as f:
        f.write(impl)

    results = []
    for i, claim in enumerate(CLAIMS):
        print(f"\n  [{i+1}/{len(CLAIMS)}] {claim[:80]}...")
        t0 = time.time()
        test = generate_test(model, tokenizer, METHOD, claim, impl)
        result = verify_claim(impl, test, timeout=VERIFICATION_TIMEOUT)
        result["claim"] = claim
        result["claim_index"] = i + 1
        result["test_code"] = test
        result["time"] = time.time() - t0
        results.append(result)
        print(f"    → {result['verdict']} ({result['time']:.1f}s)")
        if result["reason"]:
            print(f"      {result['reason'][:100]}")

    pass_count = sum(1 for r in results if r["verdict"] == "PASS")
    fail_count = sum(1 for r in results if r["verdict"] == "FAIL")
    confused_count = sum(1 for r in results if r["verdict"] == "CONFUSED")
    pass_rate = pass_count / len(CLAIMS) if CLAIMS else 0

    # Get exp01 per-claim results for comparison
    exp01_per_claim = exp01_data.get("results", [])

    # Comparison
    print("\n" + "=" * 70)
    print("RETENTION COMPARISON")
    print("=" * 70)
    print(f"\n{'Metric':<25} {'Exp01 (cold)':>15} {'Exp04 (adapter)':>17} {'Delta':>10}")
    print("-" * 67)
    print(f"{'Pass rate':<25} {baseline_pass_rate:>14.1%} {pass_rate:>16.1%} {pass_rate-baseline_pass_rate:>+9.1%}")
    print(f"{'Pass count':<25} {exp01_data.get('pass',0):>15} {pass_count:>17} {pass_count-exp01_data.get('pass',0):>+10}")
    print(f"{'Fail count':<25} {exp01_data.get('fail',0):>15} {fail_count:>17} {fail_count-exp01_data.get('fail',0):>+10}")
    print(f"{'Confused count':<25} {exp01_data.get('confused',0):>15} {confused_count:>17} {confused_count-exp01_data.get('confused',0):>+10}")

    print(f"\nPer-claim retention:")
    print(f"{'#':<3} {'Exp01':>8} {'Exp04':>8} {'Retained?':>10} {'Claim':<50}")
    print("-" * 80)
    for i in range(len(CLAIMS)):
        old = exp01_per_claim[i]["verdict"] if i < len(exp01_per_claim) else "?"
        new = results[i]["verdict"] if i < len(results) else "?"
        # Retained = same or improved
        if old == new:
            retained = "SAME"
        elif old != "PASS" and new == "PASS":
            retained = "GAINED"
        elif old == "PASS" and new != "PASS":
            retained = "LOST"
        else:
            retained = "CHANGED"
        print(f"{i+1:<3} {old:>8} {new:>8} {retained:>10} {CLAIMS[i][:50]}")

    # Retention verdict
    # Good: pass_rate >= baseline (no forgetting)
    # OK: pass_rate dropped by < 1 claim
    # Bad: pass_rate dropped by >= 1 claim (catastrophic forgetting)
    delta = pass_rate - baseline_pass_rate
    if delta >= 0:
        retention_verdict = "RETAINED — no forgetting"
    elif delta > -0.2:
        retention_verdict = "MINOR DRIFT — acceptable"
    else:
        retention_verdict = "CATASTROPHIC FORGETTING — avr-cl gate failed"

    print(f"\nRetention verdict: {retention_verdict}")

    # Save
    retention_data = {
        "experiment": "exp04_retention",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": MODEL_ID,
        "method": METHOD,
        "baseline_pass_rate": baseline_pass_rate,
        "retention_pass_rate": pass_rate,
        "delta": delta,
        "retention_verdict": retention_verdict,
        "num_claims": len(CLAIMS),
        "pass": pass_count,
        "fail": fail_count,
        "confused": confused_count,
        "implementation": impl,
        "results": results,
        "exp01_comparison": [
            {
                "claim": CLAIMS[i],
                "exp01_verdict": exp01_per_claim[i]["verdict"] if i < len(exp01_per_claim) else "?",
                "exp04_verdict": results[i]["verdict"] if i < len(results) else "?",
            }
            for i in range(len(CLAIMS))
        ],
    }

    rpath = os.path.join(OUTPUT_DIR, "retention_results.json")
    with open(rpath, "w") as f:
        json.dump(retention_data, f, indent=2)

    cpath = os.path.join(OUTPUT_DIR, "comparison.json")
    with open(cpath, "w") as f:
        json.dump({
            "baseline_pass_rate": baseline_pass_rate,
            "retention_pass_rate": pass_rate,
            "delta": delta,
            "verdict": retention_verdict,
            "per_claim": retention_data["exp01_comparison"],
        }, f, indent=2)

    summary_path = os.path.join(OUTPUT_DIR, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Experiment 04: Retention Check\n{'='*50}\n\n")
        f.write(f"Model: {MODEL_ID}\n")
        f.write(f"Method: {METHOD} (re-tested after GRPO)\n")
        f.write(f"Date: {retention_data['timestamp']}\n\n")
        f.write(f"Baseline (exp01): {baseline_pass_rate:.1%}\n")
        f.write(f"After (exp04):    {pass_rate:.1%}\n")
        f.write(f"Delta:            {delta:+.1%}\n")
        f.write(f"Verdict:          {retention_verdict}\n\n")
        f.write(f"Per-claim:\n")
        for pc in retention_data["exp01_comparison"]:
            f.write(f"  [{pc['exp01_verdict']:>4} → {pc['exp04_verdict']:>4}] {pc['claim']}\n")

    print(f"\n{'='*70}")
    print(f"RETENTION CHECK COMPLETE")
    print(f"Baseline: {baseline_pass_rate:.1%} → After: {pass_rate:.1%} (delta: {delta:+.1%})")
    print(f"Verdict:  {retention_verdict}")
    print(f"{'='*70}")

    return retention_data

if __name__ == "__main__":
    main()
