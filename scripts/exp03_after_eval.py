"""
=============================================================================
Experiment 03: After Eval — Test on GRPO with Trained Adapter
=============================================================================

Loads the trained DoRA adapter from exp02. Asks the model to implement GRPO
(a DIFFERENT method it hasn't seen). Verifies claims. Compares to a cold
baseline (same model, no adapter).

This is the "after" half of the before/after comparison. The hypothesis:
  - If the adapter taught the model "how to implement ML methods properly"
    (not just memorized LoRA), it should generalize to GRPO.
  - Pass rate on GRPO should be higher WITH adapter than WITHOUT.

Run on Kaggle T4x2.
Self-contained: loads adapter, runs cold + adapter conditions, saves comparison.

Usage:
    python exp03_after_eval.py

Inputs:
    {exp02_dir}/adapter/  — trained DoRA adapter from exp02

Outputs:
    {output_dir}/cold_results.json     — GRPO results without adapter
    {output_dir}/adapter_results.json  — GRPO results with adapter
    {output_dir}/comparison.json       — side-by-side comparison
    {output_dir}/summary.txt           — human-readable summary
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
EXP02_DIR = os.environ.get("EXP02_DIR",
    os.path.join(PROJECT_ROOT, "download", "exp_results", "exp02")
)
EXP02_DIR = os.path.abspath(EXP02_DIR)

OUTPUT_DIR = os.environ.get("EXP_OUTPUT_DIR",
    "/kaggle/working/exp03" if os.path.exists("/kaggle") else os.path.join(PROJECT_ROOT, "download", "exp_results", "exp03")
)
OUTPUT_DIR = os.path.abspath(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "tests"), exist_ok=True)

METHOD = "GRPO (Group Relative Policy Optimization)"

# GRPO claims — different from LoRA, tests generalization
CLAIMS = [
    "GRPO computes the advantage as (reward - mean_reward) / std_reward for each response in the group, where mean and std are computed over the group's rewards.",
    "GRPO does not use a separate value/critic model — it uses the group mean as the baseline.",
    "GRPO samples a GROUP of responses for the same prompt (e.g., G=4 or G=8 responses), not just one.",
    "GRPO applies a KL penalty between the current policy and a reference policy to prevent the policy from drifting too far.",
    "GRPO's loss is similar to PPO but replaces the value-function baseline with the group-relative baseline.",
]

MAX_NEW_TOKENS_IMPL = 2048
MAX_NEW_TOKENS_TEST = 1024
VERIFICATION_TIMEOUT = 30
TEMPERATURE = 0.7
N_SAMPLES_PER_CLAIM = 1  # Set >1 for multi-sample verification

# === Model Loading ===
def load_model(with_adapter=False, adapter_path=None):
    """Load LFM2.5-2.6B, optionally with trained adapter."""
    print(f"Loading {MODEL_ID} in 4-bit NF4 (adapter={with_adapter})...")
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
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.eval()

    if with_adapter and adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
        print(f"  Adapter loaded from {adapter_path}")
        model.eval()

    return model, tokenizer

# === Generation (same as exp01) ===
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
            temperature=temperature, top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
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

# === Prompts ===
IMPL_PROMPT = """You are an expert ML engineer. Implement {method} in PyTorch.

Write a COMPLETE, RUNNABLE Python file that:
1. Defines a GRPOTrainer class or function that implements Group Relative Policy Optimization
2. The implementation must include: group sampling, advantage computation, policy loss, KL penalty
3. Include a __main__ block that creates a dummy setup and runs one step
4. Print "IMPL_READY" at the end if no errors

Requirements:
- Use torch
- Show the advantage computation explicitly: advantage = (reward - mean) / (std + eps)
- Show the group sampling: sample G responses per prompt
- Include the KL divergence penalty
- Do NOT use the trl library — implement from scratch

Write ONLY Python code. No explanation.
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
    print("  Generating implementation...")
    response = generate(model, tokenizer, IMPL_PROMPT.format(method=method),
                        max_new_tokens=MAX_NEW_TOKENS_IMPL, temperature=TEMPERATURE)
    return extract_code(response)

def generate_test(model, tokenizer, method, claim, implementation):
    response = generate(model, tokenizer, TEST_PROMPT.format(
        method=method, claim=claim, implementation=implementation
    ), max_new_tokens=MAX_NEW_TOKENS_TEST, temperature=TEMPERATURE)
    return extract_code(response)

# === Verifier (same as exp01) ===
def verify_claim(implementation_code, test_code, timeout=30):
    full_code = implementation_code + "\n\n# === TEST ===\n" + test_code
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(full_code)
        temp_path = f.name
    try:
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
        result = subprocess.run([sys.executable, temp_path], capture_output=True,
                               text=True, timeout=timeout, env=env)
        stdout = result.stdout
        stderr = result.stderr
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

# === Run One Condition ===
def run_condition(model, tokenizer, method, claims, condition_name, output_dir):
    """Run the full verify loop for one condition (cold or adapter)."""
    print(f"\n{'='*50}")
    print(f"Running condition: {condition_name}")
    print(f"{'='*50}")

    # Generate implementation
    print("\n  Generating implementation...")
    t0 = time.time()
    impl = generate_implementation(model, tokenizer, method)
    print(f"  Generated in {time.time()-t0:.1f}s, {len(impl)} chars")

    impl_path = os.path.join(output_dir, f"{condition_name}_implementation.py")
    with open(impl_path, "w") as f:
        f.write(impl)

    # Verify each claim
    results = []
    for i, claim in enumerate(claims):
        print(f"\n  [{i+1}/{len(claims)}] {claim[:80]}...")
        t0 = time.time()
        test = generate_test(model, tokenizer, method, claim, impl)
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
    pass_rate = pass_count / len(claims) if claims else 0

    return {
        "condition": condition_name,
        "method": method,
        "num_claims": len(claims),
        "pass": pass_count,
        "fail": fail_count,
        "confused": confused_count,
        "pass_rate": pass_rate,
        "implementation": impl,
        "results": results,
    }

# === Main ===
def main():
    print("=" * 70)
    print("EXPERIMENT 03: AFTER EVAL — GRPO with/without adapter")
    print("=" * 70)
    print(f"Model: {MODEL_ID}")
    print(f"Method: {METHOD}")
    print(f"Claims: {len(CLAIMS)}")
    print()

    adapter_path = os.path.join(EXP02_DIR, "adapter")
    if not os.path.exists(adapter_path):
        print(f"ERROR: Adapter not found at {adapter_path}")
        print("Run exp02_train.py first.")
        return

    # === Condition 1: Cold (no adapter) ===
    print("\n### CONDITION 1: COLD (no adapter) ###")
    model_cold, tokenizer = load_model(with_adapter=False)
    cold_results = run_condition(model_cold, tokenizer, METHOD, CLAIMS, "cold", OUTPUT_DIR)

    cold_path = os.path.join(OUTPUT_DIR, "cold_results.json")
    with open(cold_path, "w") as f:
        json.dump(cold_results, f, indent=2)

    # Free memory
    del model_cold
    torch.cuda.empty_cache()
    import gc; gc.collect()

    # === Condition 2: With adapter ===
    print("\n\n### CONDITION 2: WITH ADAPTER ###")
    model_adapter, tokenizer = load_model(with_adapter=True, adapter_path=adapter_path)
    adapter_results = run_condition(model_adapter, tokenizer, METHOD, CLAIMS, "adapter", OUTPUT_DIR)

    adapter_path_results = os.path.join(OUTPUT_DIR, "adapter_results.json")
    with open(adapter_path_results, "w") as f:
        json.dump(adapter_results, f, indent=2)

    # === Comparison ===
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    print(f"\n{'Metric':<25} {'Cold':>10} {'Adapter':>10} {'Delta':>10}")
    print("-" * 55)
    print(f"{'Pass rate':<25} {cold_results['pass_rate']:>9.1%} {adapter_results['pass_rate']:>9.1%} {adapter_results['pass_rate']-cold_results['pass_rate']:>+9.1%}")
    print(f"{'Pass count':<25} {cold_results['pass']:>10} {adapter_results['pass']:>10} {adapter_results['pass']-cold_results['pass']:>+10}")
    print(f"{'Fail count':<25} {cold_results['fail']:>10} {adapter_results['fail']:>10} {adapter_results['fail']-cold_results['fail']:>+10}")
    print(f"{'Confused count':<25} {cold_results['confused']:>10} {adapter_results['confused']:>10} {adapter_results['confused']-cold_results['confused']:>+10}")

    print(f"\nPer-claim comparison:")
    print(f"{'#':<3} {'Cold':>8} {'Adapter':>8} {'Claim':<60}")
    print("-" * 80)
    for i in range(len(CLAIMS)):
        c = cold_results["results"][i]["verdict"] if i < len(cold_results["results"]) else "?"
        a = adapter_results["results"][i]["verdict"] if i < len(adapter_results["results"]) else "?"
        claim_short = CLAIMS[i][:60]
        print(f"{i+1:<3} {c:>8} {a:>8} {claim_short}")

    # Save comparison
    comparison = {
        "experiment": "exp03_after_eval",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": MODEL_ID,
        "method": METHOD,
        "cold": {
            "pass_rate": cold_results["pass_rate"],
            "pass": cold_results["pass"],
            "fail": cold_results["fail"],
            "confused": cold_results["confused"],
        },
        "adapter": {
            "pass_rate": adapter_results["pass_rate"],
            "pass": adapter_results["pass"],
            "fail": adapter_results["fail"],
            "confused": adapter_results["confused"],
        },
        "delta": {
            "pass_rate": adapter_results["pass_rate"] - cold_results["pass_rate"],
            "pass": adapter_results["pass"] - cold_results["pass"],
        },
        "per_claim": [
            {
                "claim": CLAIMS[i],
                "cold": cold_results["results"][i]["verdict"] if i < len(cold_results["results"]) else "?",
                "adapter": adapter_results["results"][i]["verdict"] if i < len(adapter_results["results"]) else "?",
                "cold_reason": cold_results["results"][i].get("reason","") if i < len(cold_results["results"]) else "",
                "adapter_reason": adapter_results["results"][i].get("reason","") if i < len(adapter_results["results"]) else "",
            }
            for i in range(len(CLAIMS))
        ],
    }

    comp_path = os.path.join(OUTPUT_DIR, "comparison.json")
    with open(comp_path, "w") as f:
        json.dump(comparison, f, indent=2)

    # Summary
    summary_path = os.path.join(OUTPUT_DIR, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Experiment 03: After Eval (GRPO)\n{'='*50}\n\n")
        f.write(f"Model: {MODEL_ID}\n")
        f.write(f"Method: {METHOD}\n")
        f.write(f"Date: {comparison['timestamp']}\n\n")
        f.write(f"{'Metric':<20} {'Cold':>10} {'Adapter':>10} {'Delta':>10}\n")
        f.write(f"{'-'*50}\n")
        f.write(f"{'Pass rate':<20} {cold_results['pass_rate']:>9.1%} {adapter_results['pass_rate']:>9.1%} {adapter_results['pass_rate']-cold_results['pass_rate']:>+9.1%}\n")
        f.write(f"{'Pass':<20} {cold_results['pass']:>10} {adapter_results['pass']:>10} {adapter_results['pass']-cold_results['pass']:>+10}\n")
        f.write(f"{'Fail':<20} {cold_results['fail']:>10} {adapter_results['fail']:>10} {adapter_results['fail']-cold_results['fail']:>+10}\n")
        f.write(f"{'Confused':<20} {cold_results['confused']:>10} {adapter_results['confused']:>10} {adapter_results['confused']-cold_results['confused']:>+10}\n\n")
        f.write(f"Per-claim:\n")
        for i, pc in enumerate(comparison["per_claim"]):
            f.write(f"  [{pc['cold']:>4} → {pc['adapter']:>4}] {pc['claim']}\n")

    print(f"\n{'='*70}")
    print(f"AFTER EVAL COMPLETE")
    print(f"Cold pass rate:     {cold_results['pass_rate']:.1%}")
    print(f"Adapter pass rate:  {adapter_results['pass_rate']:.1%}")
    print(f"Delta:              {adapter_results['pass_rate']-cold_results['pass_rate']:+.1%}")
    print(f"{'='*70}")

    return comparison

if __name__ == "__main__":
    main()
