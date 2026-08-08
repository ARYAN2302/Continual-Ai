"""
=============================================================================
Master Runner: All 4 Experiments + Final Report
=============================================================================

Runs the full experiment pipeline:
  exp01: Cold start — implement LoRA, verify 5 claims (BEFORE)
  exp02: Train DoRA adapter on verified data (avr-cl gated)
  exp03: After eval — implement GRPO, verify 5 claims (AFTER, with/without adapter)
  exp04: Retention — re-verify LoRA with adapter (did we forget?)

Produces a final report comparing before vs after, with retention check.

Run on Kaggle T4x2.
Usage:
    python run_all.py

Output:
    {output_dir}/final_report.json — complete results
    {output_dir}/final_report.txt  — human-readable summary
=============================================================================
"""

import subprocess
import sys
import os
import json
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

def run_experiment(script_name, description):
    """Run one experiment script and capture its output."""
    print(f"\n{'#'*70}")
    print(f"# RUNNING: {description}")
    print(f"{'#'*70}\n")

    script_path = os.path.join(SCRIPT_DIR, script_name)
    if not os.path.exists(script_path):
        print(f"ERROR: {script_path} not found")
        return None

    t0 = time.time()
    result = subprocess.run(
        [sys.executable, script_path],
        capture_output=False,  # Let output stream to console
    )
    elapsed = time.time() - t0

    print(f"\n  {script_name} completed in {elapsed:.1f}s (exit code: {result.returncode})")
    return result.returncode == 0

def load_json(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None

def main():
    print("=" * 70)
    print("MASTER RUNNER: Full Experiment Pipeline")
    print("=" * 70)
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    pipeline_start = time.time()

    # === Experiment 01: Cold Start ===
    success = run_experiment("exp01_before_eval.py",
        "Exp01: Cold Start — Implement LoRA, verify 5 claims (BEFORE baseline)")
    if not success:
        print("exp01 failed. Aborting.")
        return

    # === Experiment 02: Train ===
    success = run_experiment("exp02_train.py",
        "Exp02: Train DoRA adapter on verified data (avr-cl gated)")
    if not success:
        print("exp02 failed. Aborting.")
        return

    # === Experiment 03: After Eval ===
    success = run_experiment("exp03_after_eval.py",
        "Exp03: After Eval — Implement GRPO, compare cold vs adapter")
    if not success:
        print("exp03 failed. Aborting.")
        return

    # === Experiment 04: Retention ===
    success = run_experiment("exp04_retention.py",
        "Exp04: Retention — Re-verify LoRA with adapter")
    if not success:
        print("exp04 failed. Continuing to final report.")

    # === Final Report ===
    print("\n\n" + "=" * 70)
    print("FINAL REPORT")
    print("=" * 70)

    exp01 = load_json(os.path.join(PROJECT_ROOT, "download", "exp_results", "exp01", "results.json"))
    exp02 = load_json(os.path.join(PROJECT_ROOT, "download", "exp_results", "exp02", "training_log.json"))
    exp03 = load_json(os.path.join(PROJECT_ROOT, "download", "exp_results", "exp03", "comparison.json"))
    exp04 = load_json(os.path.join(PROJECT_ROOT, "download", "exp_results", "exp04", "retention_results.json"))

    print(f"\n{'Experiment':<30} {'Result':>40}")
    print("-" * 70)

    if exp01:
        print(f"{'Exp01: LoRA cold pass rate':<30} {exp01['pass_rate']:>39.1%}")
    if exp02:
        avr = exp02.get("avr_cl", {})
        print(f"{'Exp02: Adapter committed':<30} {str(avr.get('committed', '?')):>40}")
        print(f"{'Exp02: Drift ratio':<30} {avr.get('avg_drift_ratio', 0):>39.3f}")
        print(f"{'Exp02: Repair applied':<30} {str(avr.get('repair', {}).get('repaired', False)):>40}")
    if exp03:
        print(f"{'Exp03: GRPO cold pass rate':<30} {exp03['cold']['pass_rate']:>39.1%}")
        print(f"{'Exp03: GRPO adapter pass rate':<30} {exp03['adapter']['pass_rate']:>39.1%}")
        print(f"{'Exp03: Delta (adapter - cold)':<30} {exp03['delta']['pass_rate']:>+39.1%}")
    if exp04:
        print(f"{'Exp04: LoRA retention pass rate':<30} {exp04['retention_pass_rate']:>39.1%}")
        print(f"{'Exp04: Delta (after - before)':<30} {exp04['delta']:>+39.1%}")
        print(f"{'Exp04: Verdict':<30} {exp04['retention_verdict']:>40}")

    # === The Story ===
    print(f"\n{'='*70}")
    print("THE STORY")
    print(f"{'='*70}")

    if exp01 and exp03 and exp04:
        print(f"""
1. COLD START: LFM2.5-2.6B was asked to implement LoRA from scratch.
   It passed {exp01['pass']}/{exp01['num_claims']} claims ({exp01['pass_rate']:.1%}).

2. TRAINING: A DoRA adapter was trained on verified LoRA implementations.
   avr-cl gate {'committed' if exp02.get('avr_cl',{}).get('committed') else 'REJECTED'} the update.
   {'Repair was applied.' if exp02.get('avr_cl',{}).get('repair',{}).get('repaired') else 'No repair needed.'}

3. AFTER (GRPO): The model was asked to implement GRPO (a method it hadn't seen).
   Cold:     {exp03['cold']['pass']}/{exp03['cold']['num_claims']} ({exp03['cold']['pass_rate']:.1%})
   Adapter:  {exp03['adapter']['pass']}/{exp03['adapter']['num_claims']} ({exp03['adapter']['pass_rate']:.1%})
   Delta:    {exp03['delta']['pass_rate']:+.1%}

4. RETENTION: The model was re-tested on LoRA (did it forget?).
   Before:   {exp04['baseline_pass_rate']:.1%}
   After:    {exp04['retention_pass_rate']:.1%}
   Verdict:  {exp04['retention_verdict']}
""")

    # Save final report
    final_report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_time_sec": time.time() - pipeline_start,
        "exp01": {
            "pass_rate": exp01["pass_rate"] if exp01 else None,
            "pass": exp01["pass"] if exp01 else None,
            "num_claims": exp01["num_claims"] if exp01 else None,
        } if exp01 else None,
        "exp02": {
            "committed": exp02.get("avr_cl", {}).get("committed") if exp02 else None,
            "drift_ratio": exp02.get("avr_cl", {}).get("avg_drift_ratio") if exp02 else None,
            "repaired": exp02.get("avr_cl", {}).get("repair", {}).get("repaired") if exp02 else None,
        } if exp02 else None,
        "exp03": {
            "cold_pass_rate": exp03["cold"]["pass_rate"] if exp03 else None,
            "adapter_pass_rate": exp03["adapter"]["pass_rate"] if exp03 else None,
            "delta": exp03["delta"]["pass_rate"] if exp03 else None,
        } if exp03 else None,
        "exp04": {
            "baseline_pass_rate": exp04["baseline_pass_rate"] if exp04 else None,
            "retention_pass_rate": exp04["retention_pass_rate"] if exp04 else None,
            "delta": exp04["delta"] if exp04 else None,
            "verdict": exp04["retention_verdict"] if exp04 else None,
        } if exp04 else None,
    }

    output_dir = os.path.join(PROJECT_ROOT, "download", "exp_results")
    os.makedirs(output_dir, exist_ok=True)
    final_path = os.path.join(output_dir, "final_report.json")
    with open(final_path, "w") as f:
        json.dump(final_report, f, indent=2)
    print(f"\nFinal report saved to {final_path}")

    print(f"\nTotal pipeline time: {time.time()-pipeline_start:.1f}s")
    print(f"\n{'='*70}")
    print("PIPELINE COMPLETE")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
