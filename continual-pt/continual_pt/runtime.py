"""
The runtime: the main loop.

Ties together absorb → verify → train → retain → commit.

This is the entry point. Given X, the runtime:
1. Researches X, generates practice (absorb)
2. Snapshots current adapter (anchor for rollback)
3. Trains candidate on practice (commit.train_candidate)
4. Verifies X was learned (verify)
5. Checks retention on all prior items (retain.check_retention)
6. Commits or rejects (commit.commit_gate)

No agent persona. No fixed domain. X is chosen at call time.
"""

import json
import os
import time
from typing import List, Optional, Dict
from continual_pt import Config, VerifierType, Verdict
from continual_pt.model import load_model, load_model_with_adapter, generate, snapshot_adapter, restore_adapter
from continual_pt.absorb import research_x, generate_practice
from continual_pt.verify import verify
from continual_pt.retain import Ledger, check_retention
from continual_pt.commit import train_candidate, commit_gate


def learn_x(
    model,
    tokenizer,
    x: str,
    config: Config,
    ledger: Ledger,
    verifier_type: VerifierType = VerifierType.AUTO,
    claims: Optional[List[str]] = None,
    claim: Optional[str] = None,
) -> Dict:
    """
    The full loop for learning one X.

    Returns a dict with all intermediate results and the final commit decision.
    """
    print(f"\n{'='*70}")
    print(f"LEARNING: {x}")
    print(f"{'='*70}")
    print(f"Verifier: {verifier_type.value}")
    print(f"Ledger size: {len(ledger)}")

    t_start = time.time()
    log = {"x": x, "verifier_type": verifier_type.value, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}

    # --- 1. ABSORB ---
    print(f"\n--- Phase 1: ABSORB ---")
    t0 = time.time()
    research = research_x(model, tokenizer, x, config)
    practice = generate_practice(model, tokenizer, x, research, config)
    practice_source = "model"
    if practice and practice[0].get("_generation_source") == "fallback":
        practice_source = "fallback"
    log["absorb"] = {
        "time": time.time() - t0,
        "queries": research.get("queries", []),
        "pages_fetched": len(research.get("pages", [])),
        "practice_examples": len(practice),
        "practice_source": practice_source,  # "model" or "fallback"
    }

    # --- 2. SNAPSHOT (anchor for rollback) ---
    print(f"\n--- Phase 2: SNAPSHOT (anchor) ---")
    anchor = snapshot_adapter(model)

    # --- 3. TRAIN (candidate) ---
    print(f"\n--- Phase 3: TRAIN (candidate) ---")
    t0 = time.time()
    training_log = train_candidate(model, tokenizer, practice, config)
    training_log["time"] = time.time() - t0
    # CRITICAL FLAG: training was skipped if 0 examples
    training_log["training_skipped"] = (len(practice) == 0)
    if training_log["training_skipped"]:
        training_log["skip_reason"] = "zero practice examples generated (model + fallback both failed)"
        print(f"\n[runtime] ⚠⚠ CRITICAL: TRAINING SKIPPED — 0 practice examples. "
              f"This item tests the COLD model, NOT a trained one. "
              f"The commit decision below is NOT a valid continual-learning test.")
    elif practice_source == "fallback":
        print(f"\n[runtime] ⚠ NOTE: Training used FALLBACK examples (model generation failed). "
              f"Training ran, but on programmatically-extracted text, not model-generated practice.")
    log["train"] = training_log

    # --- 4. VERIFY (post-training) ---
    print(f"\n--- Phase 4: VERIFY ---")
    t0 = time.time()
    verify_result = verify(model, tokenizer, x, config,
                          verifier_type=verifier_type,
                          claims=claims, claim=claim)
    verify_result["time"] = time.time() - t0
    log["verify"] = verify_result

    # --- 5. RETAIN (behavioral) ---
    print(f"\n--- Phase 5: RETAIN (behavioral) ---")
    t0 = time.time()
    retention_result = check_retention(model, tokenizer, ledger, config)
    retention_result["time"] = time.time() - t0
    log["retain"] = retention_result

    # --- 6. COMMIT GATE ---
    print(f"\n--- Phase 6: COMMIT GATE ---")
    commit_result = commit_gate(
        model, tokenizer, x, verify_result, retention_result,
        training_log, anchor, config, ledger,
        verifier_type=verifier_type.value,
        claims=claims, claim=claim,
        output_dir=config.output_dir,
    )
    log["commit"] = commit_result
    log["total_time"] = time.time() - t_start

    # Save log
    os.makedirs(config.output_dir, exist_ok=True)
    log_path = os.path.join(config.output_dir, f"learn_{x[:50].replace(' ', '_')}_{int(time.time())}.json")
    with open(log_path, "w") as f:
        # Filter out large fields for the log
        log_save = {k: v for k, v in log.items()}
        # Don't save full implementations in the log (too large)
        if "verify" in log_save and "implementation" in log_save["verify"]:
            log_save["verify"]["implementation"] = "[saved separately]"
        if "verify" in log_save and "sub_claims" in log_save["verify"]:
            for sc in log_save["verify"]["sub_claims"]:
                if "reconstruction" in sc:
                    sc["reconstruction"] = sc["reconstruction"][:500] + "..."
                if "sources" in sc:
                    sc["sources"] = [{"url": s.get("url",""), "title": s.get("title","")} for s in sc["sources"]]
        json.dump(log_save, f, indent=2, default=str)
    print(f"\n[runtime] Log saved to {log_path}")

    return log


def learn_sequence(
    x_list: List[Dict],
    config: Config,
    adapter_path: Optional[str] = None,
) -> List[Dict]:
    """
    Run the loop for a sequence of X's.

    Each element in x_list is a dict:
    {
        "x": "LoRA",
        "verifier_type": "executable",  # or "non_executable" or "auto"
        "claims": ["claim1", "claim2", ...],  # for executable
        "claim": "single claim string",  # for non_executable
    }

    If adapter_path is provided, loads existing adapter (resume from prior commits).
    Otherwise starts fresh.
    """
    print(f"\n{'#'*70}")
    print(f"# CONTINUAL-PT: LEARNING SEQUENCE OF {len(x_list)} ITEMS")
    print(f"{'#'*70}")

    # Load model
    if adapter_path and os.path.exists(adapter_path):
        model, tokenizer = load_model_with_adapter(config, adapter_path)
    else:
        model, tokenizer = load_model(config)

    # Load ledger
    ledger_path = os.path.join(config.output_dir, "ledger.json")
    ledger = Ledger(ledger_path)
    print(f"[runtime] Ledger loaded: {len(ledger)} prior commits")

    # Run each X
    results = []
    for i, x_spec in enumerate(x_list):
        print(f"\n\n{'='*70}")
        print(f"= SEQUENCE ITEM {i+1}/{len(x_list)}")
        print(f"{'='*70}")

        x = x_spec["x"]
        vtype_str = x_spec.get("verifier_type", "auto")
        vtype = VerifierType(vtype_str)
        claims = x_spec.get("claims")
        claim = x_spec.get("claim")

        try:
            result = learn_x(model, tokenizer, x, config, ledger,
                           verifier_type=vtype, claims=claims, claim=claim)
            results.append(result)
        except Exception as e:
            import traceback
            print(f"[runtime] ERROR on {x}: {e}")
            traceback.print_exc()
            results.append({"x": x, "error": str(e), "traceback": traceback.format_exc()})

    # Final summary
    print(f"\n\n{'#'*70}")
    print(f"# SEQUENCE COMPLETE")
    print(f"{'#'*70}")

    # Check for critical failures (training skipped)
    critical_failures = []
    for r in results:
        train = r.get("train", {})
        if train.get("training_skipped"):
            critical_failures.append({
                "x": r.get("x"),
                "issue": "TRAINING SKIPPED — 0 practice examples",
                "reason": train.get("skip_reason", "unknown"),
                "impact": "This item tested the COLD model, NOT a trained one. "
                         "The commit decision is NOT a valid continual-learning test.",
            })

    print(f"\n{'X':<40} {'Committed':>10} {'Verdict':>15} {'Train':>10} {'Regress':>8}")
    print("-" * 85)
    for r in results:
        x = r.get("x", "?")[:38]
        committed = r.get("commit", {}).get("committed", False)
        verdict = r.get("commit", {}).get("verdict", "?")
        train_status = "SKIPPED⚠" if r.get("train", {}).get("training_skipped") else "ran"
        regressions = len(r.get("retain", {}).get("regressions", []))
        print(f"{x:<40} {'✓' if committed else '✗':>10} {verdict:>15} {train_status:>10} {regressions:>8}")

    # Loud critical failure banner
    if critical_failures:
        print(f"\n{'!'*85}")
        print(f"!! CRITICAL FAILURES — {len(critical_failures)} item(s) had training skipped:")
        for cf in critical_failures:
            print(f"!!   - {cf['x']}: {cf['issue']}")
            print(f"!!     Impact: {cf['impact']}")
        print(f"!! These items' commit decisions are NOT valid continual-learning tests.")
        print(f"!! The spec's retention question CANNOT be answered with these results.")
        print(f"{'!'*85}")

    # Save final report — include critical failures prominently
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": config.model_id,
        "sequence_length": len(x_list),
        "critical_failures": critical_failures,  # TOP of report — impossible to miss
        "critical_failure_count": len(critical_failures),
        "valid_continual_learning_tests": len(results) - len(critical_failures),
        "results": [
            {
                "x": r.get("x"),
                "committed": r.get("commit", {}).get("committed"),
                "verdict": r.get("commit", {}).get("verdict"),
                "training_skipped": r.get("train", {}).get("training_skipped", False),
                "practice_examples": r.get("absorb", {}).get("practice_examples", 0),
                "practice_source": r.get("absorb", {}).get("practice_source", "unknown"),
                "regressions": len(r.get("retain", {}).get("regressions", [])),
                "total_time": r.get("total_time"),
            }
            for r in results
        ],
        "ledger_size": len(ledger),
        "spec_open_question_answered": (
            "NO — zero commits, retention never fired against a real prior item"
            if len(ledger) == 0
            else f"PARTIALLY — {len(ledger)} commit(s), retention fired {len(ledger)} time(s)"
        ),
    }
    report_path = os.path.join(config.output_dir, "final_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[runtime] Final report saved to {report_path}")

    return results
