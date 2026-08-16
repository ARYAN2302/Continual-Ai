"""
The retain phase + the commit gate + the ledger.

The ledger records every committed item with its verifier type and parameters,
so it can be re-run for retention checks.

Retention is BEHAVIORAL: re-run the original verifier on each committed item,
live, right now. Not PPL. Not any proxy. The spec is explicit:

> perplexity recovering while the actual capability stays broken is a
> documented failure mode from earlier work, and it's the one mistake this
> design is built to avoid repeating.

Repair is deliberately NOT implemented. If regression is detected, we reject
the commit and rollback. Repair will be designed against a real regression
from a real run, not before.
"""

import json
import os
import time
import copy
from typing import List, Dict, Optional
from continual_pt import Verdict, VerifierType


class Ledger:
    """Records committed items with their verifier parameters."""

    def __init__(self, path: str):
        self.path = path
        self.items: List[Dict] = []
        if os.path.exists(path):
            with open(path) as f:
                self.items = json.load(f)

    def add(self, item: Dict):
        """Add a committed item to the ledger."""
        item["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self.items.append(item)
        self.save()

    def save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self.items, f, indent=2)

    def __len__(self):
        return len(self.items)

    def __iter__(self):
        return iter(self.items)


def check_retention(model, tokenizer, ledger: Ledger, config) -> Dict:
    """
    Behavioral retention check.

    Re-runs the original verifier on EVERY previously committed item.
    Returns which items regressed (verdict changed from PASS/RECONSTRUCTED
    to FAIL/DISCARD/CONFUSED/OPEN).

    This is the critical check. PPL is not used. Only behavioral verification.
    """
    from continual_pt.verify import verify_executable, verify_non_executable

    print(f"\n[retain] Checking retention on {len(ledger)} committed items...")

    if len(ledger) == 0:
        print("[retain] No prior items. Retention check passes trivially.")
        return {
            "checked": 0,
            "regressions": [],
            "all_retained": True,
        }

    regressions = []
    checked = 0

    for i, item in enumerate(ledger.items):
        x = item["x"]
        verifier_type = item.get("verifier_type", "executable")
        original_verdict = item.get("verdict", "PASS")

        print(f"\n  [{i+1}/{len(ledger)}] Re-checking: {x[:60]}...")
        print(f"    Original verdict: {original_verdict}")

        # Re-run the original verifier
        if verifier_type == "executable":
            claims = item.get("claims", [])
            result = verify_executable(model, tokenizer, x, claims, config)
            current_verdict = "PASS" if result["all_pass"] else "FAIL"
        else:
            # Non-executable: re-run the 5-step process
            claim = item.get("claim", x)
            result = verify_non_executable(model, tokenizer, x, claim, config)
            current_verdict = result["overall_verdict"]

        print(f"    Current verdict:  {current_verdict}")

        # Check for regression
        # Regression = verdict got worse
        # PASS/RECONSTRUCTED → anything else = regression
        # OPEN → DISCARD = regression (got worse)
        # OPEN → OPEN = not regression (held open)
        # OPEN → RECONSTRUCTED = improvement (not regression)
        worse = False
        if original_verdict in ("PASS", "RECONSTRUCTED"):
            if current_verdict not in ("PASS", "RECONSTRUCTED"):
                worse = True
        elif original_verdict == "OPEN":
            if current_verdict == "DISCARD":
                worse = True

        if worse:
            print(f"    ⚠ REGRESSION DETECTED")
            regressions.append({
                "x": x,
                "original_verdict": original_verdict,
                "current_verdict": current_verdict,
                "details": result,
            })

        checked += 1

    all_retained = len(regressions) == 0
    print(f"\n[retain] Checked {checked} items. Regressions: {len(regressions)}")
    print(f"[retain] All retained: {all_retained}")

    return {
        "checked": checked,
        "regressions": regressions,
        "all_retained": all_retained,
    }
