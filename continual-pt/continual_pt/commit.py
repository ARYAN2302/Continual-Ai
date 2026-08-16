"""
The commit phase.

Trains a candidate DoRA update on verified practice examples.
The commit gate checks:
  1. Did X pass verification? (post-training)
  2. Did all prior items pass retention? (behavioral re-check)

If both pass → commit (save adapter, record in ledger).
If either fails → rollback to anchor.

Repair is deliberately not implemented. A real regression must occur first
before we know what to design repair against.
"""

import torch
import json
import os
import time
from typing import List, Dict
from continual_pt.model import snapshot_adapter, restore_adapter, save_adapter


def train_candidate(model, tokenizer, practice_examples: List[Dict], config) -> Dict:
    """
    SFT training on practice examples.

    Each example: {question, answer, source_url, source_span}
    Trains the DoRA adapter (fast weights) on (question → answer) pairs.
    """
    print(f"\n[commit] Training candidate on {len(practice_examples)} examples...")

    if len(practice_examples) == 0:
        print("[commit] No practice examples. Skipping training.")
        return {"losses": [], "avg_loss": None, "final_loss": None}

    model.train()

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.learning_rate,
        weight_decay=0.01,
    )

    all_losses = []

    for epoch in range(config.num_epochs):
        epoch_losses = []

        for step, ex in enumerate(practice_examples):
            prompt = ex["question"]
            response = ex["answer"]

            # Construct full text
            full_text = prompt + "\n" + response
            inputs = tokenizer(full_text, return_tensors="pt", truncation=True,
                             max_length=config.max_seq_len).to(model.device)

            # Mask the prompt (train only on response)
            prompt_inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                                    max_length=config.max_seq_len)
            labels = inputs.input_ids.clone()
            prompt_len = prompt_inputs.input_ids.shape[1]
            if prompt_len < labels.shape[1]:
                labels[:, :prompt_len] = -100

            try:
                outputs = model(input_ids=inputs.input_ids, labels=labels)
                loss = outputs.loss / config.grad_accum_steps
                loss.backward()
                epoch_losses.append(loss.item() * config.grad_accum_steps)

                if (step + 1) % config.grad_accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad], 1.0
                    )
                    optimizer.step()
                    optimizer.zero_grad()
            except Exception as e:
                print(f"  [commit] Training step failed: {e}")
                optimizer.zero_grad()
                continue

        avg_loss = sum(epoch_losses) / max(len(epoch_losses), 1)
        all_losses.extend(epoch_losses)
        print(f"  Epoch {epoch+1}/{config.num_epochs}: avg loss = {avg_loss:.4f}")

    model.eval()

    return {
        "losses": all_losses,
        "avg_loss": sum(all_losses) / len(all_losses) if all_losses else None,
        "final_loss": all_losses[-1] if all_losses else None,
        "num_examples": len(practice_examples),
        "epochs": config.num_epochs,
    }


def commit_gate(
    model,
    tokenizer,
    x: str,
    verify_result: Dict,
    retention_result: Dict,
    training_log: Dict,
    anchor: Dict,
    config,
    ledger,
    verifier_type: str,
    claims: List[str] = None,
    claim: str = None,
    output_dir: str = "./results",
) -> Dict:
    """
    The commit gate.

    Commits only if:
      1. X passes verification (all_pass=True or overall_verdict=RECONSTRUCTED)
      2. All prior items pass retention (all_retained=True)

    If either fails, rollback to anchor.
    """

    # Check verification
    if verifier_type == "executable":
        x_verified = verify_result.get("all_pass", False)
        verdict_str = "PASS" if x_verified else "FAIL"
    else:
        x_verified = verify_result.get("pass", False)
        verdict_str = verify_result.get("overall_verdict", "OPEN")

    # Check retention
    retention_ok = retention_result.get("all_retained", True)

    # Decision
    should_commit = x_verified and retention_ok

    print(f"\n[commit] === COMMIT GATE ===")
    print(f"  X verified:     {x_verified} ({verdict_str})")
    print(f"  Retention OK:   {retention_ok} ({retention_result.get('checked', 0)} checked, {len(retention_result.get('regressions', []))} regressions)")
    print(f"  Decision:       {'COMMIT' if should_commit else 'REJECT'}")

    if should_commit:
        # Save adapter
        adapter_path = os.path.join(output_dir, "adapters", f"adapter_v{len(ledger)+1}")
        save_adapter(model, tokenizer, adapter_path)

        # Record in ledger
        ledger_item = {
            "x": x,
            "verifier_type": verifier_type,
            "verdict": verdict_str,
            "claims": claims,
            "claim": claim,
            "training_log": {
                "num_examples": training_log.get("num_examples"),
                "final_loss": training_log.get("final_loss"),
                "avg_loss": training_log.get("avg_loss"),
            },
            "adapter_path": adapter_path,
        }
        ledger.add(ledger_item)

        print(f"[commit] ✓ Committed. Ledger now has {len(ledger)} items.")
    else:
        # Rollback
        print(f"[commit] ✗ Rejected. Rolling back to anchor.")
        restore_adapter(model, anchor)
        print(f"[commit] Rolled back.")

    return {
        "committed": should_commit,
        "x_verified": x_verified,
        "retention_ok": retention_ok,
        "verdict": verdict_str,
        "regressions": retention_result.get("regressions", []),
    }
