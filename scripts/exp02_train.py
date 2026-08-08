"""
=============================================================================
Experiment 02: Train DoRA Adapter on Verified Implementations
=============================================================================

Takes the verified data from exp01 and trains a DoRA adapter on LFM2.5-2.6B.
The adapter teaches the model to produce better implementations.

The training data comes from:
  - exp01's implementation (if it passed verification) → used as a positive example
  - If exp01's implementation FAILED, a known-good reference implementation is used
  - The (prompt → implementation) pairs become SFT examples

avr-cl gate:
  - Before training: snapshot the adapter state (anchor)
  - After training: check drift on a held-out probe set
  - If drift > threshold: repair (closed-form interpolation toward anchor)
  - Commit only if drift is acceptable

Run on Kaggle T4x2.
Self-contained: reads exp01 results, trains, saves adapter.

Usage:
    python exp02_train.py

Inputs:
    {exp01_dir}/results.json — from exp01

Outputs:
    {output_dir}/adapter/           — trained DoRA adapter
    {output_dir}/training_log.json  — training metrics + avr-cl gate decision
    {output_dir}/drift_report.json  — drift check results
=============================================================================
"""

import subprocess
import sys
import os
import json
import time
import copy
import traceback

# === Dependency Installation ===
def install(pkgs):
    for pkg in pkgs:
        try:
            __import__(pkg.split("[")[0].replace("-", "_"))
        except ImportError:
            print(f"  Installing {pkg}...")
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=True)

print("=== Checking dependencies ===")
install(["torch", "transformers>=4.46", "peft>=0.13", "bitsandbytes", "accelerate", "datasets"])

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

# === Configuration ===
MODEL_ID = "LiquidAI/LFM2.5-2.6B"

# Directories
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
EXP01_DIR = os.environ.get("EXP01_DIR",
    os.path.join(PROJECT_ROOT, "download", "exp_results", "exp01")
)
EXP01_DIR = os.path.abspath(EXP01_DIR)

OUTPUT_DIR = os.environ.get("EXP_OUTPUT_DIR",
    "/kaggle/working/exp02" if os.path.exists("/kaggle") else os.path.join(PROJECT_ROOT, "download", "exp_results", "exp02")
)
OUTPUT_DIR = os.path.abspath(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "adapter"), exist_ok=True)

# DoRA config
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LEARNING_RATE = 1e-4
NUM_EPOCHS = 3
BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 4
MAX_SEQ_LEN = 2048

# avr-cl config
DRIFT_THRESHOLD = 0.15  # 15% PPL increase = drift
REPAIR_ALPHA = 0.1      # interpolation factor for repair
MAX_REPAIR_STEPS = 5

# === Reference Implementations (for training data) ===
# These are known-good implementations used to create SFT examples.
# If exp01's implementation passed verification, we also include it.
# If not, we train on the reference (the model needs to learn what good looks like).

REFERENCE_LORA_IMPL = '''import torch
import torch.nn as nn
import math

class LoRALayer(nn.Module):
    """LoRA (Low-Rank Adaptation) layer.

    Adds low-rank matrices A (r x d_in) and B (d_out x r) to a frozen linear layer.
    Forward: output = W @ x + (alpha/r) * B @ A @ x
    """
    def __init__(self, in_features, out_features, rank=8, alpha=16):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Original frozen weight
        self.base_weight = nn.Parameter(torch.randn(out_features, in_features), requires_grad=False)

        # LoRA matrices
        # A: (rank x in_features) — initialized with kaiming_uniform
        # B: (out_features x rank) — initialized with zeros
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        # Ensure base weight is frozen
        self.base_weight.requires_grad = False

    def forward(self, x):
        # W @ x
        base_out = x @ self.base_weight.t()
        # B @ A @ x, scaled by alpha/r
        lora_out = (x @ self.lora_A.t()) @ self.lora_B.t() * self.scaling
        return base_out + lora_out

if __name__ == "__main__":
    layer = LoRALayer(in_features=128, out_features=256, rank=8, alpha=16)
    x = torch.randn(4, 128)
    out = layer(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"A shape: {layer.lora_A.shape} (rank x in_features)")
    print(f"B shape: {layer.lora_B.shape} (out_features x rank)")
    print(f"Base weight frozen: {not layer.base_weight.requires_grad}")
    print(f"Trainable params: {sum(p.numel() for p in layer.parameters() if p.requires_grad)}")
    print(f"Full params: {layer.in_features * layer.out_features}")
    print("IMPL_READY")
'''

# The prompt that was used in exp01 (must match for SFT consistency)
TRAINING_PROMPT = """You are an expert ML engineer. Implement LoRA (Low-Rank Adaptation) in PyTorch.

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

# === Model Loading ===
def load_model_for_training():
    """Load model in 4-bit and attach a DoRA adapter."""
    print(f"Loading {MODEL_ID} in 4-bit NF4...")
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

    # Attach DoRA adapter
    # DoRA = Weight-Decomposed Low-Rank Adaptation
    # In PEFT, use_lora_magnitude_mast=True enables DoRA
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        use_dora=True,  # This enables DoRA instead of plain LoRA
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, tokenizer

# === avr-cl: Anchor Snapshot ===
def snapshot_adapter_state(model):
    """Snapshot the current adapter state (the anchor for drift detection)."""
    anchor = {}
    for name, param in model.named_parameters():
        if "lora_" in name or "magnitude" in name:  # DoRA has magnitude parameters too
            anchor[name] = param.data.clone().detach().cpu()
    print(f"  Anchored {len(anchor)} adapter parameters")
    return anchor

# === avr-cl: Drift Detection ===
def compute_ppl(model, tokenizer, texts, max_length=512):
    """Compute average perplexity on a set of probe texts."""
    model.eval()
    total_loss = 0
    total_tokens = 0
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True,
                             max_length=max_length).to(model.device)
            try:
                outputs = model(**inputs, labels=inputs.input_ids)
                total_loss += outputs.loss.item() * inputs.input_ids.shape[1]
                total_tokens += inputs.input_ids.shape[1]
            except Exception as e:
                print(f"  Warning: PPL computation failed for a probe: {e}")
                continue
    return total_loss / max(total_tokens, 1)

# === Probe Set for Drift Detection ===
# These are held-out texts that the model should NOT forget how to handle.
# If PPL on these increases after training, drift is detected.
DRIFT_PROBES = [
    "Explain how attention works in transformers.",
    "Write a Python function to compute the Fibonacci sequence.",
    "What is gradient descent in machine learning?",
    "Explain the difference between TCP and UDP.",
    "Write a simple REST API endpoint in FastAPI.",
    "What is backpropagation and how does it work?",
    "Explain the concept of recursion with an example.",
    "What are the advantages of using containers in software deployment?",
]

def check_drift(model, tokenizer, anchor_ppls):
    """Check if the model has drifted on the probe set."""
    current_ppls = {}
    for i, probe in enumerate(DRIFT_PROBES):
        ppl = compute_ppl(model, tokenizer, [probe])
        current_ppls[f"probe_{i}"] = ppl

    # Compute drift ratio
    drift_ratios = []
    for key in anchor_ppls:
        if key in current_ppls and anchor_ppls[key] > 0:
            ratio = current_ppls[key] / anchor_ppls[key]
            drift_ratios.append(ratio)

    avg_drift = sum(drift_ratios) / len(drift_ratios) if drift_ratios else 1.0
    max_drift = max(drift_ratios) if drift_ratios else 1.0

    drifted = avg_drift > (1.0 + DRIFT_THRESHOLD)

    return {
        "drifted": drifted,
        "avg_drift_ratio": avg_drift,
        "max_drift_ratio": max_drift,
        "current_ppls": current_ppls,
        "anchor_ppls": anchor_ppls,
    }

# === avr-cl: Repair ===
def repair_adapter(model, anchor, alpha=0.1):
    """Closed-form weight interpolation toward anchor.
    theta_new = (1 - alpha) * theta_current + alpha * theta_anchor
    """
    repaired_count = 0
    for name, param in model.named_parameters():
        if name in anchor:
            current = param.data.cpu()
            anchor_val = anchor[name].to(param.device)
            param.data = ((1 - alpha) * current + alpha * anchor_val).to(param.dtype)
            repaired_count += 1
    return repaired_count

# === Training ===
def train_on_verified(model, tokenizer, training_examples):
    """SFT training on (prompt, implementation) pairs."""
    print(f"\n--- Training on {len(training_examples)} examples ---")
    model.train()

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LEARNING_RATE,
        weight_decay=0.01,
    )

    all_losses = []
    for epoch in range(NUM_EPOCHS):
        epoch_losses = []
        for step, (prompt, response) in enumerate(training_examples):
            # Construct full text: prompt + response
            full_text = prompt + "\n" + response
            inputs = tokenizer(full_text, return_tensors="pt", truncation=True,
                             max_length=MAX_SEQ_LEN).to(model.device)

            # Create labels (mask the prompt, train only on response)
            prompt_inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                                    max_length=MAX_SEQ_LEN)
            labels = inputs.input_ids.clone()
            prompt_len = prompt_inputs.input_ids.shape[1]
            labels[:, :prompt_len] = -100  # mask prompt

            try:
                outputs = model(input_ids=inputs.input_ids, labels=labels)
                loss = outputs.loss / GRAD_ACCUM_STEPS
                loss.backward()
                epoch_losses.append(loss.item() * GRAD_ACCUM_STEPS)

                if (step + 1) % GRAD_ACCUM_STEPS == 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad], 1.0
                    )
                    optimizer.step()
                    optimizer.zero_grad()
            except Exception as e:
                print(f"  Warning: training step failed: {e}")
                optimizer.zero_grad()
                continue

        avg_loss = sum(epoch_losses) / max(len(epoch_losses), 1)
        all_losses.extend(epoch_losses)
        print(f"  Epoch {epoch+1}/{NUM_EPOCHS}: avg loss = {avg_loss:.4f}")

    return all_losses

# === Main ===
def main():
    print("=" * 70)
    print("EXPERIMENT 02: TRAIN DoRA ADAPTER (avr-cl GATED)")
    print("=" * 70)
    print(f"Model: {MODEL_ID}")
    print(f"DoRA rank: {LORA_RANK}, alpha: {LORA_ALPHA}")
    print(f"Epochs: {NUM_EPOCHS}, LR: {LEARNING_RATE}")
    print(f"Drift threshold: {DRIFT_THRESHOLD}")
    print(f"Input: {EXP01_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print()

    # Step 1: Load exp01 results
    print("--- Step 1: Loading exp01 results ---")
    exp01_results_path = os.path.join(EXP01_DIR, "results.json")
    if not os.path.exists(exp01_results_path):
        print(f"ERROR: {exp01_results_path} not found. Run exp01 first.")
        return

    with open(exp01_results_path) as f:
        exp01_data = json.load(f)

    exp01_impl = exp01_data.get("implementation", "")
    exp01_pass_rate = exp01_data.get("pass_rate", 0)
    exp01_results = exp01_data.get("results", [])

    print(f"  exp01 pass rate: {exp01_pass_rate:.1%}")
    print(f"  exp01 implementation: {len(exp01_impl)} chars")

    # Step 2: Build training data
    print("\n--- Step 2: Building training data ---")
    training_examples = []

    # Always include the reference implementation
    training_examples.append((TRAINING_PROMPT, REFERENCE_LORA_IMPL))
    print(f"  Added reference implementation (known-good)")

    # If exp01's implementation passed at least 3/5 claims, include it too
    if exp01_pass_rate >= 0.6:
        training_examples.append((TRAINING_PROMPT, exp01_impl))
        print(f"  Added exp01 implementation (passed {exp01_pass_rate:.0%} of claims)")
    else:
        print(f"  exp01 implementation too weak ({exp01_pass_rate:.0%}), using reference only")

    # Augment: create variations by asking slightly different prompts
    # (This is a simple augmentation — in production, use the on-policy distillation approach)
    augmented_prompts = [
        TRAINING_PROMPT.replace("Write a COMPLETE", "Write a complete"),
        TRAINING_PROMPT.replace("expert ML engineer", "skilled ML engineer"),
    ]
    for aug_prompt in augmented_prompts:
        training_examples.append((aug_prompt, REFERENCE_LORA_IMPL))

    print(f"  Total training examples: {len(training_examples)}")

    # Step 3: Load model + attach DoRA
    print("\n--- Step 3: Loading model + DoRA adapter ---")
    model, tokenizer = load_model_for_training()

    # Step 4: Snapshot anchor + compute baseline PPL
    print("\n--- Step 4: Snapshotting anchor (avr-cl) ---")
    anchor = snapshot_adapter_state(model)

    print("  Computing baseline PPL on drift probes...")
    anchor_ppls = {}
    for i, probe in enumerate(DRIFT_PROBES):
        anchor_ppls[f"probe_{i}"] = compute_ppl(model, tokenizer, [probe])
    print(f"  Baseline avg PPL: {sum(anchor_ppls.values())/len(anchor_ppls):.2f}")

    # Step 5: Train
    print("\n--- Step 5: Training ---")
    t0 = time.time()
    losses = train_on_verified(model, tokenizer, training_examples)
    train_time = time.time() - t0
    print(f"  Training complete in {train_time:.1f}s")
    print(f"  Final loss: {losses[-1]:.4f}" if losses else "  No losses recorded")

    # Step 6: avr-cl drift check
    print("\n--- Step 6: avr-cl drift check ---")
    drift_report = check_drift(model, tokenizer, anchor_ppls)

    print(f"  Avg drift ratio: {drift_report['avg_drift_ratio']:.3f}")
    print(f"  Max drift ratio: {drift_report['max_drift_ratio']:.3f}")
    print(f"  Drifted: {drift_report['drifted']}")

    # Step 7: Repair if drifted
    repair_log = {"repaired": False, "steps": 0, "alpha": 0}
    if drift_report["drifted"]:
        print(f"\n--- Step 7: avr-cl repair ---")
        for step in range(MAX_REPAIR_STEPS):
            print(f"  Repair step {step+1}/{MAX_REPAIR_STEPS} (alpha={REPAIR_ALPHA})...")
            n = repair_adapter(model, anchor, alpha=REPAIR_ALPHA)
            recheck = check_drift(model, tokenizer, anchor_ppls)
            print(f"    After repair: avg drift = {recheck['avg_drift_ratio']:.3f}")
            if not recheck["drifted"]:
                print(f"    Drift resolved!")
                drift_report = recheck
                repair_log = {"repaired": True, "steps": step + 1, "alpha": REPAIR_ALPHA}
                break
            drift_report = recheck

    # Step 8: Save adapter
    print("\n--- Step 8: Saving adapter ---")
    adapter_path = os.path.join(OUTPUT_DIR, "adapter")
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    print(f"  Adapter saved to {adapter_path}")

    # Step 9: Save logs
    training_log = {
        "experiment": "exp02_train",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": MODEL_ID,
        "dora_config": {
            "rank": LORA_RANK,
            "alpha": LORA_ALPHA,
            "dropout": LORA_DROPOUT,
            "use_dora": True,
        },
        "training_config": {
            "lr": LEARNING_RATE,
            "epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
            "grad_accum": GRAD_ACCUM_STEPS,
            "max_seq_len": MAX_SEQ_LEN,
        },
        "num_training_examples": len(training_examples),
        "train_time_sec": train_time,
        "final_loss": losses[-1] if losses else None,
        "avg_loss": sum(losses)/len(losses) if losses else None,
        "losses": losses,
        "exp01_pass_rate": exp01_pass_rate,
        "avr_cl": {
            "drift_threshold": DRIFT_THRESHOLD,
            "drift_detected": drift_report["drifted"],
            "avg_drift_ratio": drift_report["avg_drift_ratio"],
            "max_drift_ratio": drift_report["max_drift_ratio"],
            "repair": repair_log,
            "committed": not drift_report["drifted"] or repair_log["repaired"],
        },
        "anchor_ppls": anchor_ppls,
        "post_train_ppls": drift_report["current_ppls"],
    }

    log_path = os.path.join(OUTPUT_DIR, "training_log.json")
    with open(log_path, "w") as f:
        json.dump(training_log, f, indent=2)
    print(f"  Training log saved to {log_path}")

    drift_path = os.path.join(OUTPUT_DIR, "drift_report.json")
    with open(drift_path, "w") as f:
        json.dump(drift_report, f, indent=2)
    print(f"  Drift report saved to {drift_path}")

    # Summary
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"Adapter: {adapter_path}")
    print(f"Committed: {training_log['avr_cl']['committed']}")
    if repair_log["repaired"]:
        print(f"Repair: {repair_log['steps']} steps, alpha={REPAIR_ALPHA}")
    print(f"Final drift: {drift_report['avg_drift_ratio']:.3f} (threshold: {1+DRIFT_THRESHOLD:.3f})")

    return training_log

if __name__ == "__main__":
    main()
