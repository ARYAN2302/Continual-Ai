"""
Modal app for continual-pt.

Runs the learning loop on an A10G GPU in detached mode.

Usage:
    # Run a sequence of X's (detached, survives disconnect)
    modal run --detached modal_app.py::run_sequence

    # Check logs
    modal app logs <app-id>

    # Run a single X
    modal run modal_app.py::run_single --x "LoRA"

The default sequence tests both verifier paths (executable + non-executable)
and checks retention across 3 sequential commits.
"""

import modal
import json
import os
import sys

# === Modal App Definition ===

app = modal.App("continual-pt")

# Image with all dependencies + local code
_local_dir = os.path.dirname(os.path.abspath(__file__))
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.1",
        "transformers>=4.46",
        "peft>=0.13",
        "bitsandbytes>=0.43",
        "accelerate>=0.34",
        "requests>=2.31",
        "beautifulsoup4>=4.12",
    )
    .apt_install("git")
    .add_local_dir(_local_dir, remote_path="/root/continual-pt",
                  ignore=[".git", "__pycache__", "tool-results", "*.pyc", "results/"])
)

# Volumes for persistence
hf_cache_vol = modal.Volume.from_name("continual-pt-hf-cache", create_if_missing=True)
results_vol = modal.Volume.from_name("continual-pt-results", create_if_missing=True)

# === Default Sequence ===
# This is the first real run. It tests:
# 1. Executable verifier (LoRA) — can the model implement and verify code?
# 2. Executable verifier (GRPO) — does it generalize to a new method?
# 3. Non-executable verifier — does the 5-step process work?
# The retention check after items 2 and 3 tests whether behavioral retention holds.

DEFAULT_SEQUENCE = [
    {
        "x": "LoRA (Low-Rank Adaptation)",
        "verifier_type": "executable",
        "claims": [
            "LoRA adds two low-rank matrices A (rank x in_features) and B (out_features x rank) to a frozen weight matrix.",
            "The original weight matrix W is frozen (requires_grad=False).",
            "The forward pass computes output = W @ x + (alpha/rank) * B @ A @ x.",
            "LoRA uses fewer trainable parameters than full fine-tuning: r*(d_in+d_out) instead of d_in*d_out.",
            "B is initialized to zeros so the initial output is unchanged from the base model.",
        ],
    },
    {
        "x": "GRPO (Group Relative Policy Optimization)",
        "verifier_type": "executable",
        "claims": [
            "GRPO samples a group of G responses for the same prompt.",
            "GRPO computes advantage as (reward - group_mean) / (group_std + eps).",
            "GRPO does not use a separate value/critic model.",
            "GRPO applies a KL penalty between current policy and reference policy.",
            "GRPO's loss is similar to PPO but with group-relative baseline instead of value function.",
        ],
    },
    {
        "x": "DiLoCo optimizer",
        "verifier_type": "non_executable",
        "claim": "DiLoCo reduces communication bandwidth in distributed training by compressing local gradients before transmission and applying outer optimizer steps less frequently than inner optimizer steps.",
    },
]


# === Modal Functions ===

@app.function(
    gpu="A10G",
    image=image,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/results": results_vol,
    },
    timeout=14400,  # 4 hours
    memory=16384,  # 16GB RAM
)
def run_sequence(sequence_json: str = None) -> str:
    """
    Run a sequence of X's. Each X goes through the full loop:
    absorb → verify → retain → commit.

    Pass sequence_json as a JSON string, or use the default sequence.
    """
    # Add local code to path
    sys.path.insert(0, "/root/continual-pt")

    from continual_pt import Config
    from continual_pt.runtime import learn_sequence

    # Parse sequence
    if sequence_json:
        x_list = json.loads(sequence_json)
    else:
        # Default: executable-only (2 items, faster first run)
        x_list = [
            {
                "x": "LoRA (Low-Rank Adaptation)",
                "verifier_type": "executable",
                "claims": [
                    "LoRA adds two low-rank matrices A (rank x in_features) and B (out_features x rank) to a frozen weight matrix.",
                    "The original weight matrix W is frozen (requires_grad=False).",
                    "The forward pass computes output = W @ x + (alpha/rank) * B @ A @ x.",
                    "LoRA uses fewer trainable parameters than full fine-tuning: r*(d_in+d_out) instead of d_in*d_out.",
                    "B is initialized to zeros so the initial output is unchanged from the base model.",
                ],
            },
            {
                "x": "GRPO (Group Relative Policy Optimization)",
                "verifier_type": "executable",
                "claims": [
                    "GRPO samples a group of G responses for the same prompt.",
                    "GRPO computes advantage as (reward - group_mean) / (group_std + eps).",
                    "GRPO does not use a separate value/critic model.",
                    "GRPO applies a KL penalty between current policy and reference policy.",
                    "GRPO's loss is similar to PPO but with group-relative baseline instead of value function.",
                ],
            },
        ]

    # Config
    config = Config(
        model_id="LiquidAI/LFM2.5-2.6B",
        output_dir="/root/results",
    )

    # Run
    results = learn_sequence(x_list, config)

    # Commit results volume
    results_vol.commit()

    # Return summary
    summary = {
        "total": len(results),
        "committed": sum(1 for r in results if r.get("commit", {}).get("committed")),
        "rejected": sum(1 for r in results if not r.get("commit", {}).get("committed")),
        "results": [
            {
                "x": r.get("x"),
                "committed": r.get("commit", {}).get("committed"),
                "verdict": r.get("commit", {}).get("verdict"),
            }
            for r in results
        ],
    }
    return json.dumps(summary, indent=2)


@app.function(
    gpu="A10G",
    image=image,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/results": results_vol,
    },
    timeout=3600,
)
def run_single(x: str, verifier_type: str = "auto", claims_json: str = None, claim: str = None) -> str:
    """Run a single X through the loop."""
    sys.path.insert(0, "/root/continual-pt")

    from continual_pt import Config
    from continual_pt.runtime import learn_sequence

    x_spec = [{
        "x": x,
        "verifier_type": verifier_type,
        "claims": json.loads(claims_json) if claims_json else None,
        "claim": claim,
    }]

    config = Config(
        model_id="LiquidAI/LFM2.5-2.6B",
        output_dir="/root/results",
    )

    results = learn_sequence(x_spec, config)
    results_vol.commit()

    return json.dumps({
        "x": results[0].get("x"),
        "committed": results[0].get("commit", {}).get("committed"),
        "verdict": results[0].get("commit", {}).get("verdict"),
    }, indent=2)


@app.function(
    gpu="A10G",
    image=image,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/results": results_vol,
    },
    timeout=7200,
)
def run_resume(sequence_json: str, adapter_path: str = None) -> str:
    """Resume from a prior adapter and continue learning."""
    sys.path.insert(0, "/root/continual-pt")

    from continual_pt import Config
    from continual_pt.runtime import learn_sequence

    x_list = json.loads(sequence_json)

    config = Config(
        model_id="LiquidAI/LFM2.5-2.6B",
        output_dir="/root/results",
    )

    results = learn_sequence(x_list, config, adapter_path=adapter_path)
    results_vol.commit()

    return json.dumps({
        "total": len(results),
        "committed": sum(1 for r in results if r.get("commit", {}).get("committed")),
    }, indent=2)


@app.local_entrypoint()
def main():
    """Default entrypoint: run the default sequence."""
    result = run_sequence.remote()
    print(result)


# === How to run ===
#
# 1. Set up Modal token:
#    modal token set --token-id <ID> --token-secret <SECRET>
#
# 2. Run the default sequence (detached, survives disconnect):
#    modal run --detached modal_app.py
#
# 3. Run a custom sequence:
#    modal run --detached modal_app.py::run_sequence --sequence-json '[...]'
#
# 4. Run a single X:
#    modal run modal_app.py::run_single --x "LoRA" --verifier executable
#
# 5. Check logs of a detached run:
#    modal app logs <app-id>
#
# 6. Resume from a prior adapter:
#    modal run modal_app.py::run_resume --sequence-json '[...]' --adapter-path /root/results/adapters/adapter_v1
