"""
continual_pt — post-training runtime for behavioral continual learning.

The runtime takes X as input, researches X, trains a candidate update,
verifies X was learned, checks retention on prior commitments, and commits
or rejects. No agent persona. No fixed domain.
"""

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class VerifierType(Enum):
    EXECUTABLE = "executable"
    NON_EXECUTABLE = "non_executable"
    AUTO = "auto"  # try executable first, fall back


class Verdict(Enum):
    # Executable verifier
    PASS = "PASS"
    FAIL = "FAIL"
    # Non-executable verifier
    RECONSTRUCTED = "RECONSTRUCTED"
    OPEN = "OPEN"
    DISCARD = "DISCARD"
    # Common
    CONFUSED = "CONFUSED"  # couldn't determine


@dataclass
class Config:
    # Model
    model_id: str = "LiquidAI/LFM2.5-2.6B"
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "bfloat16"

    # DoRA adapter
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    use_dora: bool = True
    target_modules: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])

    # Training
    learning_rate: float = 1e-4
    num_epochs: int = 3
    batch_size: int = 1
    grad_accum_steps: int = 4
    max_seq_len: int = 2048

    # Generation — LFM2.5-2.6B recommended params (from model card)
    # temperature=0.1, top_k=50, repetition_penalty=1.1
    # Token budgets increased: model is verbose and was truncating mid-docstring
    max_new_tokens_impl: int = 4096
    max_new_tokens_test: int = 2048
    max_new_tokens_research: int = 4096
    temperature: float = 0.1  # Liquid AI recommended (was 0.7 — caused verbose broken code)

    # Verification
    verification_timeout: int = 60  # seconds for subprocess

    # Web research
    max_search_results: int = 5
    max_pages_to_read: int = 3
    search_delay: float = 2.0  # seconds between searches (rate limit)

    # Output
    output_dir: str = "./results"
    log_every: int = 1  # log every N steps
