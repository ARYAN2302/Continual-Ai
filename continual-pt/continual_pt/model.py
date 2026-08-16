"""
Model loading and generation utilities.

Loads LFM2.5-2.6B (or any HF model) in 4-bit NF4 with a DoRA adapter.
The adapter is the "fast weights" — the part that gets updated during learning.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from typing import Optional, Tuple
import copy


def load_model(config) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load model in 4-bit with DoRA adapter attached."""
    print(f"[model] Loading {config.model_id} in 4-bit NF4...")

    compute_dtype = getattr(torch, config.bnb_4bit_compute_dtype, torch.bfloat16)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=config.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=compute_dtype,
        trust_remote_code=True,
    )
    model.eval()

    # Attach DoRA adapter
    lora_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        use_dora=config.use_dora,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print(f"[model] Loaded. Device map: {getattr(model, 'hf_device_map', 'auto')}")
    return model, tokenizer


def load_model_with_adapter(config, adapter_path: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load base model + existing adapter from disk."""
    print(f"[model] Loading {config.model_id} with adapter from {adapter_path}...")

    compute_dtype = getattr(torch, config.bnb_4bit_compute_dtype, torch.bfloat16)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=config.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=compute_dtype,
        trust_remote_code=True,
    )

    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    print(f"[model] Loaded with adapter.")
    return model, tokenizer


def generate(model, tokenizer, user_prompt: str,
             max_new_tokens: int = 2048, temperature: float = 0.7) -> str:
    """Generate a response from the model."""
    messages = [{"role": "user", "content": user_prompt}]
    try:
        input_ids = tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True
        ).to(model.device)
    except Exception:
        input_ids = tokenizer(user_prompt, return_tensors="pt").input_ids.to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    response = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
    return response.strip()


def snapshot_adapter(model) -> dict:
    """Snapshot current adapter state (the anchor for rollback)."""
    anchor = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            anchor[name] = param.data.clone().detach().cpu()
    print(f"[avr] Anchored {len(anchor)} adapter parameters")
    return anchor


def restore_adapter(model, anchor: dict):
    """Restore adapter to a previous state (rollback)."""
    for name, param in model.named_parameters():
        if name in anchor:
            param.data = anchor[name].to(param.device).to(param.dtype)
    print(f"[avr] Restored {len(anchor)} parameters from anchor")


def save_adapter(model, tokenizer, path: str):
    """Save the current adapter."""
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    print(f"[model] Adapter saved to {path}")
