# continual-kit — A Continual Learning Toolkit for Open-Weight LLM Fine-Tuners

> Status: Proposal draft, August 2026
> Author: Aryan (with research assistant)
> Target: Useful open-source tool + Prime Intellect hiring signal

---

## The pain (verified, August 2026)

Every fine-tune silently breaks something.

The field's confession, in their own words:
- *"Our AI feature was quietly degrading for 10 days after a model update. No errors, no crashes — just gradually worse."* (OpenAI community, 2026)
- *"I Audited a Fine-Tuned LLM That Lost 50 Percentage Points"* (Medium, 2026)
- *"Your adapter may degrade silently. Plan quarterly revalidation."* (BigDataBoutique, 2026)
- *"Catastrophic forgetting is the most common silent regression in fine-tuning."* (FutureAGI, 2026)

The current state of the art for catching this is "quarterly revalidation" — i.e., find out you broke it three months later.

## Who actually needs this

Three populations, each with real pain:

1. **Open-source model fine-tuners.** The Hugging Face community. Tens of thousands of people fine-tuning Qwen, Llama, Mistral every week. They all hit the same wall: every new fine-tune erases something from the previous one. There is no tool for them.

2. **Domain-specific model builders.** Medical, legal, financial, code-specific. They have a base model, fine-tune on domain data, then need to *keep updating* as the domain shifts. Full retrain is too expensive.

3. **Multi-LoRA practitioners.** People running multiple LoRA adapters on one base model. The current solution is adapter merging or routing — both brittle. A continual-learning approach that trains a sequence of LoRAs without catastrophic forgetting would replace this entire workflow.

**Population #1 is the actual market.** The Hugging Face fine-tuning community is the real user base for a useful continual learning tool. They're already doing the operations that cause forgetting. They already have the pain. They just don't have the tool.

## The toolkit

Three modes, all on a T4x2:

### 1. Diagnose

```bash
$ continual-kit diagnose \
    --base Qwen/Qwen3-4B-Instruct \
    --adapter ./my-finetune.safetensors \
    --suite default

→ Diagnosing adapter against base...
→ Running 12-probe capability suite (math, code, instruct, safety, reasoning)...
→ Results:
    math:       -8%  (drift detected)
    code:       +12% (target gain, expected)
    instruct:   -3%  (minor drift)
    safety:     -15% (CRITICAL DRIFT)
    reasoning:  +1%  (stable)
→ 2 silent regressions found.
→ Run `continual-kit repair` to fix.
```

### 2. Repair

```bash
$ continual-kit repair \
    --base Qwen/Qwen3-4B-Instruct \
    --adapter ./my-finetune.safetensors \
    --output ./my-finetune-repaired.safetensors

→ Repairing drift on math, safety dimensions...
→ Closed-form interpolation, alpha=0.1, 5 steps...
→ Re-checking:
    math:       -1%  (repaired)
    safety:     -2%  (repaired)
→ Repaired adapter saved.
```

### 3. Stack

```bash
$ continual-kit stack \
    --base Qwen/Qwen3-4B-Instruct \
    --adapters ./domain-a.safetensors ./domain-b.safetensors ./domain-c.safetensors \
    --output ./stacked.safetensors

→ Stacking 3 adapters with verify-repair gate...
→ After domain-a:  baseline + A
→ After domain-b:  baseline + A + B (A retained at 98%)
→ After domain-c:  baseline + A + B + C (A retained at 97%, B retained at 99%)
→ Stacked adapter saved.
```

## Why this is genuinely useful (not a demo)

- **Real users, real pain, real tool.** Not theater.
- **Adoption is the wow.** A tool with 500 stars and real issues is a different category than a demo.
- **It's the missing primitive in the open-source LLM stack.** TRL/Axolotl/Unsloth train. Mergekit merges. Nothing repairs forgetting. That's the gap.
- **It composes with Prime Intellect's stack at a real place.** Prime-RL runs could use it between RL stages. Prime Agent checkpoints could be diagnosed with it.
- **It's the wedge for the bigger vision.** Once adopted, the continual-learning-loop story has a user base and a proven primitive behind it.

## Why it makes Prime Intellect say "wow"

1. **It's a real primitive they don't have.** They have training infra, agents, harnesses, verifiers. They do not have a *forgetting repair* primitive.
2. **It demonstrates the primitive at scale of adoption, not scale of compute.** "Thousands of HF users run my forgetting-repair tool" is a stronger signal than "I ran one agent for 14 days."
3. **It's the wedge for the bigger vision.** The tool is the wedge; the platform is the follow-up.

## The default capability probe suite

The opinionated core. Needs to be:
- **Small enough to run on a T4 in 5 minutes** (12 probes, not 500)
- **Broad enough to catch real drift** (math, code, instruction-following, safety, reasoning)
- **Standardized** so results are comparable across models/adapters
- **Open** so users can contribute probes for specific domains

## Why this isn't sloppy

- No fake users. No dogfooding theater. No "production agent" pretending to be a product.
- The tool either works or it doesn't. Users will tell you immediately.
- The bar is measurable: stars, forks, issues, real usage.
- It composes with the existing ecosystem (TRL, Axolotl, Unsloth, mergekit) instead of competing with it.
- It's small enough to ship in 2-3 weeks and useful enough to grow for years.

## The X post (short, because the tool does the talking)

> Every fine-tune silently breaks something.
>
> I built a tool that tells you what, and fixes it.
>
> $ continual-kit diagnose --base Qwen3-4B --adapter ./yours.safetensors
>
> [screenshot of report]
>
> $ continual-kit repair ...
>
> [screenshot of repaired report]
>
> Closed-form weight interpolation, no gradients, no old data. Runs on T4. Composes with TRL/Axolotl/Unsloth.
>
> Three modes: diagnose, repair, stack (sequential fine-tunes without forgetting).
>
> Open source. The capability probe suite is open — contribute probes for your domain.
>
> [link]

## Open questions

1. **Model size scope for v1.** T4x2 comfortably handles 4B in 4-bit with LoRA. Can v1 target 1.5B–7B? 7B+ in 4-bit fits on T4 but tight.

2. **Default capability probe suite.** Design from scratch, or reuse an existing eval set (lm-evaluation-harness subset, Open LLM Leaderboard subset, custom from avr-cl)?

3. **Timeline.** 2 weeks to a usable v1 with diagnose + repair? 3 weeks with stack mode too?
