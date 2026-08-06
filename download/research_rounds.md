# Research Rounds: LFM2.5-2.6B Research Harness + Continual Learning
# Date: August 7, 2026

---

# ROUND 1: How to build the research harness on LFM2.5-2.6B

## The model (released August 4, 2026 — 3 days ago)

LFM2.5-2.6B benchmarks:
- ToolSandbox: 77.83 (beats Qwen3.5-9B at 76.44)
- IFBench: 59.17
- Agentic tool use rank: #59/131, score 48.3/100 — competitive with models 4x its size
- 128K context, 220 tok/s on M5 Max, runs in <2.5GB
- Hybrid GQA + short convolutions (not pure transformer) — KV cache reduction is the architectural edge

**Implication:** This is a real agentic base, not a toy. The harness doesn't have to fight the model. The 2.6B can follow tool-call patterns natively.

## The dominant OSS research-agent architecture (converging pattern)

GPT-Researcher (28.8k stars), Perplexica, Anthropic's multi-agent research system all use the same shape:
- **Planner agent** generates sub-questions / research outline
- **Executor agent(s)** gather information per sub-question
- **Synthesizer** writes the final cited report
- Orchestrator-worker pattern (Anthropic's term)

**Implication:** Validates liquid_researcher v5 architecture. The shape was right. Don't redesign it.

## Distillation state of the art (2026 — past naive SFT)

Three relevant advances:

1. **On-Policy Distillation (MOPD)** — the 2026 frontier pattern. Sample student trajectories, have teacher grade them, train on graded traces. Better than off-policy SFT because the student learns from its own failure modes.

2. **Agent Distillation framework (NeurIPS 2025)** — specifically transfers tool-use behavior from LLMs to SLMs. Two methods: first-thought prefix + retrieval-augmented distillation.

3. **Trajectory-Refined Distillation (2026)** — teacher revises student's failed rollouts. Right pattern when student mostly works but fails on specific cases.

4. **SDFT (Self-Distillation Fine-Tuning, ICML 2026)** — on-policy learning directly from demonstrations. KL-constrained. Proven to enable continual learning. (Critical for Round 2.)

**Implication:** v5 used plain LIMA-style SFT. v6 should use on-policy distillation. This is the upgrade.

## Eval benchmarks (mature)

- **DeepResearch Bench** — 100 PhD-level tasks, 22 domains, RACE + FACT metrics
- **DeepResearch Bench II** (Jan 2026) — 132 tasks, adds report-quality eval
- **FRAMES** — multi-hop factuality
- **HalluHard** (Feb 2026) — hard multi-turn hallucination benchmark

**Implication:** Use DRB + FRAMES. Don't invent new evals.

## DoRA > LoRA for agentic distillation

Confirmed by NVIDIA, 2026 practice. Drop-in via PEFT, +2-4% quality. v2.3 already had this.

## Harness shape that fits LFM2.5-2.6B

- Base: LFM2.5-2.6B (frozen, 4-bit NF4 on T4)
- Adapters: 2 DoRA heads (Planner+Executor combined, Synthesizer separate)
- Training: on-policy distillation (not pure SFT)
- Tools: SEARCH (SearXNG), READ (fetcher), NOTE (summarizer), ANSWER
- Context: 128K is the model's strength — keep full raw snippets, not compressed notes
- Skip SANDBOX_RUN for v1

---

# ROUND 2: How to add continual learning

## The 2026 CL landscape

Key finding: **The field has converged on three patterns, and they compose.**

### Pattern A: LoRA-based CL (your existing primitive)

- **"Merge before Forget" (ICLR 2026)** — orthogonally initializes and sequentially merges LoRA updates into a single adapter. Provably reduces forgetting.
- **"LoRA Provably Reduces Forgetting" (OpenReview)** — theoretical proof that LoRA is more forgetting-resistant than full fine-tuning.
- **C-LoRA** — self-regularized low-rank adaptation in cross-attention layers.
- **Stacked LoRA (ACL 2025)** — isolated low-rank adaptation for lifelong learning.
- **SeqLoRA** — naive sequential fine-tuning baseline (forgets significantly; the cautionary tale).

**Implication:** Your avr-cl approach (verify-repair between LoRA updates) fits directly into the ICLR 2026 "merge before forget" paradigm. You're not inventing a new CL method — you're adding the missing verify-repair gate to an established pattern.

### Pattern B: Self-Distillation Fine-Tuning (SDFT, ICML 2026)

- On-policy learning directly from demonstrations
- KL-constrained policy optimization
- Proven to enable continual learning without forgetting
- Additive update rule = exact closed-form solution to KL-constrained objective

**Implication:** This is theoretically the cleanest CL method. But it requires running the student model during training (on-policy), which is more expensive than SFT. For T4x2, this might be too slow for a 2.6B model.

### Pattern C: Replay-based methods

- Mix old data (or generated approximations) back into training
- "Naive replay still beats most sophisticated methods" (HuggingFace, June 2026)
- Practical but requires storing old training data

**Implication:** This is the pragmatic baseline. Your avr-cl doesn't use replay (no old data stored), which is a strength (privacy, simplicity) but also a limitation (replay is empirically the strongest anti-forgetting method).

## How to integrate CL into the research harness

The architecture has three layers:

### Layer 1: Distilled harness (from Round 1)
LFM2.5-2.6B + 2 DoRA heads (Planner+Executor, Synthesizer). On-policy distillation from frontier teacher. Produces the research loop shape.

### Layer 2: Verification gate (you have this)
After each research session, verify:
- Sources are real (URL + content hash)
- Synthesis is grounded (every claim has source span)
- No hallucinated citations
- Quality bar met (LLM-judge score above threshold)

Only verified sessions become training data. This is critical — CL on garbage amplifies hallucination.

### Layer 3: Continual knowledge loop (the new thing)
Three implementation options, ranked by tractability on T4x2:

**Option 1 (recommended): avr-cl verify-repair between SFT updates**
- Each verified research session → SFT example (question → grounded synthesis)
- Accumulate N verified sessions → train candidate DoRA update
- avr-cl verify-repair gate: check drift on prior domains, repair or commit
- Pros: uses your existing primitive, runs on T4x2, proven to work
- Cons: off-policy (doesn't learn from student's own failures)

**Option 2: SDFT (on-policy self-distillation)**
- Same as Option 1, but use SDFT instead of SFT
- Student generates its own trajectory, teacher grades, train on graded trace
- Pros: theoretically cleaner, learns from student's failure modes
- Cons: ~3-5x more compute (student inference during training), might not fit on T4x2

**Option 3: Replay-augmented avr-cl**
- Option 1 + small replay buffer of prior verified sessions
- Pros: empirically strongest anti-forgetting
- Cons: requires storing old data (less elegant, but practical)

**Recommendation:** Start with Option 1 (avr-cl + SFT). Move to Option 3 (add replay) if forgetting is severe. Skip Option 2 unless you have more compute.

## The verification-gate design (critical)

This is what makes CL safe. Without it, you amplify hallucination.

For each research session, verify:
1. **Source existence** — every URL in the report resolves to a real page (HTTP 200)
2. **Source grounding** — every claim in the synthesis has a quoted span from a fetched source
3. **Citation accuracy** — every [n] marker resolves to a URL in the trace
4. **Quality bar** — LLM-judge (or rule-based) scores the report above threshold
5. **No training-cutoff fallback** — the model didn't just emit pretrained knowledge; it actually used fetched sources

Only sessions passing all 5 checks become training data. Sessions that fail are logged for analysis but not trained on.

## The measurable demonstration

The "wow" is a before/after comparison:

**Phase 1 (cold start):** Run agent on 10 questions in Domain A. Measure:
- Hallucination rate (citations that don't exist)
- Citation accuracy (URLs that resolve to claimed content)
- Synthesis quality (LLM-judge score)

**Phase 2 (CL update):** Feed verified findings from Phase 1 through AVR-gated loop. Weights update.

**Phase 3 (re-test):** Run agent on 10 NEW questions in Domain A. Measure same metrics. Show:
- Hallucination rate dropped
- Citation accuracy improved
- Synthesis quality improved

**Phase 4 (retention):** Run on 10 questions in Domain B. Show no degradation.

**The graph:** hallucination rate vs. research sessions, showing staircase down. Retention held.

---

# ROUND 3: What domains to target

## The selection criteria

A good domain for demonstrating CL-on-research must satisfy ALL of:

1. **Knowledge-bound, not reasoning-bound** — CL adds knowledge, not reasoning. The domain's difficulty must come from "model doesn't know this" not "model can't reason about this."
2. **Rapidly evolving** — the model's training cutoff is a real handicap. CL has measurable impact.
3. **Verifiable sources** — every claim can be checked against arxiv/papers/code/docs. No subjective claims.
4. **Frontier models hallucinate here** — if Claude/GPT also hallucinate, your CL advantage is visible.
5. **You can generate 50+ real research questions** — enough to demonstrate accumulation.
6. **Domain expert (you) can verify** — you need to be able to spot hallucinated citations.

## The candidates (ranked)

### Tier 1 (strongest fit)

**1. AI/ML research frontier (2025-2026 papers)**
- Knowledge-bound: yes, the model doesn't know post-cutoff papers
- Rapidly evolving: extremely — new papers weekly
- Verifiable: arxiv IDs, paper titles, GitHub repos
- Frontier hallucination: high — models invent plausible arxiv IDs constantly
- Question generation: easy — you read this field
- Your verification: strong — you know the field
- **Pros:** Perfect fit. Prime Intellect will immediately grok it (it's their world).
- **Cons:** Crowded — everyone demos on AI research.

**2. Distributed training protocols (Prime Intellect's exact domain)**
- Knowledge-bound: yes — prime-RL, TOPLOC, INTELLECT-2, Pluralis, Psyche are all post-cutoff
- Rapidly evolving: extremely — Prime Agent launched 3 days ago
- Verifiable: GitHub repos, blog posts, arxiv papers
- Frontier hallucination: very high — even Claude doesn't know August 2026 developments
- Question generation: easy — this entire conversation is material
- Your verification: strong — you've researched this deeply
- **Pros:** Directly relevant to Prime Intellect. They will be forced to engage.
- **Cons:** Narrow — might not generalize.

**3. Continual learning research itself**
- Knowledge-bound: yes — CL papers from 2026 are post-cutoff
- Rapidly evolving: yes — SDFT, merge-before-forget, etc. are all 2026
- Verifiable: arxiv, GitHub
- Frontier hallucination: high
- Question generation: easy — you've been reading this
- Your verification: strong
- **Pros:** Meta-demonstration — the agent learns about learning. Cool narrative.
- **Cons:** Niche audience.

### Tier 2 (good fit, less directly relevant)

**4. A specific open-source codebase (e.g., PyTorch FSDP2, Hugging Face transformers)**
- Knowledge-bound: yes — APIs change fast
- Rapidly evolving: yes
- Verifiable: docs, source code, tests
- Frontier hallucination: moderate — models often cite deprecated APIs
- Question generation: easy
- Your verification: strong if you know the codebase
- **Pros:** Practical, useful, immediately verifiable.
- **Cons:** Less research-flavored, more QA-flavored.

**5. Bio/medical research (specific subdomain)**
- Knowledge-bound: yes
- Rapidly evolving: yes
- Verifiable: PubMed, clinicaltrials.gov
- Frontier hallucination: high
- **Pros:** High-impact domain.
- **Cons:** Verification is hard without domain expertise. Risk of giving wrong medical info.

### Tier 3 (avoid for v1)

**6. Finance (10-Ks, earnings)**
- Verifiable but tabular — CL for tables is a different problem
- Less knowledge-bound, more reasoning-bound

**7. History / art / humanities**
- Not rapidly evolving
- Harder to verify claims objectively

**8. Legal**
- Hallucination risk too high — Bar Association is suing over this
- Don't touch for a demo

## Recommendation: Tier 1, two domains in parallel

**Domain A (primary): Distributed training protocols / decentralized AI (August 2026 state)**
- Directly relevant to Prime Intellect
- Frontier models genuinely don't know this (post-cutoff)
- You have deep context from this conversation
- 50+ research questions easy to generate

**Domain B (retention check): A different technical domain**
- Pick something stable and well-known (e.g., "transformer architecture fundamentals" or "Linux kernel basics")
- Use this to prove retention — the model doesn't forget Domain B while learning Domain A
- This is the "BWT stays near zero" measurement from your avr-cl paper

## Why this domain choice makes the demo land

The X post becomes:

> I built a research agent on LFM2.5-2.6B that learns from every verified research session.
>
> Domain: decentralized AI training (Prime Intellect, Pluralis, Nous Psyche).
>
> Cold start: 32% citation accuracy, 41% hallucination rate.
> After 50 verified sessions: 78% citation accuracy, 9% hallucination rate.
> Retention on unrelated domain (transformer fundamentals): 99% held.
>
> The agent now knows more about decentralized AI training than Claude, because Claude is frozen and this one isn't.
>
> [link to repro + learning log]

That's the post. It's specific, measurable, relevant to Prime Intellect, and demonstrates the exact capability they don't have.

---

# SUMMARY: The build

## Stack
- Base: LFM2.5-2.6B (4-bit NF4, T4x2)
- Harness: 2 DoRA heads (Planner+Executor, Synthesizer), on-policy distillation from frontier teacher
- CL: avr-cl verify-repair gate between SFT updates on verified research sessions
- Verification: 5-check gate (source existence, grounding, citation accuracy, quality, no-cutoff-fallback)
- Search: SearXNG self-hosted
- Eval: DeepResearch Bench + FRAMES + custom domain eval

## Domains
- Primary: decentralized AI training protocols (Aug 2026 state)
- Retention: transformer fundamentals

## Demonstration
- Cold start → 50 verified sessions → re-test → retention check
- Metrics: hallucination rate, citation accuracy, synthesis quality, BWT

## Timeline (rough)
- Weeks 1-2: Port harness to LFM2.5-2.6B, on-policy distillation training
- Week 3: Verification gate + CL loop integration
- Weeks 4-5: Run 50 research sessions, CL updates, measure
- Week 6: Eval, writeup, X post

## The wow for Prime Intellect
- A 2.6B model that knows more about their own field than Claude does
- Because it learned from verified research, retention-gated, on a T4x2
- This is the small-AGI primitive in cell form
