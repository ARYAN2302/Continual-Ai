# The Project: A Self-Verifying Continual Research Agent on LFM2.5-2.6B

> **Status:** Final spec, August 7, 2026
> **Author:** Aryan
> **Foundation:** avr-cl + liquid_researcher lineage + the three-perspective synthesis
> **Goal:** A research agent that learns from every verified research session, on a T4x2, aimed at Prime Intellect

---

## The One-Sentence Pitch

A small-model research agent that verifies every claim it learns by *running it*, retention-gated by avr-cl, where the verifier does triple duty as the contradiction check, the write gate, and the improvement gate.

---

## The Architectural Decisions (locked from your answers)

These five decisions shape everything downstream. They're locked.

### Decision 1: Verification is operational
The verifier doesn't check "does the source exist" or "is this consistent with other claims." It checks **"can this claim be run, and does it produce the predicted outcome."**

This forces the domain to be operationalizable — claims that can be tested by execution, not just prose claims. This is the single most important constraint, and it determines the domain.

### Decision 2: Memory is weight-space (avr-style)
The agent's "knowledge" lives in weights, updated via LoRA/DoRA, gated by avr-cl verify-repair. The claim graph exists only as an audit log — a record of what was learned, with provenance — but it is NOT the substrate of learning. Weights are the substrate.

This means: every verified research session produces a candidate LoRA update. The avr-cl gate decides whether to commit (drift check on prior domains). If commit, weights change. If not, rollback.

### Decision 3: Curiosity is instrumental + mechanistic
The curriculum isn't driven by contradiction count or embedding distance. It's driven by:
- **Instrumental pull:** "I need to understand X to build Y" — the agent has a goal (build something), and the gap between current capability and goal drives the next research question.
- **Mechanistic pull:** "Why does this work this way?" — when the agent verifies a claim and the verification succeeds but the *reason* isn't clear, that's a curiosity pull.

This means the agent needs a goal. Not "research decentralized AI" (too vague). A concrete build target. The research serves the build.

### Decision 4: One verifier, triple duty
A single verifier component does:
- **Write gate (Q11):** before a claim commits to memory, the verifier checks "is this runnable without confusion?" — if the claim can't be operationalized or the operationalization fails, don't commit.
- **Contradiction check (Q5):** before a new claim writes, the verifier checks "does running this contradict a previously verified claim?" — if both can be run but produce contradictory outcomes, flag it.
- **Improvement gate (Q9):** before a weight update commits, the verifier checks "did the agent actually get better at the operational task?" — external signal, not self-eval.

The verifier's primitive operation: **run the claim, observe the outcome, compare to prediction.**

### Decision 5: The base is LFM2.5-2.6B
Confirmed. Released August 4, 2026. ToolSandbox 77.83. 128K context. Hybrid architecture. The model is good enough at agents; the harness is the differentiator.

---

## The Critical Implication: The Domain Must Be Operational

Because verification is operational (Decision 1), the domain can't be "decentralized AI training protocols" in the prose sense. It has to be a domain where claims can be *run.*

This is a hard constraint that actually simplifies the project. The candidates:

### Candidate A: Code/Tool Documentation (e.g., PyTorch FSDP2, Hugging Face transformers, FastAPI)
- Every claim is operationalizable: "FSDP2 uses DeviceMesh for sharding" → run a small FSDP2 setup, check DeviceMesh is involved.
- Verifier: execute the code, check the output matches the claim.
- Pros: maximally verifiable, huge community need, immediate utility.
- Cons: less research-flavored, more QA-flavored.

### Candidate B: ML Research Code (papers with code — e.g., attention mechanisms, LoRA, GRPO)
- Every claim is operationalizable: "LoRA reduces VRAM by X%" → run with and without LoRA, measure.
- Verifier: implement the method, run the experiment, check the result matches the paper's claim.
- Pros: research-flavored, directly relevant to Prime Intellect, the verification IS the research.
- Cons: harder to verify (need to actually run ML experiments), more compute.

### Candidate C: Distributed Systems Protocols (e.g., consensus, sharding, verification schemes)
- Some claims operationalizable: "TOPLOC detects mismatched models" → run TOPLOC on mismatched models, check it flags.
- Some claims not operationalizable: "decentralized training is safer for AI" → can't run this.
- Pros: directly relevant to Prime Intellect.
- Cons: mixed verifiability — only some claims can be run.

### My Recommendation: Candidate B (ML Research Code)

Reasons:
1. **Maximally aligned with Prime Intellect.** They train models. An agent that learns ML methods by implementing and verifying them is exactly the "autonomous AI research agent" Vincent wants.
2. **Verification IS the research.** Implementing a paper's method and checking if it works is literally what research is. The verifier is not a separate component — it's the research itself.
3. **The agent becomes useful.** After 50 verified sessions, the agent has implementations of 50 ML methods. That's a real artifact.
4. **Curiosity has a natural pull.** "I implemented GRPO. Why does it use group-relative baseline? Let me research that." The mechanistic curiosity (Decision 3) is naturally triggered by implementation.

The goal/build target (Decision 3): **the agent builds a library of verified ML method implementations.** Each research session adds one method to the library. The library is the goal; the research serves it.

---

## The Architecture

### Layer 0: The Base
- LFM2.5-2.6B, 4-bit NF4, frozen
- 128K context — exploit this. Keep full code, full paper text, full execution traces in context.
- T4x2: one GPU for inference, one for execution/verification.

### Layer 1: The Distilled Harness (from liquid_researcher v5, ported)
- 2 DoRA heads: Planner+Executor (combined), Synthesizer
- Trained via on-policy distillation from frontier teacher (the 2026 upgrade from v5's plain SFT)
- Tools: SEARCH (SearXNG), READ (fetcher), NOTE (summarizer), CODE_WRITE (writes implementation), CODE_RUN (executes in sandbox), ANSWER (synthesis)
- The new tools vs v5: CODE_WRITE and CODE_RUN. These are how verification happens.

### Layer 2: The Operational Verifier (the load-bearing component)
This is the single component that does triple duty. Its primitive operation:

```
verify(claim, implementation):
    1. Operationalize: translate the claim into a runnable test
       (e.g., "GRPO uses group-relative baseline" →
        test: run GRPO on a group, check the baseline is the group mean)
    2. Execute: run the implementation + the test
    3. Observe: capture the outcome
    4. Compare: does the outcome match the claim's prediction?
    5. Verdict: PASS (claim verified) / FAIL (claim falsified) / CONFUSED (can't operationalize)
```

The three jobs:
- **Write gate:** before committing a claim to memory, run verify(). Only PASS commits.
- **Contradiction check:** before committing, run verify() on the new claim AND verify() on any existing claim it might contradict. If both PASS but contradict, flag.
- **Improvement gate:** after a candidate LoRA update, run verify() on a held-out set of claims. If pass rate drops, the update is rejected (avr-cl repair or rollback).

### Layer 3: The Continual Loop (avr-style)
- Each verified research session → SFT example (question → grounded synthesis + implementation)
- Accumulate N verified sessions → train candidate DoRA update
- avr-cl verify-repair gate: check drift on prior domains (using verify() pass rate as the drift signal — if prior claims start failing, drift detected)
- Commit or rollback

### Layer 4: The Curiosity Queue (instrumental + mechanistic)
- The agent has a goal: "build a library of verified ML method implementations"
- Instrumental pull: "I want to add method X to the library. What do I need to research to implement it?"
- Mechanistic pull: "I verified method X, but I don't understand why step 3 works. Research that."
- The queue is derived from the goal and from gaps in verification — not from a separate curiosity model.

### Layer 5: The Audit Log (symbolic, secondary)
- Every verified claim logged with: claim text, source, implementation, test, outcome, timestamp, adapter version
- This is for the X post and for human inspection. It's not the substrate of learning (weights are).
- This is where the "wow" lives — a public log of every claim the agent verified, with runnable code.

---

## The Verification Loop (the heart of the system)

This is what one research session looks like:

```
1. Goal: "Add GRPO to the library"
2. Research: agent searches, reads, notes
   → claim: "GRPO uses group-relative baseline instead of per-sample"
   → claim: "GRPO reduces variance vs PPO"
   → claim: "GRPO doesn't need a value model"
3. Implement: agent writes GRPO code
4. Verify each claim:
   - claim 1: run GRPO, check baseline is group mean → PASS
   - claim 2: run GRPO + PPO on same task, compare variance → PASS/FAIL
   - claim 3: inspect GRPO code, check no value model → PASS
5. Commit: only PASS claims enter the library + become SFT data
6. Curiosity: "claim 2 PASS but I don't understand why variance is lower. Queue: research variance reduction in GRPO."
7. Weight update: train DoRA on the verified session. avr-cl gate checks drift on prior methods. Commit or rollback.
```

The money shot: the library grows. Each entry has runnable code + verified claims + provenance. The agent gets better at implementing methods because it's been training on verified implementations.

---

## The Demonstration (the X post)

Run the agent for 4 weeks. It builds a library of ~15-20 verified ML method implementations. Each entry:
- Method name (e.g., "GRPO", "LoRA", "Toploc-style verification hashing", "DisTrO bandwidth reduction")
- The claims it makes about the method
- The runnable code that implements the method
- The tests that verify each claim
- The verification outcomes (PASS/FAIL with execution logs)
- Source provenance

The before/after:
- Cold start: agent implements a method, 30% of claims pass verification (it hallucinates details)
- After 15 verified sessions: agent implements a new method, 75% of claims pass (it learned what real implementation details look like)
- Retention: methods learned in session 1 still pass verification after session 15

The X thread:
> I built a research agent on LFM2.5-2.6B that learns ML methods by implementing and verifying them.
>
> It runs on a T4x2. It's been building a library of verified implementations for 4 weeks.
>
> Cold start: 30% of claims passed verification. After 15 verified sessions: 75%.
> Retention: methods from session 1 still pass at session 15.
>
> Every library entry has runnable code + tests + verification logs + source provenance. You can audit what the agent knows and how it came to know it.
>
> The loop: research → implement → verify → commit (if pass) → weight update (avr-gated) → next method.
>
> One verifier does triple duty: write gate, contradiction check, improvement gate. It runs the claim and observes the outcome.
>
> [link to library + repro]
>
> [tag Prime Intellect, Vincent, etc.]

---

## Timeline (6 weeks)

### Week 1: Port harness to LFM2.5-2.6B
- Port liquid_researcher v5 harness to 2.6B base
- Add CODE_WRITE and CODE_RUN tools
- Set up sandboxed execution (Modal function or local subprocess)
- Train initial DoRA heads via on-policy distillation

### Week 2: Build the verifier
- Implement verify(claim, implementation) primitive
- Operationalize-Execute-Observe-Compare-Verdict loop
- Test on 5 hand-picked ML methods with known implementations
- This is the hardest week. The verifier is everything.

### Week 3: Integrate the continual loop
- Connect verifier to avr-cl verify-repair gate
- SFT example generation from verified sessions
- Drift detection via verify() pass rate on prior methods
- Curiosity queue (instrumental + mechanistic)

### Week 4-5: Run the agent
- 15-20 research sessions
- Build the library
- Measure: verification pass rate over time, retention, hallucination rate
- Iterate on the verifier as failure modes surface

### Week 6: Polish and post
- Write up the library
- Prepare the X thread
- Repro repo
- Post

---

## What Could Kill This (honest risks)

1. **The verifier is too hard to build.** Operationalizing arbitrary claims into runnable tests is genuinely hard. If the agent can't operationalize most claims, the loop doesn't close.
   - Mitigation: start with a narrow set of claim types (e.g., "X uses Y" → check if Y appears in X's code; "X reduces Z by N%" → run with/without X, measure Z). Expand gradually.

2. **LFM2.5-2.6B can't write good enough code.** If the implementations are broken, verification always fails, nothing commits, no learning.
   - Mitigation: the agent can iterate on implementations (write, run, fix, re-run). The verifier gives feedback. This is the loop.

3. **avr-cl drift detection might not catch semantic drift.** PPL-based drift might not catch "the model forgot how to implement LoRA" if the PPL on LoRA-related text is stable.
   - Mitigation: use verify() pass rate as the drift signal, not just PPL. This is the upgrade from avr-cl's current drift detection.

4. **On-policy distillation is expensive on T4x2.** Might not fit.
   - Mitigation: fall back to plain SFT (v5's approach) if on-policy is too slow. The harness still works; it's just less sample-efficient.

5. **The domain is too narrow.** "ML method implementations" might not feel like "research" to Prime Intellect.
   - Mitigation: the methods are research methods (GRPO, TOPLOC, etc.). The agent is doing research by implementing it. The framing is right.

---

## What This Is Not

- Not a benchmark win. Not chasing SWE-bench scores.
- Not a framework. Not asking other people to build on it.
- Not a production agent. Not pretending to be a product.
- Not avr-cl repackaged. avr-cl is one component (the gate). The verifier is the new thing.
- Not a research paper. A working artifact with a public log.

---

## What This Is

A small-model research agent that learns by verifying. The verifier is the project. The library is the proof. The X post is the delivery.

One sentence: **the agent learns ML methods by implementing them and running the implementations to check they work, retention-gated by avr-cl, with one verifier doing triple duty as the write gate, contradiction check, and improvement gate.**

That's the project. The verifier is the missing primitive. Everything else is execution.
