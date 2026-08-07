# Synthesis: Three Model Perspectives on the Bottlenecks
# Date: August 7, 2026

---

## How to read this document

Three models answered the twelve bottleneck questions. Each came from a different angle:

- **Model 1** spoke as the artifact itself — reductive, mathematical, "I don't learn, I don't believe." Proposes external machinery for everything.
- **Model 2** reviewed Model 1 like a research note — sharpened distinctions, added nuance, hybrid solutions.
- **Model 3** was the practitioner with citations — grounded claims in real papers, corrected Model 1's pessimism with data, proposed concrete implementations.

What follows is not a summary. It's the **convergent insights** (where all three agree), the **divergences** (where they don't), the **new questions that emerged**, and the **questions that still need a human answer grounded from first principles**.

---

## Part 1: The convergent insights

These are the findings all three perspectives converge on. Treat these as established.

### Convergent Insight 1: The agent is not the LLM — the agent is the LLM plus a harness

All three treat the base model as a stateless conditional distribution $P(x_t | x_{<t})$ with frozen weights at inference. All the "human-seeming" capabilities — learning from a paper, tracking contradictions, curiosity, long-term improvement — are assigned to external machinery built on top.

**Implication:** The research agent is not "a better LLM." It's a harness wrapped around a frozen LLM. The harness is where the design work lives. This validates the project direction — the harness is the thing.

### Convergent Insight 2: The memory unit is wrong

All three agree: storing tokens (context window) or embeddings (vector DB) is the wrong unit of memory. The right unit is a **structured claim** with provenance.

Model 1: "structured knowledge graph where nodes are verified claims and edges are provenance links"
Model 2: hybrid — retrieval pulls candidates, then symbolic structures store the claims
Model 3: concrete schema — `{claim, confidence, provenance (source span + re-fetchable pointer), timestamp, scope tag, links to what it supports/contradicts}`

**Implication:** The memory layer should be a claim graph, not a vector store. This is a real architectural decision.

### Convergent Insight 3: The LLM cannot verify itself

All three agree that self-improvement without an external signal collapses. Model 1 calls it "mathematical certainty." Model 2 says the critic needs information the generator doesn't have. Model 3 provides the data: REINFORCE on own outputs goes 25% → 81% → 0% (collapse). SWE-bench with external test verification goes 17% → 53% (works).

**Implication:** Every learning loop needs an external verifier. The verifier is not optional. The verifier is the load-bearing component.

### Convergent Insight 4: Plausibility ≠ truth — the base objective is the enemy

All three identify the fundamental mismatch: the LLM is trained for plausibility (fluency, coherence), not truth. Distillation transfers plausibility. The harness must fight this.

Model 1: "the detector must monitor my internal state — logit variance, attention dispersion"
Model 2: "penalize plausible guessing in low-confidence regions"
Model 3: "stopping rule halts the loop once the model's answer stabilizes AND its calibrated probability clears a threshold"

**Implication:** The harness needs a confidence signal that is NOT the model's text output. The model will confidently emit falsehoods. Reading the text doesn't tell you when it's confused.

### Convergent Insight 5: Gap emission should be a first-class output

All three agree: the binary (answer / don't answer) is wrong. The honest research output is "partial answer + identified gap."

Model 1: "the reward model must penalize plausible guessing and reward boundary mapping"
Model 2: concrete format — Known / Uncertain / Needs-external-lookup
Model 3: every harness output carries a `gaps` field alongside its answer; an empty gaps list is itself a checkable claim

**Implication:** The synthesizer should emit structured gaps, not just an answer. The gaps feed the curiosity queue. The loop closes.

---

## Part 2: The divergences

These are the places the three perspectives don't agree. These are design decisions, not settled facts.

### Divergence 1: How to detect "belief" vs "pattern-match"

- **Model 1:** Mechanistic interpretability — find a direction in activations that stays consistent. Or robustness to perturbation.
- **Model 2:** Combine probing with semantic entropy (Farquhar, Nature). Grade the gap between literal and paraphrased accuracy.
- **Model 3:** Two options, both partial. Probing direction (with caveat: follow-up work found non-knowledge features that satisfy the same structure). Semantic entropy (cheaper, deployable, clusters meanings and computes entropy over clusters).

**The open question:** Which belief-test do you bet on? Probing is expensive but precise. Semantic entropy is cheap but noisy. Both have known failure modes.

### Divergence 2: How to handle scope (transformers vs mamba problem)

- **Model 1:** EWC or orthogonal gradient descent — mathematical constraints on weight updates.
- **Model 2:** Acknowledges classical AI belief-revision theory. Doesn't commit.
- **Model 3:** LoRA adapter per scope + router. Tag each training example with "applies when" precondition. Explicitly calls out ROME/MEMIT as the cautionary tale (direct weight editing fails to propagate to logically related facts).

**The open question:** Is scope a weight-space property (Model 1/3) or a symbolic property (classical AI)? Your avr-cl is weight-space. The claim-graph is symbolic. Which do you trust?

### Divergence 3: How curiosity works

- **Model 1:** Intrinsic reward function in RL. Explicitly engineered.
- **Model 2:** Doesn't address directly.
- **Model 3:** Derive from existing signals — unresolved contradictions, low-confidence high-centrality claims, recurring failure signatures. Sequential Bayesian experimental design. No separate model needed.

**The open question:** Is curiosity an RL reward (expensive, needs training) or a derived signal from the claim graph (cheap, deterministic)?

### Divergence 4: How self-improvement actually works

- **Model 1:** Pessimistic — "mathematical certainty" of collapse. Needs external oracle.
- **Model 2:** Distinguishes collapse (no filter) from gated improvement (external correctness signal).
- **Model 3:** Corrects Model 1 with SIA data. Scaffold-only plateaus. Weight updates with external reward work (25% gain on legal, 12% faster kernels). The rule: "never let the critic be the same model in a different prompt."

**The open question:** Is self-improvement real (Model 3's framing: external-reward-gated weight updates) or is it always just "more training data with extra steps" (Model 1's framing)?

---

## Part 3: The single most important realization

Model 3 states it explicitly, and it's the architectural insight that could define the whole project:

> **"Three of them (Q5's contradiction check, Q9's external verifier, Q11's confusion gate) are the same component doing different jobs: a verifier. If that's really the bottleneck, that's where the leverage actually concentrates — get one verifier design right and it does triple duty instead of needing three separate builds."**

This is not a small claim. It means the project isn't twelve systems. It's one verifier, deployed at three points in the loop:

1. **At the write gate** (Q11): before a claim commits to memory, the verifier checks "is this confusing?" (low agreement across resamples = confusion = don't commit)
2. **At the contradiction gate** (Q5): before a new claim writes, the verifier checks "does this contradict existing claims?" (entailment check against the claim graph)
3. **At the improvement gate** (Q9): before a weight update commits, the verifier checks "did the agent actually get better?" (external reward signal, not self-eval)

One verifier. Three jobs. If this is right, the project simplifies dramatically.

But — and this is the critical question — **what IS the verifier for non-executable claims?**

For code: the verifier is a test suite (SWE-bench model). For math: the verifier is a solver. But for "Prime Intellect uses TOPLOC for verifiable inference" — what verifies that? The source URL existing? Another model checking? A rule-based NLI model? A human?

This is the deepest open question. All three models punt on it. Model 3 says "external checkable signal" but for research claims, the "checkable" part is exactly what's undefined.

---

## Part 4: New questions that emerged from the three perspectives

These weren't in the original twelve. They surfaced from the synthesis.

### NQ1. Does retrieval quality dominate memory schema?

Model 3 (Q4) drops a bombshell: *"a recent evaluation found retrieval and ranking dominate end-to-end memory quality far more than write-time structuring does — one comparison even found a plain deterministic ranking pipeline beating fancier LLM-structured stores."*

If true, this means the claim-graph schema (which all three models propose) might matter less than the retrieval pipeline. The fancy schema could be over-engineering.

**The question:** Do you invest in the claim-graph schema, or in the retrieval pipeline? The field says retrieval. Your intuition?

### NQ2. Does avr-cl have the ROME/MEMIT propagation problem?

Model 3 (Q2): *"ROME and MEMIT localize a target fact cleanly but reliably fail to propagate the edit to logically related facts, and carry a real risk of degrading nearby knowledge."*

Your avr-cl does closed-form weight interpolation. Does it propagate? If you repair "math capability," does the repair propagate to "math reasoning in code"? Or does it stay localized?

**The question:** Has your avr-cl testing shown propagation, or just localized retention? This determines whether weight-space repair is sufficient or whether it needs a symbolic layer on top.

### NQ3. Is "feature density from specialized pretraining" a real escape hatch?

Model 2 (Q8): *"feature density doesn't only come from size; it can also come from specialized, high-quality pretraining in niches, which gives small models expert-like performance within narrow bands."*

If true, this means LFM2.5-2.6B might be more capable in narrow domains than its size suggests — IF the domain aligns with its pretraining.

**The question:** Does LFM2.5-2.6B's pretraining align with your target domain (decentralized AI training)? Or do you need a domain where the base model is already dense?

### NQ4. The "calibrated stopping rule" — is it the cheapest high-leverage primitive?

Model 3 (Q7): *"a training-free stopping rule halts the loop once the model's answer stabilizes across rounds and its calibrated probability of being correct clears a fixed threshold, using only a simple frozen calibrator."*

This is cheap, training-free, and directly addresses the over-searching failure mode in your liquid_researcher. It might be the single highest-ROI primitive to add.

**The question:** Is this worth adding to the harness before anything else? It's the cheapest fix for a documented failure.

### NQ5. The "one verifier, triple duty" claim — is it actually true?

Model 3 claims the contradiction check, external verifier, and confusion gate are the same component. But they detect different things:
- Contradiction: new claim vs existing claims (semantic)
- External verification: claim vs ground truth (factual)
- Confusion: internal state of the model (epistemic)

Are these really the same component? Or are they three different verifiers that happen to all be called "verifier"?

**The question:** Does the unification hold? If yes, the project is one component. If no, it's three.

---

## Part 5: The questions that need YOUR human take

These are the questions where the field's answers don't resolve the design decision. They need a human answer grounded in how you actually think, learn, and ship — the same kind of first-principles intuition that produced avr-cl.

### Human Q1. When you verify a research claim that isn't code or math, what is your actual mental operation?

This is the crux. Code can be run. Math can be checked. But "continual learning is the bottleneck for AGI" — how do you know that's true?

When you read a research claim and decide "yes, I believe this," what did you actually do? Did you:
- Check the source exists? (provenance)
- Check the source is authoritative? (reliability)
- Check it against other sources? (corroboration)
- Check it against your existing beliefs? (consistency)
- Check the reasoning? (internal logic)
- Try to use it? (operationalization)

The verifier design depends on this answer. If your verification is mostly "check the source + check consistency with what I already know," the verifier is a provenance check + an entailment check. If it's "try to use it," the verifier is operational — and that limits the domain to things you can try.

**Why this matters:** This is the question that determines the entire verifier architecture. Every model punted on it. Only you can answer it because only you know how you actually decide a research claim is verified.

### Human Q2. When you hold two facts with different scope (transformers use attention / mamba doesn't), how do you actually keep them separate in your head?

This is the scope problem. The field says: tag with "applies when" precondition. But how do you actually do it?

When you learned "mamba doesn't use attention," did you:
- Create a new mental category ("state-space models") and file mamba there?
- Add a tag to the attention fact ("applies to transformers only")?
- Just remember "transformers and mamba are different" without explicit scope?
- Something else?

The scope mechanism depends on this. If your scoping is categorical (new category per scope), the system needs a taxonomy. If it's tagged (precondition per fact), the system needs a tag schema. If it's implicit (you just know they're different), the system can't replicate that — it needs explicit structure.

**Why this matters:** avr-cl operates in weight space. The claim-graph operates in symbol space. The question of which is right depends on how scoping actually works in your head. If your scoping is weight-space-feeling ("I just know"), avr-cl might be sufficient. If it's structural ("I have a taxonomy"), you need the claim graph.

### Human Q3. When you're pulled into a research rabbit hole, what actually pulled you?

This is the curiosity signal. The field says: derive it from contradictions, low-confidence high-centrality claims, or recurring failures. But which of those actually pulls YOU?

Think about the last time you went down a rabbit hole. What was the trigger?
- "Wait, that contradicts what I thought" (contradiction)
- "I don't understand this and it seems important" (gap)
- "This keeps coming up and I keep not getting it" (recurring failure)
- "This is surprising and I want to know why" (surprise)
- "I need this for something else" (instrumental)

The curiosity queue design depends on this. If your pull is contradiction, the queue is contradiction-driven. If it's surprise, the queue needs a surprise detector (which is the Q11 confusion gate wearing a different hat).

**Why this matters:** Curiosity drives the curriculum. The curriculum determines what the agent learns next. If the curiosity signal is wrong, the agent learns the wrong things.

### Human Q4. Do you trust weight-space memory (avr-cl) or symbolic memory (claim graph) more?

This is the architectural fork. The three models lean toward symbolic (claim graph) because weight-space can't do scope cleanly. But your avr-cl is weight-space and it works.

When you think about "the model knows X," do you picture:
- The weights having shifted to encode X (connectionist)
- A stored claim "X is true" that can be retrieved (symbolic)
- Both — weights for the feel, symbols for the audit

If you trust weights, the project is avr-cl extended. If you trust symbols, the project is a claim-graph with an LLM frontend. If both, the project is a hybrid — and the hybrid is the hardest to build.

**Why this matters:** This is the biggest architectural decision. It determines everything downstream. And it's not a question the field can answer — it's a question about what you believe.

### Human Q5. Does the "one verifier, triple duty" claim land as true from your experience?

Model 3 says contradiction check, external verifier, and confusion gate are the same component. Does that feel right to you?

When you're reviewing a research claim:
- Checking it against what you already know (contradiction)
- Checking it against the source (verification)
- Checking whether you understand it (confusion)

Are these the same mental operation? Or are they three different things you do?

If they're the same, the project is one verifier. If they're different, the project is three verifiers — and the "simplicity" is illusory.

**Why this matters:** This determines whether the project is "build one verifier well" or "build three verifiers and coordinate them." The scope difference is enormous.

---

## Part 6: What the project looks like depending on your answers

Not predicting your answers — just showing how they fork the design.

**If you trust weight-space (avr-cl) + verification is provenance+consistency:**
The project is avr-cl extended to research claims. Each verified research session → SFT example → avr-cl verify-repair gate. The claim graph is optional (for audit, not for learning). Curiosity is derived from drift signals. This is the simplest path and closest to what you've already built.

**If you trust symbolic (claim graph) + verification is operational:**
The project is a claim-graph agent. The LLM extracts claims, the verifier checks them against sources, the claim graph stores them with scope, the curiosity queue pulls from contradictions. Weight updates are secondary (maybe not needed at all). This is a bigger build but more auditable.

**If you trust both (hybrid) + the verifier does triple duty:**
The project is a claim-graph for audit + avr-cl for weight updates + one verifier that gates writes, checks contradictions, and validates improvements. This is the most ambitious and the most likely to be what actually works. It's also the hardest to ship in 6 weeks.

**If the verifier is domain-specific (code/math vs prose):**
The project starts with a domain where verification is cheap (code, structured docs) and expands to prose later. This changes the domain choice — decentralized AI training might be too prose-heavy. A code-heavy domain (like PyTorch internals) might be a better v1.

---

## Part 7: My recommendation for next steps

1. **Answer Human Q1-Q5.** These are the forks. Everything else is implementation.

2. **The verifier is the project.** Regardless of your answers, all three perspectives converge on this: the verifier is the load-bearing component. The harness, the memory, the CL loop — they're all secondary to getting the verifier right.

3. **The domain should match the verifier.** If your verifier is "source exists + claim is grounded in source span," the domain can be prose-heavy (decentralized AI). If your verifier is "the claim can be executed/tested," the domain needs to be code or structured (PyTorch internals, API docs). Pick the domain after you know the verifier.

4. **The avr-cl question (NQ2) is testable now.** You can check whether your existing avr-cl results show propagation or just localized retention. This doesn't need new work — just re-analysis of existing data. If avr-cl propagates, weight-space is sufficient. If not, you need the symbolic layer.

5. **The cheapest high-leverage primitive (NQ4) is the calibrated stopping rule.** Training-free, directly addresses over-searching, can be added to the harness in a day. Worth doing regardless of architectural choices.

---

## Appendix: The three perspectives, condensed

### Model 1 (the artifact speaking as itself)
- Stance: reductive, mathematical, honest about limitations
- Key move: externalize everything — memory, verification, curiosity, confusion detection
- Weakness: doesn't distinguish collapse-without-filter from gated-improvement (Model 3 corrects this)
- Strongest insight: "the detector must monitor internal state, not text output"

### Model 2 (the reviewer)
- Stance: sharpening distinctions, hybridizing
- Key move: distinguish inference-time vs training-time learning; hybrid RAG+knowledge graph; structured gap emission
- Weakness: less grounded in specific papers than Model 3
- Strongest insight: "feature density can come from specialized pretraining, not just size"

### Model 3 (the practitioner with citations)
- Stance: grounded, corrective, actionable
- Key move: cite real papers (STaR, Self-RAG, ROME, MEMIT, SIA, semantic entropy, sequential Bayesian experimental design); correct Model 1's pessimism with data
- Weakness: leans toward "the field has partial answers" — might under-weight what's genuinely unsolved
- Strongest insight: "three of these are the same component doing different jobs: a verifier"
