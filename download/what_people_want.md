# What People Want in Research Agents (August 2026)

Deep research across Reddit, HN, academic blogs, product reviews, and practitioner complaints.

---

## The five things people actually want (ranked by frequency of complaint)

### 1. Stop hallucinating citations — verify before I trust

This is the #1 complaint, by a wide margin. It cuts across every tool — Perplexity, ChatGPT Deep Research, Gemini, Elicit, Consensus. The arxiv paper from April 2026 puts numbers on it: "Deep research agents generate substantially more citations per query than search-augmented LLMs but hallucinate URLs at higher rates." Domain-specific tools (legal, medical) are worse — the fake legal citation problem is so bad that bar associations are suing.

The user pain is not "sometimes the URL is wrong." The pain is: **"I cannot trust any citation without manually checking it, which defeats the purpose of using the tool."** Researchers report spending as much time verifying AI citations as they would have spent finding sources manually.

What people want is not "fewer hallucinations" (a percentage). They want **"every citation is guaranteed real, or the agent tells me it couldn't find one."** The binary trust problem. Currently, the agent emits plausible-looking URLs with confidence regardless of whether they exist. Users want the agent to either prove the citation is real or admit it doesn't have one.

### 2. Stop summarizing — actually synthesize

The "insight gap" complaint. From Medium's Data Science Collective: "Deep research tools tend to be weak at proving absence: 'I found no evidence' may simply mean the tool did not search the right database." From AI Goes to College: "Deep research is great at compiling a lot of material quickly and comprehensively, but on its own, it's not great at making true insights."

The user pain: **the agent reads 20 sources and produces a 5-page summary that says nothing I couldn't have figured out by skim-reading the abstracts.** It compresses; it doesn't connect. A good human researcher reading the same 20 sources would say "Source A contradicts Source B, and here's why B is more trustworthy because of C." The agent never does this.

What people want is **cross-source reasoning** — the agent noticing contradictions, weighing source reliability, identifying the consensus vs. the outliers. Not "here are 20 summaries glued together." Nobody is delivering this well.

### 3. Remember what I already know — don't start from scratch every time

The persistent memory complaint, especially from researchers and domain experts. From the Reddit thread on persistent memory: "Our team is running into a wall with our agent setup." From Letta's positioning: "persistent agents with the ability to continuously learn and adapt from their own experience." From Hermes Agent: "remembers what you read, tracks findings, monitors sources, builds a persistent knowledge base across every session."

The user pain: **every research session starts from zero.** If I researched "decentralized training" last week, today's session doesn't know that. I have to re-explain my context, re-establish my level, re-state what I already know. The agent treats me like a stranger every time.

What people want is **an agent that accumulates context across sessions** — knows what I've already researched, knows my level, knows my preferences, builds on previous sessions instead of restarting. This is the personalization question, and the answer is clearly "yes, people want this." The products that deliver it (Letta, Hermes) are growing fast.

### 4. Let me steer mid-research — not just one-shot ask-and-wait

The turn-taking complaint. Every current research agent works the same way: you ask, it thinks for 5 minutes, it returns a report. You can't interrupt. You can't say "wait, go deeper on section 3." You can't say "actually, I meant X, not Y" mid-stream. You get the output, then you start over.

The user pain: **research is iterative, but the tools are batch-mode.** A human researcher would never write a 5-page report without checking in with you partway through. But every AI tool forces you to wait for the full output before you can redirect.

What people want is **a research agent that pauses and checks in** — "here's my plan, ok?" "I've found 3 sources, want me to go deeper on any?" "I'm seeing a contradiction, how do you want me to handle it?" Real-time steering, not batch processing. Almost nobody is doing this. LiquidText does it badly. NotebookLM doesn't do it at all.

### 5. Prove absence — don't just say "I found no evidence"

The most subtle but most valuable complaint. From Information Today (July 2026): "Deep research tools tend to be weak at proving absence: 'I found no evidence' may simply mean the tool did not search the right database."

The user pain: **when the agent finds nothing, you don't know if there's actually nothing or if the agent just failed to search properly.** A human researcher who comes back empty-handed can tell you *where* they looked, *what* they searched for, and *why* they think the absence is real. The agent just says "no results found" and you have to trust it.

What people want is **a research agent that documents its negative results** — "I searched these 12 databases, with these queries, and found nothing matching. Here's my search trail. The absence is real (or: I might have missed X, here's why)." Nobody is doing this.

---

## The taxonomy: personalized vs. research vs. something else

You asked the right question. Let me lay out the actual landscape of what people want, based on the complaints:

### Type A: "Research agent" (general-purpose, one-shot)
- What it does: you ask a question, it researches, it returns a report.
- Examples: Perplexity Deep Research, ChatGPT Deep Research, Gemini Deep Research.
- What users say: "Good for quick overviews. Useless for serious work because of hallucinations and shallow synthesis."
- Saturation: crowded. Big tech owns this. Hard to compete.

### Type B: "Personalized research agent" (remembers you, accumulates context)
- What it does: learns your domain, remembers what you've researched, builds on past sessions.
- Examples: Letta, Hermes Agent, OpenClaw's persistent knowledge layer.
- What users say: "This is what I actually want. The one-shot tools are useless to me as a domain expert because they don't know what I already know."
- Saturation: emerging. Letta and Hermes are growing fast but not dominant. Real opportunity.

### Type C: "Research collaborator" (turn-taking, steerable, iterative)
- What it does: researches *with* you. Pauses, checks in, lets you redirect. Asks clarifying questions. Surfaces contradictions mid-research.
- Examples: nobody is doing this well. LiquidText attempts it (badly). NotebookLM doesn't try.
- What users say: "I want to steer mid-research, not wait for a 5-page report I have to redo."
- Saturation: empty. This is the gap.

### Type D: "Experimental research agent" (runs experiments, not just reads)
- What it does: reads papers, writes code, runs experiments, verifies claims.
- Examples: AlphaEvolve, AutoResearch (Karpathy's inspiration), research reproduction agents.
- What users say: "I don't want another summarizer. I want an agent that tests claims."
- Saturation: research frontier. Not a consumer product yet. Hard to build.

### Type E: "Domain expert agent" (narrow, deep, verified)
- What it does: specializes in one domain (legal, medical, financial, code). Verifies every claim against domain-specific sources.
- Examples: Elicit (academic), Consensus (scientific), Scite (citations). Legal AI tools.
- What users say: "General tools hallucinate too much. I need one that knows my domain cold."
- Saturation: fragmented by domain. Each domain has 1-2 players, none dominant.

---

## What's actually underserved (the honest mapping)

The complaints cluster into a clear pattern. People want **all five of these things at once:**

1. Verified citations (not hallucinated)
2. Real synthesis (not summarization)
3. Persistent memory (not from-scratch)
4. Iterative steering (not batch)
5. Documented absence (not "no results found")

No product delivers more than 1-2 of these. The gap isn't "build a better research agent." The gap is **"build a research agent that does all five."** That's the underserved space.

The closest existing products:
- Letta / Hermes: deliver #3 (memory). Miss the others.
- Perplexity: delivers #1 partially (citations, but hallucinated). Misses the rest.
- Elicit: delivers #5 partially (academic provenance). Misses the rest.
- LiquidText: attempts #4 (steering). Does it badly.

Nobody combines them. That's the opening.

---

## The three real product directions (not mutually exclusive)

### Direction 1: "The verified research agent"
Bets on #1 + #5. Every citation is verified (URL resolves, content matches claim). Every negative result is documented (search trail). The agent never hallucinates because it can't — it only emits claims it has grounded.

- Hardest part: verification at scale. Checking every URL, every claim, every source. Either slow or expensive.
- Differentiator: trust. Researchers will pay for this.
- Competes with: Elicit, Consensus, Scite. But broader (not just academic).

### Direction 2: "The research collaborator"
Bets on #3 + #4. The agent is turn-taking, iterative, remembers you. You have a multi-turn research conversation. It pauses, checks in, lets you redirect. It accumulates context across sessions.

- Hardest part: the turn-taking UX. Most agent frameworks are batch-mode. Building real-time steering is a systems problem, not just an ML problem.
- Differentiator: feels like working with a human researcher. The "alive" feeling.
- Competes with: nobody, really. LiquidText is the closest and it's not close.

### Direction 3: "The accumulating expert"
Bets on #2 + #3. The agent becomes a domain expert over time. Each research session adds to its knowledge. After 50 sessions on decentralized AI, it knows more than Claude. The synthesis improves because the agent has deep context.

- Hardest part: continual learning without hallucination amplification. (This is exactly what avr-cl + the verifier address.)
- Differentiator: the agent gets better the more you use it. Every other tool is static.
- Competes with: Letta, Hermes. But with actual weight-level learning, not just memory.

---

## The pattern across all three

Every direction converges on the same primitives:

- **A verifier** (for #1, #5) — checks claims against sources, documents search trails.
- **A memory layer** (for #2, #3) — accumulates verified claims with provenance.
- **A turn-taking harness** (for #4) — pauses, checks in, lets you steer.
- **A continual learning loop** (for #2, #3) — folds verified findings into weights, retention-gated.

These are the same four components you've been building toward. The market is asking for exactly the primitives you've been developing. The question is which framing to lead with.

---

## My read on what to build

The "research collaborator" framing (Direction 2) is the most differentiated and the hardest for incumbents to copy. Perplexity and ChatGPT are batch-mode by design — their architecture is ask-and-wait. Building turn-taking into them would require rearchitecting their entire product. A startup that leads with "research is iterative, our agent is too" has a clear wedge.

But the "accumulating expert" (Direction 3) is the one that connects to your existing work (avr-cl, continual-pt) and to Prime Intellect's thesis (small models that learn). It's also the one where the technology is hardest — which means if you ship it, it's hardest to copy.

The strongest play might be **Direction 2 + Direction 3 combined**: a turn-taking research agent that accumulates domain expertise across sessions. The turn-taking makes it usable; the accumulation makes it valuable over time. Each solves a weakness of the other:
- Turn-taking alone = cool UX, but the agent doesn't get smarter. Users churn after the novelty.
- Accumulation alone = valuable long-term, but the UX is batch-mode. Users don't stick around long enough to see the accumulation.

Together: the turn-taking keeps users engaged session-over-session, and the accumulation makes each session better than the last. That's a retention loop, not just a feature.

---

## Open questions for you

1. **Which direction hits hardest for you?** The verified agent, the collaborator, or the accumulating expert? Or some combination?

2. **Who's the user?** A researcher, a student, a developer, a domain expert, yourself? The product shape changes drastically based on this.

3. **What's the smallest version that would be useful to you, personally, this month?** Not the X post version — the version you'd actually use.
