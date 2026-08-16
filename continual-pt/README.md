# continual-pt

A post-training runtime for a model that is told to learn X.

It researches X itself, creates source-grounded learning experiences from
what it finds, trains temporary fast weights on that experience, and commits
the update only if X improves and nothing previously learned regresses.

## The loop

1. **Absorb** — research X on the web, generate source-grounded practice
2. **Verify** — gate whether X was actually learned (two paths: executable / non-executable)
3. **Retain** — re-check every previously committed item with its original verifier, live, right now. Not PPL. Not any proxy.
4. **Commit** — only if X passes AND nothing regresses

## What's proven

- **avr-cl**: closed-form weight interpolation for retention. 5.8x less forgetting than naive sequential fine-tuning on TRACE benchmark.
- Everything else is a design, not a result.

## The non-executable verifier (the novel piece)

For claims that can't be run as code:

1. **Decompose** — split into sub-claims, each gets its own verdict
2. **Reconstruct** — derive why it would be true (necessary, not sufficient)
3. **Search** — external evidence, not self-narration
4. **Trace lineage** — INDEPENDENT vs DERIVATIVE sources. Only independent counts.
5. **Verdict** — RECONSTRUCTED / OPEN / DISCARD

## The open question

Does retention hold past 2-3 sequential commits when the retention check is
behavioral instead of proxy-based?

## Usage

```bash
# Install
pip install -e .

# Run locally
python -m continual_pt.cli learn --x "LoRA" --output ./results

# Run on Modal (detached)
modal run modal_app.py::run_sequence --x-list '["LoRA", "GRPO"]'
```

## License

MIT
