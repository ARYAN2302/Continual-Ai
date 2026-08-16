"""
CLI entry point for continual-pt.

Usage:
    continual-pt learn --x "LoRA" --verifier executable --output ./results
    continual-pt sequence --config sequence.json
"""

import argparse
import json
import sys
from continual_pt import Config, VerifierType


def main():
    parser = argparse.ArgumentParser(description="continual-pt: post-training runtime")
    subparsers = parser.add_subparsers(dest="command")

    # learn command
    learn_parser = subparsers.add_parser("learn", help="Learn a single X")
    learn_parser.add_argument("--x", required=True, help="What to learn")
    learn_parser.add_argument("--verifier", default="auto",
                             choices=["executable", "non_executable", "auto"])
    learn_parser.add_argument("--model", default="LiquidAI/LFM2.5-2.6B")
    learn_parser.add_argument("--output", default="./results")
    learn_parser.add_argument("--claims", type=str, default=None,
                             help="JSON file with list of claims (for executable)")
    learn_parser.add_argument("--claim", type=str, default=None,
                             help="Single claim string (for non-executable)")

    # sequence command
    seq_parser = subparsers.add_parser("sequence", help="Learn a sequence of X's")
    seq_parser.add_argument("--config", required=True,
                           help="JSON file with sequence spec")
    seq_parser.add_argument("--model", default="LiquidAI/LFM2.5-2.6B")
    seq_parser.add_argument("--output", default="./results")
    seq_parser.add_argument("--resume", default=None,
                           help="Path to existing adapter to resume from")

    args = parser.parse_args()

    if args.command == "learn":
        config = Config(model_id=args.model, output_dir=args.output)

        claims = None
        if args.claims:
            with open(args.claims) as f:
                claims = json.load(f)

        from continual_pt.runtime import learn_x, learn_sequence
        x_spec = [{
            "x": args.x,
            "verifier_type": args.verifier,
            "claims": claims,
            "claim": args.claim,
        }]
        learn_sequence(x_spec, config)

    elif args.command == "sequence":
        config = Config(model_id=args.model, output_dir=args.output)

        with open(args.config) as f:
            x_list = json.load(f)

        from continual_pt.runtime import learn_sequence
        learn_sequence(x_list, config, adapter_path=args.resume)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
