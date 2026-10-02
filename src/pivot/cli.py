import argparse
import json
from pathlib import Path

from . import data, experiment

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
SEED = 2026
PER_CATEGORY = 200
PILOT_PER_CATEGORY = 5


def items(split: str) -> list[data.Item]:
    data.download(DATA)
    gmmlu = data.gmmlu(DATA)
    zno = data.zno(DATA)
    test = data.stratified(gmmlu, PER_CATEGORY, SEED) + zno
    if split == "test":
        return test
    used = {it.id for it in test}
    return data.stratified([it for it in gmmlu if it.id not in used], PILOT_PER_CATEGORY, SEED)


def cmd_prepare(_):
    test = items("test")
    manifest = {"seed": SEED, "per_category": PER_CATEGORY, "items": [it.id for it in test]}
    (DATA / "sample.json").write_text(json.dumps(manifest, indent=1) + "\n")


def cmd_run(args):
    experiment.run(args.models, items(args.split), RESULTS / f"calls_{args.split}.jsonl")


def cmd_analyze(_):
    from . import analysis

    analysis.main(items("test"), RESULTS)


def main():
    parser = argparse.ArgumentParser(prog="pivot")
    sub = parser.add_subparsers(required=True)
    sub.add_parser("prepare").set_defaults(func=cmd_prepare)
    run = sub.add_parser("run")
    run.add_argument("split", choices=["pilot", "test"])
    run.add_argument("--models", nargs="+", default=experiment.MODELS, choices=experiment.MODELS)
    run.set_defaults(func=cmd_run)
    sub.add_parser("analyze").set_defaults(func=cmd_analyze)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
