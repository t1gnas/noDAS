import argparse
import sys
from pathlib import Path

from das import audit, cnn, lgbm, loro, stream
from das.config import Config, data_dir, load_config
from das.evaluate import compare_rows, evaluate, print_eval
from das.predict import MODELS, class_names, scan
from das.report import print_events, print_limits, print_table, print_title


def cmd_inspect(args: argparse.Namespace, cfg: Config) -> None:
    audit.run(Path(args.path) if args.path else data_dir(cfg))


def cmd_train(args: argparse.Namespace, cfg: Config) -> None:
    if args.model == "lgbm":
        lgbm.fit(cfg)
    else:
        cnn.fit(cfg, args.epochs)


def cmd_eval(args: argparse.Namespace, cfg: Config) -> None:
    print_eval(args.model, evaluate(args.model, cfg), class_names(cfg), args.coarse)


def try_evaluate(name: str, cfg: Config) -> dict | None:
    try:
        return evaluate(name, cfg)
    except SystemExit as e:
        print(e)
        return None


def cmd_compare(args: argparse.Namespace, cfg: Config) -> None:
    summaries = {m: s for m in MODELS if (s := try_evaluate(m, cfg)) is not None}
    assert summaries, "no trained models found; run `train lgbm` and `train cnn` first"
    print_title("official test split: models vs reference (quick per-cell LightGBM)")
    print_table(compare_rows(summaries, class_names(cfg)), ["metric", *summaries, "reference"])
    print("reference LORO recall: car 0.64, fence 0.73, walk 0.50, longboard 0.50 (run `loro` for this model's numbers)")
    print_limits()


def cmd_loro(args: argparse.Namespace, cfg: Config) -> None:
    loro.run_all(args.model, cfg, args.hold)


def cmd_predict(args: argparse.Namespace, cfg: Config) -> None:
    print_events(scan(args.file, args.model, cfg, args.t0, args.t1, args.threshold))


def cmd_stream(args: argparse.Namespace, cfg: Config) -> None:
    stream.replay(args.file, args.model, cfg, args.speed, args.t0, args.t1)


def add_window(s: argparse.ArgumentParser) -> None:
    s.add_argument("file")
    s.add_argument("--model", choices=MODELS, default="cnn")
    s.add_argument("--t0", type=float, default=None, help="start time in seconds")
    s.add_argument("--t1", type=float, default=None, help="end time in seconds")


def add_simple(sub: argparse._SubParsersAction) -> None:
    s = sub.add_parser("inspect", help="dataset audit and consistency checks")
    s.add_argument("path", nargs="?")
    s.set_defaults(fn=cmd_inspect)
    s = sub.add_parser("train", help="train and save a model")
    s.add_argument("model", choices=MODELS)
    s.add_argument("--epochs", type=int, default=None)
    s.set_defaults(fn=cmd_train)
    s = sub.add_parser("compare", help="eval both models side by side with reference numbers")
    s.set_defaults(fn=cmd_compare)


def add_model_cmds(sub: argparse._SubParsersAction) -> None:
    s = sub.add_parser("eval", help="official test split metrics")
    s.add_argument("--model", choices=MODELS, required=True)
    s.add_argument("--coarse", action="store_true")
    s.set_defaults(fn=cmd_eval)
    s = sub.add_parser("loro", help="leave-one-recording-out evaluation")
    s.add_argument("--model", choices=MODELS, required=True)
    s.add_argument("--hold", default=None, help="recording id to hold out (default: all multi-recording classes)")
    s.set_defaults(fn=cmd_loro)


def add_file_cmds(sub: argparse._SubParsersAction) -> None:
    s = sub.add_parser("predict", help="scan one v2 .h5 file and print events")
    add_window(s)
    s.add_argument("--threshold", type=float, default=None)
    s.set_defaults(fn=cmd_predict)
    s = sub.add_parser("stream", help="paced replay with live alerts")
    add_window(s)
    s.add_argument("--speed", type=float, default=5.0, help="replay speed factor, 0 = as fast as possible")
    s.set_defaults(fn=cmd_stream)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="das", description="DAS event classifier v2")
    p.add_argument("--config", default="config.yaml")
    sub = p.add_subparsers(dest="command", required=True)
    add_simple(sub)
    add_model_cmds(sub)
    add_file_cmds(sub)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        args.fn(args, load_config(args.config))
    except AssertionError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)
