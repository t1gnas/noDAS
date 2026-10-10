import numpy as np


def fmt(v: object) -> str:
    if isinstance(v, (float, np.floating)):
        return "nan" if np.isnan(v) else f"{v:.3f}"
    return str(v)


def format_table(rows: list[list], header: list[str]) -> str:
    cells = [[fmt(v) for v in r] for r in rows]
    widths = [max([len(h), *(len(r[i]) for r in cells)]) for i, h in enumerate(header)]
    lines = ["  ".join(h.ljust(w) for h, w in zip(header, widths)), "  ".join("-" * w for w in widths)]
    return "\n".join(lines + ["  ".join(v.ljust(w) for v, w in zip(r, widths)) for r in cells])


def print_table(rows: list[list], header: list[str]) -> None:
    print(format_table(rows, header))


def print_title(text: str) -> None:
    print(f"\n== {text} ==")


def print_confusion(cm: np.ndarray, names: list[str]) -> None:
    print("rows = true, columns = predicted")
    rows = [[names[i], *cm[i].tolist()] for i in range(len(names)) if cm[i].sum() > 0]
    print_table(rows, ["true\\pred", *[n[:6] for n in names]])


def print_cell_report(rep: dict, names: list[str]) -> None:
    rows = [[names[i], rep["precision"][i], rep["recall"][i], rep["f1"][i], int(rep["support"][i])] for i in range(len(names)) if rep["support"][i] > 0]
    print_table(rows, ["class", "precision", "recall", "f1", "support"])
    print(f"macro-F1 {rep['macro_f1']:.3f}  accuracy {rep['accuracy']:.3f}  cells {int(rep['support'].sum())}")


def event_row(e: dict) -> list:
    return [f"{e['t_start_s']:.2f}", f"{e['t_end_s']:.2f}", f"{e['pos_start_m']:.1f}", f"{e['pos_end_m']:.1f}", f"{e['c0']}-{e['c1']}", e["class_name"], f"{e['confidence']:.2f}", e["cells"]]


def events_table(events: list[dict]) -> str:
    header = ["t_start_s", "t_end_s", "pos_start_m", "pos_end_m", "channels", "class", "conf", "cells"]
    return format_table([event_row(e) for e in events], header) + f"\n{len(events)} events"


def print_events(events: list[dict]) -> None:
    print(events_table(events))


def print_limits() -> None:
    print("\nKnown limits:")
    print("- Test partitions share recordings with training; scores are within-session. `loro` is the honest check.")
    print("- Background comes from a single recording, so false-alarm numbers are indicative only.")
    print("- Features are 6 bands at 0.1 s; no phase or velocity information.")
    print("- Classes listed in config scope.exclude_classes are never predicted and never scored; reference numbers above include them.")
