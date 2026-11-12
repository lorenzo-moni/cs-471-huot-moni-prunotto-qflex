"""
Plot, for each host thread, how its time is split, as stacked 100% bars.

Input: the JSON written by `qstats_analyze.py --json out.json`.

With one vCPU per host thread (N:N) host thread = vCPU.

Usage:
analysis/qstats_plot_breakdown.py analysis.json [-o fig.png|fig.pdf]
                                  [--n-server 4] [--labels S0,S1,...] [--title TEXT]
"""

import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXEC_COLOR = "#2878d6"
WAIT_COLOR = "#eb6834"
SWITCH_COLOR = "#6250d6"
BQL_COLOR = "#8c8c8c"
GRID_COLOR = "#e5e5e5"
MIN_LABEL_PCT = 5  # no % label inside segments thinner than this
MIN_PCT = 0.5  # draw the BQL segment only if some thread reaches this


def load_records(
    path: str, key: str, idx: str, fields: set[str]
) -> tuple[dict, list[dict]]:
    """Returns the summary and the records of `key` sorted by `idx`."""
    with open(path) as f:
        data = json.load(f)
    records = data.get(key)
    if not records:
        sys.exit(f"Error: no '{key}' in {path}")
    missing = ({idx} | fields) - set(records[0])
    if missing:
        sys.exit(f"Error: {key} records in {path} lack {sorted(missing)}")
    return data.get("summary", {}), sorted(records, key=lambda r: r[idx])


def wall_series(
    path: str,
) -> tuple[list[int], list[tuple[str, str, list[float]]], str]:
    _, rows = load_records(
        path,
        "per_thread",
        "tid",
        {"exec_frac", "barrier_frac", "bql_frac", "ctx_switch_frac"},
    )
    series = [
        ("Execution", EXEC_COLOR, [100 * r["exec_frac"] for r in rows]),
        ("Barrier wait", WAIT_COLOR, [100 * r["barrier_frac"] for r in rows]),
    ]
    context_switch = [100 * r["ctx_switch_frac"] for r in rows]
    if max(context_switch) >= MIN_PCT:
        series.append(("Context Switch", SWITCH_COLOR, bql))
    bql = [100 * r["bql_frac"] for r in rows]
    if max(bql) >= MIN_PCT:
        series.append(("BQL wait", BQL_COLOR, bql))
    return [r["tid"] for r in rows], series, "% of wall-clock time"


def make_labels(idx: list[int], n_server: int | None, labels: str | None) -> list[str]:
    n = len(idx)
    if labels:
        out = [s.strip() for s in labels.split(",")]
        if len(out) != n:
            sys.exit(f"Error: --labels has {len(out)} entries, but there are {n} bars")
        return out
    if n_server is not None:
        if not 0 <= n_server <= n:
            sys.exit(f"Error: --n-server {n_server} is not in [0, {n}]")
        return [f"S{i}" for i in range(n_server)] + [
            f"C{i}" for i in range(n - n_server)
        ]
    return [str(i) for i in idx]


def plot(
    labels: list[str],
    series: list[tuple[str, str, list[float]]],
    ylabel: str,
    out: str,
    title: str | None,
) -> None:
    """Stacks the series bottom-up; the % label goes in the second segment (the wait)."""
    plt.rcParams.update({"font.family": "serif", "font.size": 8})
    fig, ax = plt.subplots(figsize=(3.4, 1.85))
    x = list(range(len(labels)))
    bar = dict(width=0.7, edgecolor="white", linewidth=1, zorder=2)
    bottom = [0.0] * len(labels)
    for i, (name, color, values) in enumerate(series):
        ax.bar(x, values, bottom=bottom, color=color, label=name, **bar)
        if i == 1:
            for xi, (b, v) in enumerate(zip(bottom, values)):
                if v >= MIN_LABEL_PCT:
                    ax.text(
                        xi,
                        b + v / 2,
                        f"{v:.0f}%",
                        ha="center",
                        va="center",
                        color="white",
                        fontsize=7,
                        zorder=3,
                    )
        bottom = [b + v for b, v in zip(bottom, values)]

    ax.set_xticks(x, labels)
    ax.tick_params(length=0)
    ax.set_ylim(0, 100)
    ax.set_yticks(range(0, 101, 25))
    ax.set_ylabel(ylabel)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=len(series),
        frameon=False,
        handlelength=2.5,
        columnspacing=3 if len(series) == 2 else 1.5,
    )
    if title:
        ax.set_title(title, pad=18)

    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def handle_arguments() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("json", help="output of qstats_analyze.py --json")
    ap.add_argument(
        "-o",
        "--output",
        help="output figure, format from the extension "
        "(default: <json stem>_wall.png or <json stem>_sim.png)",
    )
    ap.add_argument(
        "--n-server",
        type=int,
        help="label the first N bars S0.. (server) and the others C0.. (client)",
    )
    ap.add_argument("--labels", help="comma-separated x labels, one per bar")
    ap.add_argument("--title", help="figure title")
    return ap.parse_args()


def main():
    args = handle_arguments()
    series_fn = wall_series
    idx, series, ylabel = series_fn(args.json)
    labels = make_labels(idx, args.n_server, args.labels)

    out = args.output or f"{os.path.splitext(args.json)[0]}_{args.basis}.png"
    plot(labels, series, ylabel, out, args.title)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
