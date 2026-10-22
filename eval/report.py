"""Charts from an eval results file.

    uv run python -m eval.report                       # newest baseline_*.json
    uv run python -m eval.report eval/results/x.json

Writes PNGs next to the results file:
  <stem>_stt.png     word error rate per condition, raw and normalised
  <stem>_curves.png  normalised WER against SNR and against packet loss (mobile)
  <stem>_task.png    correct / caught / wrong per condition, if the agent stage ran
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# Reference palette (dataviz skill), light mode: slots 1 and 7 for the two WER
# series; correct / caught / wrong use blue, yellow, red so "wrong" reads as the alarm.
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
NORMALISED, RAW = "#2a78d6", "#b9b5ad"
OUTCOME_COLOURS = {"correct": "#2a78d6", "caught": "#eda100", "wrong": "#e34948"}


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def stt_chart(summary: dict, out: Path) -> None:
    ids = list(summary)[::-1]  # first condition at the top
    norm = [summary[c]["normalised"]["wer"] * 100 for c in ids]
    raw = [summary[c]["raw"]["wer"] * 100 for c in ids]
    y = range(len(ids))
    fig, ax = plt.subplots(figsize=(7.5, 0.42 * len(ids) + 1.4), facecolor=SURFACE)
    _style(ax)
    h = 0.38
    ax.barh([i + h / 2 for i in y], raw, height=h - 0.04, color=RAW, label="raw")
    ax.barh([i - h / 2 for i in y], norm, height=h - 0.04, color=NORMALISED, label="normalised (what the agent sees)")
    for i, v in zip(y, norm, strict=True):
        ax.text(v + 0.8, i - h / 2, f"{v:.0f}%", va="center", fontsize=8, color=INK)
    ax.set_yticks(list(y), ids, color=INK)
    ax.set_xlabel("word error rate, %", color=MUTED, fontsize=9)
    ax.set_title("Prisma word error rate by channel condition", loc="left", color=INK, fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc="upper right", labelcolor=MUTED)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def curves_chart(summary: dict, conditions: dict, out: Path) -> None:
    def wer(cid: str) -> float:
        return summary[cid]["normalised"]["wer"] * 100

    snr_ids = [c for c in ("mob", "mob_snr20", "mob_snr10", "mob_snr5", "mob_snr0") if c in summary]
    loss_ids = [c for c in ("mob", "mob_loss1", "mob_loss5", "mob_loss10") if c in summary]
    fig, (left, right) = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True, facecolor=SURFACE)
    for ax in (left, right):
        _style(ax)
        ax.grid(axis="y", color=GRID, linewidth=0.8)

    snr_labels = ["none" if c == "mob" else f"{conditions[c]['snr_db']:g}" for c in snr_ids]
    left.plot(range(len(snr_ids)), [wer(c) for c in snr_ids], color=NORMALISED, linewidth=2, marker="o", markersize=7)
    left.set_xticks(range(len(snr_ids)), snr_labels)
    left.set_xlabel("shop noise, SNR dB (louder noise to the right)", color=MUTED, fontsize=9)
    left.set_ylabel("normalised WER, %", color=MUTED, fontsize=9)
    left.set_title("Noise", loc="left", color=INK, fontsize=10)

    loss_x = [conditions[c]["loss_pct"] for c in loss_ids]
    right.plot(loss_x, [wer(c) for c in loss_ids], color=NORMALISED, linewidth=2, marker="o", markersize=7)
    right.set_xticks(loss_x, [f"{x:g}%" for x in loss_x])
    right.set_xlabel("packet loss, bursty", color=MUTED, fontsize=9)
    right.set_title("Packet loss", loc="left", color=INK, fontsize=10)

    for ax, ids, xs in ((left, snr_ids, range(len(snr_ids))), (right, loss_ids, loss_x)):
        for x, c in zip(xs, ids, strict=True):
            ax.annotate(f"{wer(c):.0f}%", (x, wer(c)), textcoords="offset points", xytext=(0, 8),
                        ha="center", fontsize=8, color=INK)
    left.set_ylim(bottom=0)
    fig.suptitle("Mobile call (GSM + G.711): word error rate as the line gets worse", x=0.01, ha="left",
                 color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def task_chart(summary: dict, out: Path) -> bool:
    ids = [c for c in summary if "task" in summary[c]][::-1]
    if not ids:
        return False
    fig, ax = plt.subplots(figsize=(7.5, 0.42 * len(ids) + 1.4), facecolor=SURFACE)
    _style(ax)
    left = [0.0] * len(ids)
    for outcome, colour in OUTCOME_COLOURS.items():
        share = [summary[c]["task"][outcome] / summary[c]["task"]["n"] * 100 for c in ids]
        ax.barh(range(len(ids)), share, left=left, color=colour, height=0.7, label=outcome,
                edgecolor=SURFACE, linewidth=2)
        if outcome == "correct":
            for i, v in enumerate(share):
                ax.text(v - 1.5, i, f"{v:.0f}%", va="center", ha="right", fontsize=8, color="white")
        left = [a + b for a, b in zip(left, share, strict=True)]
    ax.set_yticks(range(len(ids)), ids, color=INK)
    ax.set_xlim(0, 100)
    ax.set_xlabel("share of utterances, %", color=MUTED, fontsize=9)
    ax.set_title("Task outcome by condition: right, re-asked, or wrong", loc="left", color=INK, fontsize=11)
    ax.legend(frameon=False, fontsize=8, ncols=3, loc="upper left", bbox_to_anchor=(0, -0.12), labelcolor=MUTED)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", nargs="?", help="results JSON (default: newest baseline_*.json)")
    args = parser.parse_args()
    if args.results:
        path = Path(args.results)
    else:
        found = sorted(RESULTS_DIR.glob("baseline_*.json"))
        if not found:
            print("No baseline results yet; run eval.run_eval first.", file=sys.stderr)
            return 1
        path = found[-1]
    data = json.loads(path.read_text(encoding="utf-8"))
    summary, conditions = data["summary"], data["conditions"]
    stem = path.with_suffix("")
    written = [stem.with_name(stem.name + "_stt.png")]
    stt_chart(summary, written[0])
    curves = stem.with_name(stem.name + "_curves.png")
    curves_chart(summary, conditions, curves)
    written.append(curves)
    task = stem.with_name(stem.name + "_task.png")
    if task_chart(summary, task):
        written.append(task)
    for w in written:
        print(f"wrote {w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
