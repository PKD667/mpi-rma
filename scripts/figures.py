#!/usr/bin/env python3
"""Draw the figures and summary tables from the TSVs written by bench.sh.

    python3 mpi-rma/scripts/figures.py [datadir=data]

Needs only the standard library plus matplotlib. Repeats are reduced to a
median with a percentile bootstrap 95% confidence interval (2000 draws, seed
stable per cell); the raw measurements stay in the TSVs. Writes
figures/*.{svg,png}, plus summary.tsv, loss-summary.tsv and tables.md next to
the data.
"""

import csv
import random
import statistics as st
import sys
import zlib
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib.ticker import FuncFormatter, NullFormatter

# Categorical slots, assigned in fixed order and never cycled. Validated for
# colour-vision deficiency as a four-slot set against a light surface.
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#dcdcd8"
SURFACE = "#fcfcfb"

TRANSPORTS = [
    ("ring-safe", BLUE),
    ("ring-raw", ORANGE),
    ("p2p-send", AQUA),
    ("p2p-bsend", YELLOW),
]
DEPTHS = [(2, BLUE), (8, ORANGE), (32, AQUA), (128, YELLOW)]

DRAWS = 2000


def read(path):
    if not path.is_file() or path.stat().st_size == 0:
        raise SystemExit(f"{path}: no measurements yet (run scripts/bench.sh)")
    with open(path) as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    skim = [r for r in rows if None not in r.values()]
    if len(skim) < len(rows):
        print(f"warning: {path}: dropped {len(rows) - len(skim)} partial row(s)")
    return skim


def bytesize(n):
    for unit in ("B", "KiB", "MiB"):
        if n < 1024 or unit == "MiB":
            return f"{n:g} {unit}"
        n /= 1024


def fmt(v):
    """Readable without noise: three significant figures, commas past 1000."""
    if abs(v) >= 1000:
        return f"{v:,.0f}"
    return f"{v:.3g}"


def median_ci(key, values):
    """Median with a percentile bootstrap 95% CI. The seed is a stable hash of
    the cell key, so a run always reports the same interval."""
    seed = zlib.crc32(key.encode()) & 0xFFFFFFFF
    rng = random.Random(seed)
    boots = sorted(st.median(rng.choices(values, k=len(values))) for _ in range(DRAWS))
    lo, hi = boots[int(0.025 * DRAWS) - 1], boots[int(0.975 * DRAWS) - 1]
    return st.median(values), lo, hi


def compare_cells(rows):
    """Per-cell repeat lists for every measure in the compare data.

    Keys are (measure, noise_kib, transport, payload). One aggregation shared
    by the figures and the summary writers, so they cannot disagree.
    """
    agg = defaultdict(list)
    for r in rows:
        noise, transport, payload = int(r["noise_kib"]), r["transport"], int(r["payload"])
        if r["measure"] == "stream":
            agg[("goodput", noise, transport, payload)].append(float(r["goodput_MiB_per_s"]))
            agg[("goodput_s", noise, transport, payload)].append(float(r["goodput_per_s"]))
            agg[("inject_s", noise, transport, payload)].append(float(r["inject_per_s"]))
            agg[("delivered_pct", noise, transport, payload)].append(
                100 * int(r["delivered"]) / max(int(r["sent"]), 1)
            )
        else:
            agg[("latency", noise, transport, payload)].append(float(r["us_per_msg"]))
            if "max_us_per_msg" in r:
                agg[("latency_max", noise, transport, payload)].append(
                    float(r["max_us_per_msg"])
                )
    return agg


def loss_cells(rows):
    """Per-cell repeat lists for the loss data, loss summed across ranks.

    Keys are (depth, pace_ns, noise_kib). Every repeat of every rank lands in
    one cell, so a slow run shows up as a worse point rather than a wider CI.
    """
    runs = defaultdict(lambda: [0, 0])
    for r in rows:
        key = (int(r["depth"]), int(r["pace_ns"]), int(r["noise_kib"]), int(r["rep"]))
        runs[key][0] += int(r["lost"])
        runs[key][1] += int(r["sent"])
    cells = defaultdict(list)
    for (depth, pace, noise, _), (lost, sent) in runs.items():
        cells[(depth, pace, noise)].append(100 * lost / sent)
    return cells


def scale_cells(rows):
    """Per-width repeat lists for the scale data, summed across ranks.

    Keys are (ranks, noise_kib). Delivered per cent of everything sent, plus
    the worst ack-gate spin per message as a slowdown indicator.
    """
    runs = defaultdict(lambda: [0, 0, 0])
    for r in rows:
        key = (int(r["ranks"]), int(r["noise_kib"]), int(r["rep"]))
        runs[key][0] += int(r["sent"])
        runs[key][1] += int(r["lost"])
        runs[key][2] = max(runs[key][2], int(r["waits"]))
    cells = defaultdict(list)
    waits = defaultdict(int)
    for (ranks, noise, _), (sent, lost, spins) in runs.items():
        cells[(ranks, noise)].append(100 * (sent - lost) / sent)
        waits[(ranks, noise)] = max(waits[(ranks, noise)], spins)
    return cells, waits


def style(ax):
    """Recessive axes: the data is the ink, the frame is not."""
    ax.set_facecolor(SURFACE)
    ax.grid(True, which="major", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, which="both", length=3)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color(MUTED)


def label_ends(ax, entries, gap=0.055):
    """Direct labels at the right end of each series, pushed apart to fit.

    Every series carries a visible label and not colour alone. Converging
    series would stack their labels, so place them in axes fraction and
    enforce a minimum separation.

    `entries` is a list of (x, y, text, colour) in data coordinates.
    """
    if not entries:
        return
    lo, hi = ax.get_ylim()
    log = ax.get_yscale() == "log"

    def frac(y):
        import math

        if log:
            return (math.log10(y) - math.log10(lo)) / (math.log10(hi) - math.log10(lo))
        return (y - lo) / (hi - lo)

    placed = sorted(((frac(y), x, text, colour) for x, y, text, colour in entries))
    for i in range(1, len(placed)):
        if placed[i][0] - placed[i - 1][0] < gap:
            f, x, text, colour = placed[i]
            placed[i] = (placed[i - 1][0] + gap, x, text, colour)
    blend = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    for f, x, text, colour in placed:
        ax.annotate(
            text,
            xy=(x, min(f, 0.99)),
            xycoords=blend,
            xytext=(7, 0),
            textcoords="offset points",
            color=colour,
            fontsize=8,
            fontweight="bold",
            va="center",
            annotation_clip=False,
        )


def series(ax, xs, cells, colour, dashed=False):
    """One transport's median curve with its CI as error bars.

    `cells` maps each x to (median, ci_lo, ci_hi). Returns the series' end
    point for label_ends.
    """
    ys = [cells[x][0] for x in xs]
    neg = [y - lo for y, (_, lo, _) in zip(ys, (cells[x] for x in xs))]
    pos = [hi - y for y, (_, _, hi) in zip(ys, (cells[x] for x in xs))]
    ax.errorbar(
        xs,
        ys,
        yerr=[neg, pos],
        color=colour,
        linewidth=2,
        linestyle="--" if dashed else "-",
        marker="o",
        markersize=5,
        markeredgecolor=SURFACE,
        markeredgewidth=1.2,
        capsize=2,
        capthick=1,
        elinewidth=1,
        ecolor=colour,
        zorder=3,
    )
    return xs[-1], ys[-1]


def log_axes(ax, payloads, xmax=3.6):
    """Shared log-axis setup for the payload sweeps."""
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(payloads)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: bytesize(v)))
    # Decades only: the minor labels on a log axis crowd out the data.
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_xlabel("payload", color=MUTED, fontsize=9)
    ax.set_xlim(payloads[0] / 1.6, payloads[-1] * xmax)


def compare_figure(rows, out):
    """Ring against plain MPI point-to-point, throughput and latency."""
    agg = compare_cells(rows)
    has_max = "max_us_per_msg" in rows[0]
    payloads = sorted({int(r["payload"]) for r in rows})
    noises = sorted({int(r["noise_kib"]) for r in rows})
    panels = [
        ("goodput", "delivered MiB/s", "Throughput"),
        ("latency", "µs per one-way hop", "Latency, mean"),
    ]
    if has_max:
        panels.append(("latency_max", "µs per one-way hop", "Latency, max"))

    fig, axes = plt.subplots(
        len(noises),
        len(panels),
        figsize=(5.2 * len(panels), 4.2 * len(noises)),
        facecolor=SURFACE,
    )
    axes = axes.reshape(len(noises), len(panels))

    for row, noise in enumerate(noises):
        band = "quiet" if noise == 0 else f"under {noise} KiB background traffic"
        for col, (measure, ylabel, title) in enumerate(panels):
            ax = axes[row][col]
            style(ax)
            ends = []
            for name, colour in TRANSPORTS:
                cells = {
                    p: median_ci(f"{noise} {measure} {name} {p}", agg[(measure, noise, name, p)])
                    for p in payloads
                    if agg[(measure, noise, name, p)]
                }
                xs = sorted(cells)
                if xs:
                    ends.append((*series(ax, xs, cells, colour), name, colour))
            log_axes(ax, payloads)
            ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
            ax.set_title(f"{title}: {band}", color=INK, fontsize=11, fontweight="bold", loc="left")
            label_ends(ax, ends)

    fig.suptitle(
        "Fixed-slot RMA ring against MPI point-to-point",
        color=INK,
        fontsize=13,
        fontweight="bold",
        x=0.01,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, out, "compare")


def raw_figure(rows, out):
    """What giving up the delivery guarantee actually buys.

    A lossy transport has two rates and they are not the same number: what the
    sender got onto the wire, and what the receiver took delivery of.
    """
    agg = compare_cells(rows)
    payloads = sorted({int(r["payload"]) for r in rows})
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 4.4), facecolor=SURFACE)

    # (a) send-side rate against delivered rate.
    style(ax)
    ends = []
    for label, transport, measure, colour in [
        ("ring-raw injected", "ring-raw", "inject_s", BLUE),
        ("ring-raw delivered", "ring-raw", "goodput_s", ORANGE),
        ("ring-safe delivered", "ring-safe", "goodput_s", AQUA),
        ("p2p-send delivered", "p2p-send", "goodput_s", YELLOW),
    ]:
        xs = [p for p in payloads if agg[(measure, 0, transport, p)]]
        if not xs:
            continue
        cells = {
            p: median_ci(f"{label} {p}", agg[(measure, 0, transport, p)]) for p in xs
        }
        ends.append((*series(ax, xs, cells, colour, dashed=measure == "inject_s"), label, colour))
    log_axes(ax, payloads, xmax=5.5)
    ax.set_ylabel("messages per second", color=MUTED, fontsize=9)
    ax.set_title(
        "Raw mode: what the sender achieved vs what arrived",
        color=INK,
        fontsize=11,
        fontweight="bold",
        loc="left",
    )
    label_ends(ax, ends)

    # (b) the fraction that survived.
    style(bx)
    width = 0.36
    for i, (noise, colour, name) in enumerate(
        [(0, BLUE, "quiet"), (64, ORANGE, "with background traffic")]
    ):
        cells = [
            median_ci(f"delivered {noise} {p}", agg[("delivered_pct", noise, "ring-raw", p)])
            if agg[("delivered_pct", noise, "ring-raw", p)]
            else (0, 0, 0)
            for p in payloads
        ]
        xs = [j + (i - 0.5) * width for j in range(len(payloads))]
        bars(bx, xs, cells, colour, name)
        for rect, y in zip(bx.patches[-len(payloads):], [c[0] for c in cells]):
            bx.annotate(
                f"{y:.0f}%",
                xy=(rect.get_x() + rect.get_width() / 2, y + 4),
                ha="center",
                color=MUTED,
                fontsize=8,
            )
    bx.set_xticks(range(len(payloads)))
    bx.set_xticklabels([bytesize(p) for p in payloads])
    bx.set_ylim(0, 118)
    bx.axhline(100, color=GRID, linewidth=1.2, zorder=2)
    bx.annotate(
        "safe mode sits on this line",
        xy=(len(payloads) - 0.5, 102),
        color=MUTED,
        fontsize=8,
        ha="right",
        style="italic",
    )
    bx.set_xlabel("payload", color=MUTED, fontsize=9)
    bx.set_ylabel("of what was sent, delivered (%)", color=MUTED, fontsize=9)
    bx.set_title(
        "Raw mode: the share that arrived",
        color=INK,
        fontsize=11,
        fontweight="bold",
        loc="left",
    )
    # The tall bars are on the left and the 100% annotation is top right, so the
    # only clear space is the middle of the right-hand side.
    bx.legend(
        frameon=False, fontsize=8, labelcolor=MUTED, loc="center right", handlelength=1.6
    )

    fig.suptitle(
        "Raw mode: injection rate and delivered messages",
        color=INK,
        fontsize=13,
        fontweight="bold",
        x=0.01,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, out, "raw")


def bars(ax, xs, cells, colour, label):
    """A bar group with CI error bars; `cells` maps x to (median, lo, hi)."""
    ys = [c[0] for c in cells]
    neg = [y - lo for y, (_, lo, _) in zip(ys, cells)]
    pos = [hi - y for y, (_, _, hi) in zip(ys, cells)]
    ax.bar(
        xs,
        ys,
        width=0.33,
        color=colour,
        label=label,
        zorder=3,
        linewidth=0,
        yerr=[neg, pos],
        capsize=2,
        error_kw=dict(ecolor=colour, elinewidth=1, capthick=1),
    )


def loss_figure(rows, out):
    """Where raw mode stops keeping up, against rate and against depth."""
    cells = loss_cells(rows)
    paces = sorted({p for (_, p, n) in cells if n == 0 and p > 0})
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 4.4), facecolor=SURFACE)

    # (a) loss against offered rate, one line per depth.
    style(ax)
    ends = []
    for depth, colour in DEPTHS:
        xs = [1e9 / p for p in paces if cells[(depth, p, 0)]]
        points = {
            x: median_ci(f"loss d{depth} {p}", cells[(depth, p, 0)])
            for x, p in zip(xs, paces)
            if cells[(depth, p, 0)]
        }
        order = sorted(points)
        ys = [points[x][0] for x in order]
        ends.append(
            (*series(ax, order, points, colour), f"depth {depth}", colour)
        )
    ax.set_xscale("log")
    ax.set_xlabel("offered rate (messages per lane per second)", color=MUTED, fontsize=9)
    ax.set_ylabel("messages lost (%)", color=MUTED, fontsize=9)
    ax.set_ylim(-3, 84)
    ax.set_xlim(min(1e9 / p for p in paces) / 1.5, max(1e9 / p for p in paces) * 4.5)
    label_ends(ax, ends)
    ax.set_title(
        "Raw mode: loss against offered rate",
        color=INK,
        fontsize=11,
        fontweight="bold",
        loc="left",
    )

    # (b) unpaced, quiet against noisy. Two series, so bars read better.
    style(bx)
    depths = [d for d, _ in DEPTHS]
    width = 0.36
    for i, (noise, colour, name) in enumerate(
        [(0, BLUE, "quiet"), (64, ORANGE, "with background traffic")]
    ):
        points = [
            median_ci(f"loss bars {d} {noise}", cells[(d, 0, noise)])
            if cells[(d, 0, noise)]
            else (0, 0, 0)
            for d in depths
        ]
        xs = [j + (i - 0.5) * width for j in range(len(depths))]
        bars(bx, xs, points, colour, name)
        for rect, y in zip(bx.patches[-len(depths):], [p[0] for p in points]):
            bx.annotate(
                f"{y:.0f}%",
                xy=(rect.get_x() + rect.get_width() / 2, y + 3),
                ha="center",
                color=MUTED,
                fontsize=8,
            )
    bx.set_xticks(range(len(depths)))
    bx.set_xticklabels([f"depth {d}" for d in depths])
    bx.set_ylim(0, 88)
    bx.set_xlabel("slots per lane", color=MUTED, fontsize=9)
    bx.set_ylabel("messages lost (%)", color=MUTED, fontsize=9)
    bx.set_title(
        "Raw mode, senders unpaced",
        color=INK,
        fontsize=11,
        fontweight="bold",
        loc="left",
    )
    bx.legend(frameon=False, fontsize=8, labelcolor=MUTED, loc="upper right", handlelength=1.6)

    fig.suptitle(
        "Raw-mode loss by offered rate and lane depth",
        color=INK,
        fontsize=13,
        fontweight="bold",
        x=0.01,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, out, "loss")


def save(fig, out, name):
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "png"):
        path = out / f"{name}.{ext}"
        fig.savefig(path, dpi=160, facecolor=SURFACE, bbox_inches="tight")
        print(f"wrote {path}")
    plt.close(fig)


def rate_label(pace):
    """Offered rate of a paced lane, in a human unit."""
    rate = 1e9 / pace
    if rate >= 1e6:
        return f"{rate / 1e6:.1f} M/s"
    return f"{rate / 1e3:.0f} k/s"


def summary_tsv(rows, out):
    """One row per cell of the compare data: median plus CI, all measures."""
    agg = compare_cells(rows)
    path = out / "summary.tsv"
    with open(path, "w") as f:
        f.write("measure\tnoise\ttransport\tpayload\tn\tmedian\tci_lo\tci_hi\n")
        for measure, noise, transport, payload in sorted(agg):
            values = agg[(measure, noise, transport, payload)]
            med, lo, hi = median_ci(f"{measure} {noise} {transport} {payload}", values)
            f.write(
                f"{measure}\t{noise}\t{transport}\t{payload}\t{len(values)}\t"
                f"{med:.6g}\t{lo:.6g}\t{hi:.6g}\n"
            )
    print(f"wrote {path}")


def loss_summary_tsv(rows, out):
    cells = loss_cells(rows)
    path = out / "loss-summary.tsv"
    with open(path, "w") as f:
        f.write("depth\tpace_ns\tnoise\tn\tmedian\tci_lo\tci_hi\n")
        for depth, pace, noise in sorted(cells):
            med, lo, hi = median_ci(f"loss {depth} {pace} {noise}", cells[(depth, pace, noise)])
            f.write(f"{depth}\t{pace}\t{noise}\t{len(cells[(depth, pace, noise)])}\t{med:.6g}\t{lo:.6g}\t{hi:.6g}\n")
    print(f"wrote {path}")


def tables_md(compare, loss, out):
    """The tables BENCHMARKS.md is built from: medians with CI brackets."""
    agg = compare_cells(compare)
    loss = loss_cells(loss)
    has_max = any(measure == "latency_max" for measure, _, _, _ in agg)
    payloads = sorted({int(r["payload"]) for r in compare})

    def cell(measure, noise, transport, payload):
        values = agg[(measure, noise, transport, payload)]
        if not values:
            return None
        return median_ci(f"{measure} {noise} {transport} {payload}", values)

    def table(label, measure, noise, transports, ratio=True):
        out.write(f"### {label}\n\n")
        heads = transports + (["ring-safe / p2p-send"] if ratio else [])
        out.write("| payload | " + " | ".join(heads) + " |\n")
        out.write("|--------:|" + "----------:|" * len(heads) + "\n")
        for p in payloads:
            cells = []
            for t in transports:
                c = cell(measure, noise, t, p)
                cells.append(f"{fmt(c[0])} [{fmt(c[1])}, {fmt(c[2])}]" if c else "n/a")
            if ratio:
                rs = cell(measure, noise, "ring-safe", p)
                ps = cell(measure, noise, "p2p-send", p)
                cells.append(f"{rs[0] / ps[0]:.2f}x" if rs and ps else "n/a")
            out.write(f"| {bytesize(p)} | " + " | ".join(cells) + " |\n")
        out.write("\n")

    out.write("# Benchmark tables\n\n")
    out.write("Median over repeats with a 95% bootstrap CI in brackets. "
              "Raw measurements: `compare.tsv`, `loss.tsv`, `scale.tsv`.\n\n")

    table("Goodput, quiet, MiB/s delivered", "goodput", 0,
          ["ring-safe", "ring-raw", "p2p-send", "p2p-bsend"])
    table("Goodput, under 64 KiB background traffic, MiB/s delivered", "goodput", 64,
          ["ring-safe", "ring-raw", "p2p-send", "p2p-bsend"])
    table("One-way latency, quiet, µs", "latency", 0,
          ["ring-safe", "ring-raw", "p2p-send"])
    table("One-way latency, under 64 KiB background traffic, µs", "latency", 64,
          ["ring-safe", "ring-raw", "p2p-send"])
    if has_max:
        table("Worst one-way latency, quiet, µs", "latency_max", 0,
              ["ring-safe", "ring-raw", "p2p-send"])
        table("Worst one-way latency, under 64 KiB background traffic, µs", "latency_max", 64,
              ["ring-safe", "ring-raw", "p2p-send"])

    # Raw mode: what the sender achieved vs what arrived.
    out.write("### Raw mode, quiet: send side vs delivery\n\n")
    out.write("| payload | raw inject (msg/s) | raw goodput (msg/s) | delivered (%) | safe goodput (msg/s) |\n")
    out.write("|---|---:|---:|---:|---:|\n")
    for p in payloads:
        cells = []
        for measure, transport in [
            ("inject_s", "ring-raw"),
            ("goodput_s", "ring-raw"),
            ("delivered_pct", "ring-raw"),
            ("goodput_s", "ring-safe"),
        ]:
            c = cell(measure, 0, transport, p)
            cells.append(f"{fmt(c[0])} [{fmt(c[1])}, {fmt(c[2])}]" if c else "n/a")
        out.write(f"| {bytesize(p)} | " + " | ".join(cells) + " |\n")
    out.write("\n")

    # Loss by offered rate, one block per noise level.
    for noise, band in [(0, "quiet"), (64, "under 64 KiB background traffic")]:
        paces = sorted({p for (_, p, n) in loss if n == noise and p > 0})
        out.write(f"### Loss by offered rate, {band} (%)\n\n")
        heads = ["flat out"] + [rate_label(p) for p in paces]
        out.write("| depth | " + " | ".join(heads) + " |\n")
        out.write("|" + "---:|" * (len(heads) + 1) + "\n")
        for depth, _ in DEPTHS:
            cells = []
            for pace in [0] + paces:
                values = loss[(depth, pace, noise)]
                if values:
                    med, lo, hi = median_ci(f"loss table {depth} {pace} {noise}", values)
                    cells.append(f"{fmt(med)} [{fmt(lo)}, {fmt(hi)}]")
                else:
                    cells.append("")
            out.write(f"| depth {depth} | " + " | ".join(cells) + " |\n")
        out.write("\n")
    print(f"wrote {out.name}")


def tables_scale(scale, out):
    """Safe-mode delivery as the ring widens, one block per noise level."""
    cells, waits = scale_cells(scale)
    for noise, band in [(0, "quiet"), (64, "under 64 KiB background traffic")]:
        out.write(f"### Safe mode at width, {band}\n\n")
        out.write("| ranks | delivered (%) | max blocked sends on one rank |\n")
        out.write("|---:|---:|---:|\n")
        for ranks in sorted({k[0] for k in cells if k[1] == noise}):
            values = cells[(ranks, noise)]
            med, lo, hi = median_ci(f"scale table {ranks} {noise}", values)
            out.write(f"| {ranks} | {fmt(med)} [{fmt(lo)}, {fmt(hi)}] | "
                      f"{waits[(ranks, noise)]} |\n")
        out.write("\n")


def main():
    data = Path(sys.argv[1] if len(sys.argv) > 1 else "data")
    out = data / "figures"
    compare = read(data / "compare.tsv")
    loss = read(data / "loss.tsv")
    scale = read(data / "scale.tsv")
    compare_figure(compare, out)
    raw_figure(compare, out)
    loss_figure(loss, out)
    summary_tsv(compare, data)
    loss_summary_tsv(loss, data)
    with open(data / "tables.md", "w") as f:
        tables_md(compare, loss, f)
        tables_scale(scale, f)


if __name__ == "__main__":
    main()
