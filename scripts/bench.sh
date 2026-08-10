#!/bin/sh
# Collect every measurement the figures are drawn from, into TSVs under data/.
#
#   mpi-rma/scripts/bench.sh [ranks=8] [repeats=11] [outdir=data]
#   mpi-rma/scripts/bench.sh 24 11 data
#
# Files are the cache: a measurement whose file exists is skipped. Delete the
# file to redo that measurement. 
# Each measurement writes to a staging file first and is moved in only when the run completes
# A failed run is retried three times. T
#
# Each file makes a point: 
#    + compare.tsv   ring against plain MPI p2p, quiet and under noise
#    + loss.tsv      raw-mode loss against ring depth and offered rate
#    + scale.tsv     safe-mode delivery as rank count grows
set -eu
ROOT=$(CDPATH= cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
N=${1:-8}
REPEATS=${2:-11}
OUT=${3:-data}
mkdir -p "$OUT"

export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
# One rank per core is not available on a workstation running everything else.
MPIRUN="mpirun --map-by :OVERSUBSCRIBE"
# prterun resolves the executable itself, so its path must be absolute.
TARGET=$(cargo metadata --no-deps --format-version 1 |
    python3 -c 'import json,sys; print(json.load(sys.stdin)["target_directory"])')

cargo build --release --examples

# Drop the header of a block joining a file that already has one
append() {
    if [ -s "$1" ]; then tail -n +2 "$2"; else cat "$2"; fi >>"$1"
}

# prterun occasionally splits a rank's last write into a short trailing line.
# Reject a staging block whose data rows do not all match the header's column
# count, and let run() retry it.
aligned() {
    want=$(awk -F'\t' 'NR==1 {print NF; exit}' "$1")
    awk -F'\t' -v want="$want" 'NR>1 && NF != want {bad=1} END {exit bad}' "$1"
}

# Run one measurement, retrying up to three times, appending its rows only
# after a complete run. Output goes to a staging file first so an interrupted
# run cannot leave a truncated TSV behind.
run() {
    name=$1
    file=$2
    shift 2
    try=0
    while [ $try -lt 3 ]; do
        if $MPIRUN "$@" >"$OUT/$name.staging" 2>/dev/null &&
            aligned "$OUT/$name.staging"; then
            append "$OUT/$file" "$OUT/$name.staging"
            rm -f "$OUT/$name.staging"
            return
        fi
        try=$((try + 1))
        echo "[bench] retry $try of $name after failure"
    done
    echo "[bench] FAILED: $name" >&2
    exit 1
}

if [ ! -s "$OUT/compare.tsv" ]; then
    echo "[bench] compare: ring vs p2p, quiet, $REPEATS repeats"
    run compare-quiet compare.tsv -n 2 \
        "$TARGET"/release/examples/compare 20000 32 0 "$REPEATS"
    if [ "$N" -ge 4 ]; then
        echo "[bench] compare: ring vs p2p, under $((N - 2)) noise ranks"
        run compare-noisy compare.tsv -n "$N" \
            "$TARGET"/release/examples/compare 20000 32 64 "$REPEATS"
    fi
fi

# Raw mode drops when a sender laps the receiver. 
# Ring depth buys burst headroom and pacing removes the pressure
# the two only separate when swept together, so sweep the grid rather than one axis at a time. 
#Five repeats: each is a full depth x pace grid, and the figure aggregates over the noise dimension too.

# The sweep must run at a width whose receiver can actually drain: the poll
# passes over every incoming lane, so the per-lane drain rate falls as ranks
# grow, and a wide sweep saturates at every pace. Eight ranks keep the receiver
# ahead of the slowest offered rate while staying honest about a shared node.
if [ ! -s "$OUT/loss.tsv" ]; then
    LOSS=$N
    if [ "$LOSS" -gt 8 ]; then LOSS=8; fi
    for rep in 0 1 2 3 4; do
        for depth in 2 8 32 128; do
            for pace in 0 500 1000 2000 5000 10000 20000 50000; do
                echo "[bench] loss: rep $rep, depth $depth, pace ${pace}ns"
                run "loss-r$rep-d$depth-p$pace" loss.tsv -n "$LOSS" \
                    "$TARGET"/release/examples/soak raw 20000 256 "$depth" 0 "$pace" "$rep"
            done
        done
        for depth in 2 8 32 128; do
            echo "[bench] loss: rep $rep, depth $depth, noisy"
            run "loss-r$rep-d$depth-n" loss.tsv -n "$LOSS" \
                "$TARGET"/release/examples/soak raw 20000 256 "$depth" 64 0 "$rep"
        done
    done
fi

# Safe mode must stay lossless at every width the machine can host, quiet or not.
if [ ! -s "$OUT/scale.tsv" ]; then
    for rep in 0 1 2 3 4; do
        r=2
        while :; do
            echo "[bench] scale: rep $rep, safe, $r ranks"
            run "scale-r$rep-$r" scale.tsv -n "$r" \
                "$TARGET"/release/examples/soak safe 20000 256 8 0 0 "$rep"
            echo "[bench] scale: rep $rep, safe, $r ranks, noisy"
            run "scale-r$rep-$r-n" scale.tsv -n "$r" \
                "$TARGET"/release/examples/soak safe 20000 256 8 64 0 "$rep"
            if [ "$r" -eq "$N" ]; then break; fi
            r=$((r * 2))
            if [ "$r" -gt "$N" ]; then r=$N; fi
        done
    done
fi

echo "[bench] wrote $OUT/{compare,loss,scale}.tsv"
