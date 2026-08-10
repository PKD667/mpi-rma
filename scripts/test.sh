#!/bin/sh
# mpi-rma test suite: cargo checks, then the test examples under mpirun.
# Usage: mpi-rma/scripts/test.sh
set -eu
ROOT=$(CDPATH= cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

cargo test
cargo build --examples

export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

# One rank per core is not available on a workstation running everything else.
MPIRUN="mpirun --map-by :OVERSUBSCRIBE"
# prterun resolves the executable itself, so its path must be absolute.
TARGET=$(cargo metadata --no-deps --format-version 1 |
    python3 -c 'import json,sys; print(json.load(sys.stdin)["target_directory"])')

for spec in "2 test_window" "2 test_ring" "3 test_layout"; do
    set -- $spec
    echo "[test] mpirun -n $1 $2"
    $MPIRUN -n "$1" "$TARGET/debug/examples/$2"
done

# The soak is the real correctness statement: sustained all-to-all traffic with
# self-describing payloads, verified message by message. Release, because the
# race it is looking for needs the senders to actually outrun the receivers.
cargo build --release --example soak
for spec in "2 safe 0" "4 safe 0" "4 safe 64" "5 raw 0" "4 raw 64"; do
    set -- $spec
    echo "[test] soak -n $1 $2 (noise ${3} KiB)"
    $MPIRUN -n "$1" "$TARGET/release/examples/soak" "$2" 20000 256 8 "$3" >/dev/null
done
