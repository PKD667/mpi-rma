#!/bin/sh
# Build and run mpi-rma smoke for benching locally.
# Usage: local.sh [ranks=2] [iters=1000] [max_kib=1024]
set -eu
ROOT=$(CDPATH= cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"
N=${1:-2}
ITERS=${2:-1000}
MAX_KIB=${3:-1024}
cargo build --release -p mpi-rma --examples
mpirun -n "$N" target/release/examples/smoke
mpirun -n "$N" target/release/examples/bench "$ITERS" "$MAX_KIB"
