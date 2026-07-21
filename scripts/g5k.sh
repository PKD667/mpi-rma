#!/bin/sh
# Ship the RMA bench to Grid'5000 and run it inside one OAR job.
#
#   mpi-rma/scripts/g5k.sh [nodes=2] [walltime=0:15] [extra oarsub args...]
#
# Env: G5K_HOST (frontend), G5K_DIR (remote dir, default ~/mpi-rma-bench),
#      G5K_SSH (extra ssh options). Requires: rsync, patchelf, ldd.
set -eu

ROOT=$(CDPATH= cd "$(dirname "$0")/../.." && pwd)
# shellcheck source=../../build/oar.sh
. "$ROOT/build/oar.sh"

NODES=${1:-2}
WALLTIME=${2:-0:15}
[ $# -ge 2 ] && shift 2 || shift $#

G5K_DIR=${G5K_DIR:-mpi-rma-bench}
STAGE="$ROOT/mpi-rma/target/g5k"

echo "[g5k] building bench"
cd "$ROOT"
cargo build --release -p mpi-rma --example bench

echo "[g5k] bundling binary + lib closure"
rm -rf "$STAGE"
mkdir -p "$STAGE/bin" "$STAGE/lib"
cp target/release/examples/bench "$STAGE/bin/"
ldd "$STAGE/bin/bench" | while IFS= read -r line; do
	line=${line#"${line%%[![:space:]]*}"}
	case "$line" in
	*" => "*) so=${line#* => }; so=${so%% *} ;;
	/*) so=${line%% *} ;;
	*) continue ;;
	esac
	[ -f "$so" ] || continue
	cp -L "$so" "$STAGE/lib/$(basename "$so")" 2>/dev/null || :
done
LOADER=$(basename "$(ldd "$STAGE/bin/bench" | sed -n 's|.*\(/ld-linux[^ ]*\).*|\1|p')")
cp -L "$(ldd "$STAGE/bin/bench" | sed -n 's|.*\(/ld-linux[^ ]*\).*|\1|p')" "$STAGE/lib/$LOADER"
for f in "$STAGE/lib/"*; do
	patchelf --set-rpath '$ORIGIN' "$f" 2>/dev/null || :
done
patchelf --set-interpreter '$ORIGIN/../lib/'"$LOADER" --set-rpath '$ORIGIN/../lib' "$STAGE/bin/bench"

cat >"$STAGE/run.sh" <<'EOF'
#!/bin/sh
# Runs on the job's first node. One proc per node (uniq nodefile).
set -eu
DIR=$(CDPATH= cd "$(dirname "$0")" && pwd)
LOADER=$(ls "$DIR"/lib/ld-linux*)
sort -u "$OAR_NODEFILE" >"$DIR/nodes"
RANKS=$(wc -l <"$DIR/nodes")
echo "[g5k] $RANKS ranks on: $(tr '\n' ' ' <"$DIR/nodes")"
OMPI_MCA_plm_rsh_agent=oarsh mpirun -n "$RANKS" -machinefile "$DIR/nodes" \
	"$DIR/lib/$(basename "$LOADER")" --library-path "$DIR/lib" \
	"$DIR/bin/bench" "${ITERS:-1000}" "${MAX_KIB:-1024}"
EOF
chmod +x "$STAGE/run.sh"

echo "[g5k] syncing to $G5K_HOST:$G5K_DIR"
# shellcheck disable=SC2086
rsync -aq --delete $G5K_SSH "$STAGE/" "$G5K_HOST:$G5K_DIR/"

echo "[g5k] submitting job ($NODES nodes, $WALLTIME)"
oar_run -l "nodes=$NODES,walltime=$WALLTIME" "$@" -- sh "$G5K_DIR/run.sh"
