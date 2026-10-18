"""Run one mpi-rma benchmark and record its rows. The closure's program: `bench.py compare|loss|scale`.

Each case is one `mpirun` of an example binary. The examples print a TSV block from rank 0 (compare)
or one row per rank (soak). A block is recorded only when every row has the header's column count,
because prterun occasionally splits a rank's last write. A failed or misaligned run is retried twice.
"""
import csv, io, json, os, subprocess, sys
from nixsci import lab

EXAMPLES = os.environ["MPI_RMA_EXAMPLES"]
MPIRUN = ["mpirun", "--map-by", ":OVERSUBSCRIBE"]
SCHEMA = json.loads(os.environ["NIX_LAB_SCHEMA"])
CAST = {"str": str, "int": int, "float": float}
P = lab.params()


def block(out):
    rows = list(csv.reader(io.StringIO(out), delimiter="\t"))
    if len(rows) < 2 or any(len(r) != len(rows[0]) for r in rows[1:]):
        return None
    return [dict(zip(rows[0], r)) for r in rows[1:]]


def case(dataset, ranks, exe, *args):
    cols = SCHEMA[dataset]
    for attempt in range(3):
        p = subprocess.run([*MPIRUN, "-n", str(ranks), f"{EXAMPLES}/{exe}", *map(str, args)],
                           capture_output=True, text=True)
        rows = block(p.stdout) if p.returncode == 0 else None
        if rows is not None:
            for row in rows:
                lab.record(dataset, {c: CAST[t.rstrip("?")](row[c]) for c, t in cols.items()})
            return
        print(f"[bench] retry {attempt + 1}: {exe} -n {ranks} {args}", file=sys.stderr)
    raise SystemExit(f"[bench] failed: {exe} -n {ranks} {args}")


def compare():
    n, reps, msgs, depth = P["ranks"], P["repeats"], P["messages"], P["depth"]
    case("compare", 2, "compare", msgs, depth, 0, reps)
    if n >= 4:
        case("compare", n, "compare", msgs, depth, 64, reps)


def loss():
    n, reps, msgs = min(P["ranks"], 8), P["repeats"], P["messages"]
    for rep in range(reps):
        for depth in (2, 8, 32, 128):
            for pace in (0, 500, 1000, 2000, 5000, 10000, 20000, 50000):
                case("soak", n, "soak", "raw", msgs, 256, depth, 0, pace, rep)
        for depth in (2, 8, 32, 128):
            case("soak", n, "soak", "raw", msgs, 256, depth, 64, 0, rep)


def scale():
    top, reps, msgs = P["ranks"], P["repeats"], P["messages"]
    for rep in range(reps):
        r = 2
        while True:
            case("soak", r, "soak", "safe", msgs, 256, 8, 0, 0, rep)
            case("soak", r, "soak", "safe", msgs, 256, 8, 64, 0, rep)
            if r == top:
                break
            r = min(r * 2, top)


{"compare": compare, "loss": loss, "scale": scale}[sys.argv[1]]()
