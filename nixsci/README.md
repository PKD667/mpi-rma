# Benchmarks as nixsci experiments

Three experiments run the benchmarks in `examples/`. `compare` runs the ring against MPI point-to-point. `loss` sweeps raw-mode loss over ring depth and offered rate. `scale` checks that safe mode stays lossless as ranks grow.

```sh
nix run .#compare        # one run, recorded and sealed under .nixsci/runs
git add lab.lock         # the lock names the run by hash
nix build .#perf         # mpi-rma-perf.svg and summary.csv
nix build .#tables       # loss.csv and scale.csv
```

The runs in `lab.lock` were taken on an AMD Ryzen 5 PRO 4650U laptop with 12 threads and Open MPI 5.0.11, at 12 ranks. Each run records its host, so the figure names the machine it came from. `BENCHMARKS.md` still holds the earlier dual-Xeon numbers, taken with `scripts/bench.sh`. Those are not reproduced here and carry no sealed record.

Nix builds the example binaries and the analysis environment. A run is clean when it measured the current closure. Timing is not reproducible byte for byte, so a re-measurement gives new numbers with the same provenance. On this machine the ring wins at 512 B and 4 KiB, ties at 64 B and loses at 32 KiB and 256 KiB. One node says nothing about a network.
