# mpi-rma

Typed, safe MPI one-sided communication (RMA: Remote Memory Access) for Rust, as an extension of
[rsmpi](https://docs.rs/mpi) 0.8. 
This crate adds windows over `rsmpi` communicators and datatypes, wrapping the raw `mpi::ffi`
calls contained inside.

## Quick start

```rust
use mpi_rma::traits::*;

let (universe, _) = mpi::initialize_with_threading(mpi::Threading::Multiple)?;
let world = universe.world();

let win = world.allocate_window::<u64>(1024)?;
win.lock_all()?;
win.put(1, 0, &[1, 2, 3])?;      // into rank 1's window, offset 0
let mut buf = [0; 3];
win.get(1, 0, &mut buf)?;        // back from rank 1's window
win.close()?;                    // collective
```

Rust cannot add inherent methods to a foreign type, so the plugin surface is
an extension trait: you just have to import `mpi_rma::traits::*` and the methods we want will appear on the `rsmpi` objects.

## Model

A `Window<T>` is a fixed-size, MPI-allocated region shared by every rank of
one communicator. Elements are sealed to scalar POD types (`u8`..`u64`,
`i8`..`i64`, `usize`, `isize`, `f32`, `f64`).

RMA operations require an access epoch. This crate supports the one a
long-lived transfer window wants: `lock_all()` once, `unlock_all()` at
`close()`. Epoch state is checked on every operation. This is not full sync its alright.

Every `put` and `get` ends with `MPI_Win_flush`: when the call returns, the
operation is complete at the target and the borrowed buffer is reusable.

Window memory is zeroed collectively at allocation, so counters and slots
start deterministic on every rank.

## Noncontiguous targets: `Indexed<T>`

MPI derived datatypes can scatter one contiguous origin buffer into an
arbitrary fixed target layout:

```rust
let scatter = mpi_rma::Indexed::new(&[2, 5, 11], win.len())?;
win.put_indexed(dest, &[a, b, c], &scatter)?;
```

`Indexed::new` validates distinct, in-bounds indices; `put_indexed` compiles
an rsmpi `UserDatatype`, performs one scatter Put, flushes, and frees it.

This is the full extent of "complex datatypes over RMA". A derived datatype
is a static memory map: it cannot compare keys, follow pointers, allocate,
or run target-side logic. 

Unfortunately, it cannot insert into a `BTreeMap` either (no one-sided reservoir)
or any other Rust collection, whose nodes are allocator-owned and whose
invariants only local code may touch. Sorted placement has to be designed
into the layout itself (fixed buckets, per-origin runs) or done by the
receiver after delivery.

## Safety contract

- Construction requires `MPI_THREAD_MULTIPLE`; that is what justifies
  `Window: Send + Sync`.
- Rank, range, count, and epoch checks precede every operation.
- Remotely mutable memory is never exposed as a Rust reference; reads and
  writes go through completed `get`/`put` calls.
- `close()` is the collective free. `Drop` frees too, for symmetric
  programs, but `close` makes the boundary explicit.

## Benchmarking

`examples/bench.rs` measures put/get against p2p send, quiet and under
background traffic. Ranks 0-1 are the measured pair and ranks 2+ pair up for
continuous ping-pong noise.

```bash
scripts/local.sh 4              # build, smoke, bench with 4 local ranks
mpirun -n 4 target/release/examples/bench 1000 1024 16
#   iters=1000  max_msg=1024 KiB  noise_msg=16 KiB
```

### Grid'5000

`scripts/g5k.sh` builds the bench, bundles it with its library closure (no
root, no nix needed on the node), rsyncs it to the frontend, and submits one
OAR job through the shared `build/oar.sh` lifecycle helpers:

```bash
mpi-rma/scripts/g5k.sh 2 0:15                     # 2 nodes, 15 minutes
G5K_HOST=rennes.g5k mpi-rma/scripts/g5k.sh 4 0:30 -t besteffort
```


## Tests

```bash
cargo test -p mpi-rma                                   # unit + singleton MPI
cargo build -p mpi-rma --example smoke
mpirun -n 2 target/debug/examples/smoke                 # 2-rank roundtrip
```
