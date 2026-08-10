# Design

`Ring` is a sparse set of directed fixed-slot lanes over MPI RMA. The raw path has no publication
counter, acknowledgement, or receiver-side MPI call.

## Layout

Construction receives the active lanes as `(source, destination, depth, capacity)` tuples in a
slice. Each rank passes the same list, without ordering. The `depth` defines the number of slots a
lane holds, `capacity` the payload bytes one slot fits:

```rust
//    (0 -> 1 depth 8 capacity 64 KiB) & (2 -> 1 depth 32 capacity 1024 B)
let ring = Ring::safe(comm, &[(0, 1, 8, 64 * 1024), (2, 1, 32, 1024)])?;
```

Every destination exposes only its incoming lanes. They are packed contiguously into one
`Window<u8>`, so ranks with no incoming lane allocate zero bytes. A slot is:

```text
[sequence: u64]
[length: u64]
[crc32: u32]
[payload: u8; capacity]
[guard: u64]                 guard = !sequence
```

Sequence zero is unused. The checksum covers `sequence`, `length`, and the first `length` payload
bytes. Padding past `length` is not covered and is not zeroed: nothing reads it, because `length`
bounds what a receiver copies out. The duplicated, complemented sequence guards the two ends of the
image.

A sender always Puts the whole image, including that padding. The guard sits at a fixed offset from
the end of the slot, so a receiver can find it without first trusting `length`. Writing only the
used prefix would leave the guard unwritten. One Put covering the whole slot is what makes the tail
guard meaningful.

The image is built in a per-lane scratch buffer that lives as long as the ring, not in a fresh
allocation per send. Only the header, the payload and the guard are rewritten. On a wide lane the
allocation and zero-fill of a slot-sized buffer cost more than the transfer itself.

Safe rings allocate one additional `Window<u64>`. It contains one cumulative acknowledgement per
active outgoing lane, packed at the source. Raw rings allocate no counter window.

## Construction

All ranks sort and all-gather the lane list and mode before allocating a window. A mismatch fails on
every rank before any collective allocation starts. Window element types must match, but local
lengths may differ.

Local polling without `MPI_Win_sync` requires MPI's unified memory model. Ring construction checks
`MPI_WIN_MODEL` once and rejects a separate-model window. A checksum can reject torn bytes, but it
cannot make a stale private copy become visible. This is the one place where the MPI memory model
constrains the design.

Both windows enter one passive-target `MPI_Win_lock_all` epoch at construction and leave it at
`close` or `Drop`.

## Raw Send

For sequence `s`, the sender chooses slot `(s - 1) % depth`, builds the complete fixed-size image,
and performs:

```text
MPI_Put(image)
MPI_Win_flush(destination)
```

There is no second Put and no publication operation. The sender then advances only its local
sequence state.

## Poll

Polling does not call MPI. For each incoming lane, the receiver reads the expected slot's sequence
and guard from its own window memory.

If they name the expected sequence, it snapshots the slot and accepts it only when the guard,
length, and checksum all agree. A failed snapshot is retried on a later poll.

If the slot contains a later valid sequence, the receiver has been lapped. In a raw ring it scans
every slot, validates each snapshot, sorts the retained sequences, returns them in increasing order,
and counts the gaps as lost. `seen` only moves forward.

A safe ring cannot be lapped, so it fails instead of recovering. The sender cannot reach that slot
again until the expected sequence has been acknowledged, and an acknowledgement only follows
consumption; reaching the lapped branch means the counter or the window has been corrupted. Falling
through to the raw path there would quietly turn a guaranteed lane into a lossy one, so `poll`
returns `Error::Lapped`.

The sequence guard makes a torn header or footer overwhelmingly likely to fail immediately. CRC32
catches mixed interiors. Neither changes MPI's rule that overlapping Put and local access has
undefined contents. Raw mode is deliberately best effort: validation turns observed tearing into a
retry or an erasure rather than accepting malformed data.

## Safe Mode

Safe mode uses the same slot image, send, and poll path. It adds only overwrite gating.

Before reusing a slot, the sender reads its lane's cumulative ACK from local unified window memory.
If `sequence - acknowledged > depth`, it yields until the receiver advances the ACK. The receiver
advances it with one atomic `MPI_Fetch_and_op` after consuming messages. The sender checks that the
counter never regresses or exceeds what it sent.

The gate is exactly tight. Sequence `s` occupies slot `(s - 1) mod depth`, whose previous occupant
is `s - depth`, so `s - acknowledged <= depth` is precisely `acknowledged >= s - depth`. An
acknowledgement is only issued after `poll` has copied the frame out, so the slot is free when the
sender is let through, with no slack in either direction.

The cached counter is refreshed lazily: it only moves when a send finds the gate shut, which is
roughly once per `depth` sends. A gate check therefore re-reads before it concludes anything, and
`waits` counts senders that blocked after refreshing rather than senders that found a stale cache.

`send` blocks indefinitely when a peer stops acknowledging. That is the backpressure contract, not
an oversight: there is no deadline, no cancellation, and a peer that has stopped draining will hang
its senders. A caller that needs to survive a dead peer has to notice out of band.

No slot is overwritten before consumption and wee keep async-ness.

## Costs

For a directed lane `source -> destination`:

```text
slot bytes = 28 + capacity
destination bytes = depth * slot bytes
safe source bytes += 8
```

Total storage is proportional to active directed worker connections, not all worker pairs. Raw send
costs one Put and one flush. Raw poll costs local loads only. Safe mode adds one atomic operation per
`ack` call and waits only when a sender would overwrite unread data.

A send always moves `slot` bytes, whatever the payload, so a lane whose capacity greatly exceeds its
typical message pays for the difference. Size capacity to the messages actually sent.

---

*We measured some numbers on our hardware and reported them in `BENCHMARKS.md`. Against MPI point-to-point the ring wins below roughly 8 KiB payload and loses above it. For Safe vs Raw, we see that depth stops reducing raw-mode loss once the offered rate passes what the receiver can drain.*
