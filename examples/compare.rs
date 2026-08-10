// Ring transport against plain MPI point-to-point, over a payload sweep.
//
//   mpirun -n <2+> compare [messages=20000] [depth=32] [noise_kib=0] [repeats=5]
//
// Ranks 0 and 1 are the measured pair; ranks 2+ generate background ping-pong
// when `noise_kib > 0`, which requires at least four ranks. Each transport is
// measured two ways:
//
//   stream   rank 0 pushes a fixed message count as hard as it can, rank 1
//            drains. The count is capped by total bytes, so a large payload does
//            not have to move 20k of them.
//
//            Two rates come out of this and a lossy transport separates them.
//            `inject` is what the sender achieved, measured until its last send
//            returned; `goodput` is what the receiver actually took delivery of.
//            For safe mode and p2p the two agree and the sender pays in stall
//            time (`wait_s`). For raw mode the sender never stalls and pays in
//            messages that never arrive. Reporting only one of the two would
//            flatter whichever mode is being looked at.
//   rtt      one message out, one back, repeated. Half the round trip is the
//            one-way latency of an unloaded lane.
//
// Transports:
//
//   ring-safe   Ring::safe, acknowledged, no loss
//   ring-raw    Ring::raw, no acknowledgement, overwrites unread slots
//   p2p-send    MPI_Send / MPI_Recv
//   p2p-bsend   MPI_Bsend / MPI_Recv, the closest p2p analogue to a ring send
//               in that the sender does not wait for the receiver
//
// One TSV row per (transport, payload, measurement, repeat) on stdout, from rank 0.
// Repeats are emitted rather than averaged: a microbenchmark on a shared machine
// has a long right tail, and the aggregation belongs to whatever reads the file.
// The rtt rows carry `us_per_msg`, the mean one-way latency, and
// `max_us_per_msg`, the worst single one-way hop of the run.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use mpi::Threading;
use mpi::collective::CommunicatorCollectives;
use mpi::point_to_point::{Destination, Source};
use mpi::topology::{Communicator, Group, Rank, SimpleCommunicator};

use mpi_rma::Ring;

const TAG_DATA: i32 = 1;
const TAG_STOP: i32 = 2;
const TAG_NOISE: i32 = 11;
const TAG_QUIT: i32 = 12;

const SIZES: [usize; 5] = [64, 512, 4096, 32768, 262_144];
/// Ceiling on bytes moved by one stream row.
const STREAM_BYTES: u64 = 64 << 20;
/// Attached buffer for `MPI_Bsend`, comfortably past `STREAM_BYTES`.
const BSEND_BUF: usize = 512 << 20;

/// Carries a communicator reference across a thread spawn. rsmpi wraps a raw
/// handle, so its communicators are neither `Send` nor `Sync`; under
/// `MPI_THREAD_MULTIPLE` the concurrent use here is legal.
struct Shared<'a>(&'a SimpleCommunicator);
unsafe impl Send for Shared<'_> {}

const HEADER: &str = "transport\tmeasure\tranks\tdepth\tnoise_kib\trep\tpayload\
\tsent\tdelivered\tinject_s\ttotal_s\twait_s\tinject_per_s\tgoodput_per_s\
\tgoodput_MiB_per_s\tus_per_msg\tlost\tmax_us_per_msg";

/// One measurement, as it lands in the TSV.
struct Row {
    transport: &'static str,
    measure: &'static str,
    payload: usize,
    /// What the sender put on the wire.
    sent: u64,
    /// What the receiver took delivery of. Below `sent` only in raw mode.
    delivered: u64,
    /// Seconds until the sender's last send returned.
    inject: f64,
    /// Seconds until everything had drained at the receiver.
    total: f64,
    /// Seconds the sender spent blocked on acknowledgements.
    wait: f64,
    /// Worst single one-way hop, microseconds (rtt rows only).
    max_us: f64,
}

impl Row {
    fn print(&self, ranks: i32, depth: usize, noise_kib: usize, rep: u32) {
        let inject = self.sent as f64 / self.inject;
        let goodput = self.delivered as f64 / self.total;
        println!(
            "{}\t{}\t{ranks}\t{depth}\t{noise_kib}\t{rep}\t{}\t{}\t{}\
             \t{:.6}\t{:.6}\t{:.6}\t{inject:.1}\t{goodput:.1}\t{:.4}\t{:.3}\t{}\t{:.3}",
            self.transport,
            self.measure,
            self.payload,
            self.sent,
            self.delivered,
            self.inject,
            self.total,
            self.wait,
            goodput * self.payload as f64 / (1 << 20) as f64,
            self.total * 1e6 / self.delivered.max(1) as f64,
            self.sent - self.delivered,
            self.max_us,
        );
    }
}

/// Deterministic per-repeat payload: a xorshift64 Fisher-Yates walk over a
/// fixed fill. Consecutive repeats never move the same byte pattern, so L3
/// and prefetcher effects do not alias across repeats, and the walk is a
/// function of `(rep, payload)` alone, so the measurement is reproducible.
fn shuffled(rep: u32, len: usize) -> Vec<u8> {
    let mut buf = vec![0xC3u8; len];
    let mut x = (rep as u64 + 1) | 1;
    let mut rnd = || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        x
    };
    for i in (1..buf.len()).rev() {
        buf.swap(i, (rnd() % (i as u64 + 1)) as usize);
    }
    buf
}

// ── Ring ────────────────────────────────────────────────────────────────────

/// Rank 0 pushes `n` messages flat out; rank 1 drains and reports arrivals.
///
/// Returns `(sent, delivered, inject, total, wait)`. `inject` stops when the
/// last send returns and `total` when the receiver confirms it has drained, so
/// a mode that lets its sender run ahead is visible as a gap between them.
fn ring_stream(
    comm: &SimpleCommunicator,
    ring: &Ring,
    rank: Rank,
    rep: u32,
    payload: usize,
    n: u64,
) -> (u64, u64, f64, f64, f64) {
    comm.barrier();
    let stalled = ring.wait_ns();
    let t0 = Instant::now();
    if rank == 0 {
        let buf = shuffled(rep, payload);
        let mut sent = 0u64;
        while sent < n {
            ring.send(1, &buf).unwrap();
            sent += 1;
        }
        let inject = t0.elapsed().as_secs_f64();
        let wait = (ring.wait_ns() - stalled) as f64 * 1e-9;
        // Tell the drain side to stop, then collect what it actually saw.
        comm.process_at_rank(1).send_with_tag(&[0u8], TAG_STOP);
        let (reply, _) = comm
            .process_at_rank(1)
            .receive_vec_with_tag::<u64>(TAG_STOP);
        (sent, reply[0], inject, t0.elapsed().as_secs_f64(), wait)
    } else {
        let mut got = 0u64;
        let mut stopping = false;
        loop {
            let batch = ring.poll().unwrap();
            let empty = batch.is_empty();
            let mut furthest = 0u64;
            for m in &batch {
                got += 1;
                furthest = furthest.max(m.sequence);
            }
            if furthest > 0 {
                ring.ack(0, furthest).unwrap();
            }
            if stopping && empty {
                break;
            }
            if !stopping
                && comm
                    .any_process()
                    .immediate_probe_with_tag(TAG_STOP)
                    .is_some()
            {
                comm.any_process().receive_vec_with_tag::<u8>(TAG_STOP);
                stopping = true;
            }
            if empty {
                std::thread::yield_now();
            }
        }
        comm.process_at_rank(0).send_with_tag(&[got][..], TAG_STOP);
        let dt = t0.elapsed().as_secs_f64();
        (0, got, dt, dt, 0.0)
    }
}

/// One message each way, `iters` times, on the ring.
///
/// Returns `(total, max_us)`: the run wall time and the worst single
/// round trip, halved to a one-way latency so it shares the `us_per_msg`
/// unit of the mean.
fn ring_rtt(ring: &Ring, rank: Rank, rep: u32, payload: usize, iters: u64) -> (f64, f64) {
    let buf = shuffled(rep, payload);
    let other: Rank = if rank == 0 { 1 } else { 0 };
    let mut seen = 0u64;
    let await_one = |ring: &Ring, seen: &mut u64| {
        loop {
            let batch = ring.poll().unwrap();
            if let Some(last) = batch.last() {
                *seen = last.sequence;
                ring.ack(other, *seen).unwrap();
                return;
            }
            std::hint::spin_loop();
        }
    };
    let mut max_us = 0.0f64;
    let t0 = Instant::now();
    for _ in 0..iters {
        let start = Instant::now();
        if rank == 0 {
            ring.send(1, &buf).unwrap();
            await_one(ring, &mut seen);
        } else {
            await_one(ring, &mut seen);
            ring.send(0, &buf).unwrap();
        }
        max_us = max_us.max(start.elapsed().as_secs_f64() * 5e5);
    }
    (t0.elapsed().as_secs_f64(), max_us)
}

// ── Point to point ──────────────────────────────────────────────────────────

fn p2p_stream(
    comm: &SimpleCommunicator,
    rank: Rank,
    rep: u32,
    payload: usize,
    n: u64,
    buffered: bool,
) -> (u64, u64, f64, f64) {
    comm.barrier();
    let t0 = Instant::now();
    if rank == 0 {
        let buf = shuffled(rep, payload);
        let peer = comm.process_at_rank(1);
        let mut sent = 0u64;
        while sent < n {
            if buffered {
                peer.buffered_send_with_tag(&buf[..], TAG_DATA);
            } else {
                peer.send_with_tag(&buf[..], TAG_DATA);
            }
            sent += 1;
        }
        let inject = t0.elapsed().as_secs_f64();
        peer.send_with_tag(&[0u8], TAG_STOP);
        // The receiver's count is authoritative and equals the sender's here:
        // p2p does not drop. Waiting for it also drains the queue before the
        // next measurement starts.
        let (reply, _) = comm
            .process_at_rank(1)
            .receive_vec_with_tag::<u64>(TAG_STOP);
        (sent, reply[0], inject, t0.elapsed().as_secs_f64())
    } else {
        let mut got = 0u64;
        let source = comm.process_at_rank(0);
        loop {
            let (_, status) = source.receive_vec::<u8>();
            if status.tag() == TAG_STOP {
                break;
            }
            got += 1;
        }
        comm.process_at_rank(0).send_with_tag(&[got][..], TAG_STOP);
        let dt = t0.elapsed().as_secs_f64();
        (got, got, dt, dt)
    }
}

fn p2p_rtt(
    comm: &SimpleCommunicator,
    rank: Rank,
    rep: u32,
    payload: usize,
    iters: u64,
) -> (f64, f64) {
    let buf = shuffled(rep, payload);
    let peer = comm.process_at_rank(if rank == 0 { 1 } else { 0 });
    let mut max_us = 0.0f64;
    let t0 = Instant::now();
    for _ in 0..iters {
        let start = Instant::now();
        if rank == 0 {
            peer.send_with_tag(&buf[..], TAG_DATA);
            peer.receive_vec_with_tag::<u8>(TAG_DATA);
        } else {
            peer.receive_vec_with_tag::<u8>(TAG_DATA);
            peer.send_with_tag(&buf[..], TAG_DATA);
        }
        max_us = max_us.max(start.elapsed().as_secs_f64() * 5e5);
    }
    (t0.elapsed().as_secs_f64(), max_us)
}

/// Background ping-pong among ranks 2+, in pairs, until `stop`.
fn noise(world: &SimpleCommunicator, kib: usize, stop: &AtomicBool) {
    let rank = world.rank();
    let base = rank - 2;
    let partner = if base % 2 == 0 { rank + 1 } else { rank - 1 };
    if partner >= world.size() {
        return;
    }
    let buf = vec![0xA5u8; kib * 1024];
    let peer = world.process_at_rank(partner);
    loop {
        if rank < partner {
            if stop.load(Ordering::Relaxed) {
                peer.send_with_tag(&[0u8], TAG_QUIT);
                return;
            }
            peer.send_with_tag(&buf[..], TAG_NOISE);
            peer.receive_vec_with_tag::<u8>(TAG_NOISE);
        } else {
            let (_, status) = peer.receive_vec::<u8>();
            if status.tag() == TAG_QUIT {
                return;
            }
            peer.send_with_tag(&buf[..], TAG_NOISE);
        }
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let messages: u64 = args.next().and_then(|s| s.parse().ok()).unwrap_or(20_000);
    let depth: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(32);
    let noise_kib: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(0);
    let repeats: u32 = args.next().and_then(|s| s.parse().ok()).unwrap_or(5);

    let (mut universe, provided) =
        mpi::initialize_with_threading(Threading::Multiple).expect("MPI must initialize once");
    assert_eq!(provided, Threading::Multiple);
    let world = universe.world();
    let rank = world.rank();
    let size = world.size();
    assert!(size >= 2, "compare needs at least 2 ranks");
    assert!(noise_kib == 0 || size >= 4, "noise needs at least 4 ranks");
    // `p2p-bsend` needs somewhere to copy to. Only rank 0 sends buffered, so
    // only it attaches the buffer; every rank allocating 512 MiB would need
    // tens of GiB at scale. Sized past the largest amount a stream can leave
    // in flight, so the comparison measures the transport and not a buffer
    // running out.
    if rank == 0 {
        universe.set_buffer_size(BSEND_BUF);
    }

    let stop = Arc::new(AtomicBool::new(false));
    let noisy = noise_kib > 0;

    std::thread::scope(|scope| {
        if noisy && rank >= 2 {
            let stop = Arc::clone(&stop);
            let shared = Shared(&world);
            scope.spawn(move || {
                let shared = shared;
                noise(shared.0, noise_kib, &stop);
            });
        }
        if rank > 1 {
            // Noise ranks take no part in the measurement. They wait here for
            // rank 0 to call time, then let their noise thread finish.
            world
                .process_at_rank(0)
                .receive_vec_with_tag::<u8>(TAG_QUIT);
            stop.store(true, Ordering::Relaxed);
            return;
        }

        if rank == 0 {
            println!("{HEADER}");
        }

        // The measured pair only. A world barrier here would deadlock against
        // ranks sitting in the noise loop.
        let pair: Vec<Rank> = vec![0, 1];
        let group = world.group().include(&pair);
        let duo = world
            .split_by_subgroup(&group)
            .expect("ranks 0 and 1 form the measured pair");

        for rep in 0..repeats {
            for payload in SIZES {
                // Both ring modes share the lane shape, so the only difference
                // between the two rows is the acknowledgement gate.
                let lanes = vec![
                    (0 as Rank, 1 as Rank, depth, payload),
                    (1, 0, depth, payload),
                ];
                // Cap the stream by bytes as well as count: 256 KiB x 20k would be
                // 5 GiB through the wire for one row.
                let n = messages.min((STREAM_BYTES / payload as u64).max(64));
                let iters = if payload >= 32768 { 500 } else { 2000 };

                for (name, safe) in [("ring-safe", true), ("ring-raw", false)] {
                    let ring = if safe {
                        Ring::safe(&duo, &lanes).unwrap()
                    } else {
                        Ring::raw(&duo, &lanes).unwrap()
                    };
                    let (sent, got, inject, total, wait) =
                        ring_stream(&duo, &ring, rank, rep, payload, n);
                    if rank == 0 {
                        Row {
                            transport: name,
                            measure: "stream",
                            payload,
                            sent,
                            delivered: got,
                            inject,
                            total,
                            wait,
                            max_us: 0.0,
                        }
                        .print(size, depth, noise_kib, rep);
                    }
                    duo.barrier();
                    let stalled = ring.wait_ns();
                    let (dt, max_us) = ring_rtt(&ring, rank, rep, payload, iters);
                    if rank == 0 {
                        Row {
                            transport: name,
                            measure: "rtt",
                            payload,
                            sent: iters * 2,
                            delivered: iters * 2,
                            inject: dt,
                            total: dt,
                            wait: (ring.wait_ns() - stalled) as f64 * 1e-9,
                            max_us,
                        }
                        .print(size, depth, noise_kib, rep);
                    }
                    duo.barrier();
                    ring.close().unwrap();
                }

                for (name, buffered) in [("p2p-send", false), ("p2p-bsend", true)] {
                    let (sent, got, inject, total) =
                        p2p_stream(&duo, rank, rep, payload, n, buffered);
                    if rank == 0 {
                        Row {
                            transport: name,
                            measure: "stream",
                            payload,
                            sent,
                            delivered: got,
                            inject,
                            total,
                            wait: 0.0,
                            max_us: 0.0,
                        }
                        .print(size, depth, noise_kib, rep);
                    }
                    duo.barrier();
                }
                let (dt, max_us) = p2p_rtt(&duo, rank, rep, payload, iters);
                if rank == 0 {
                    Row {
                        transport: "p2p-send",
                        measure: "rtt",
                        payload,
                        sent: iters * 2,
                        delivered: iters * 2,
                        inject: dt,
                        total: dt,
                        wait: 0.0,
                        max_us,
                    }
                    .print(size, depth, noise_kib, rep);
                }
                duo.barrier();
            }
        }

        if rank == 0 {
            for r in 2..size {
                world.process_at_rank(r).send_with_tag(&[0u8], TAG_QUIT);
            }
        }
    });

    // Buffered sends must be drained before MPI tears the buffer down.
    world.barrier();
}
