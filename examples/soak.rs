// Ring correctness under sustained all-to-all load.
//
//   mpirun -n <2+> soak [mode=safe] [messages=20000] [payload=256] [depth=8]
//                       [noise_kib=0] [pace_ns=0] [rep=0]  # rep labels output rows
//
// Every rank opens a lane to every other rank and drives them all from the main
// thread while a poller thread drains its own incoming lanes. Payloads are
// self-describing: origin, sequence, and a sequence-derived fill, so a receiver
// checks *content*, not just arrival counts.
//
// What is asserted, per mode:
//
//   safe  every message arrives exactly once, in order, intact. Zero loss.
//   raw   arrivals are a strictly increasing subsequence of what was sent, every
//         arrival is intact, and delivered + lost equals what was sent. Loss is
//         expected; corruption, reordering and duplication are not.
//
// `noise_kib > 0` runs background p2p ping-pong between rank pairs on a separate
// thread, so the ring is measured against a busy progress engine rather than an
// idle one.
//
// `pace_ns > 0` holds each sender to one round every `pace_ns` nanoseconds, one
// round being one message to every peer. That is the offered-rate axis: sweeping
// it against raw-mode loss is what locates the point where a lossy lane of a
// given depth stops keeping up. At 0 the sender runs flat out.
//
// Writes one TSV row per rank on stdout, diagnostics on stderr.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use mpi::Threading;
use mpi::collective::CommunicatorCollectives;
use mpi::point_to_point::{Destination, Source};
use mpi::topology::{Communicator, Rank, SimpleCommunicator};

use mpi_rma::{Message, Ring};

const TAG_NOISE: i32 = 11;
const TAG_STOP: i32 = 12;
/// Origin and sequence, ahead of the fill.
const STAMP: usize = 12;

fn paint(origin: Rank, sequence: u64, buf: &mut [u8]) {
    buf[..4].copy_from_slice(&origin.to_le_bytes());
    buf[4..STAMP].copy_from_slice(&sequence.to_le_bytes());
    for (i, b) in buf[STAMP..].iter_mut().enumerate() {
        *b = (sequence as u8).wrapping_add(i as u8);
    }
}

/// Check a payload against the origin and sequence the ring reported for it.
fn verify(m: &Message) -> Result<(), String> {
    if m.data.len() < STAMP {
        return Err(format!(
            "payload from {} is {} bytes",
            m.origin,
            m.data.len()
        ));
    }
    let origin = i32::from_le_bytes(m.data[..4].try_into().unwrap());
    let sequence = u64::from_le_bytes(m.data[4..STAMP].try_into().unwrap());
    if origin != m.origin || sequence != m.sequence {
        return Err(format!(
            "payload says ({origin}, {sequence}), ring says ({}, {})",
            m.origin, m.sequence
        ));
    }
    for (i, &b) in m.data[STAMP..].iter().enumerate() {
        let want = (sequence as u8).wrapping_add(i as u8);
        if b != want {
            return Err(format!(
                "payload from {origin} seq {sequence} corrupt at byte {}: {b} != {want}",
                i + STAMP
            ));
        }
    }
    Ok(())
}

/// Carries a communicator reference onto the noise thread.
///
/// rsmpi wraps a raw MPI handle, so `SimpleCommunicator` is neither `Send` nor
/// `Sync`. Under `MPI_THREAD_MULTIPLE` concurrent calls are legal, and the noise
/// thread only ever talks to its own partner on its own tags.
struct Shared<'a>(&'a SimpleCommunicator);
unsafe impl Send for Shared<'_> {}

/// Background p2p ping-pong inside rank pairs (0,1), (2,3), ... until `stop`.
///
/// The lower rank of each pair owns termination: it stops sending payload and
/// sends a STOP instead, which the higher rank answers by leaving. An odd rank
/// at the end has no partner and idles.
fn noise(world: &SimpleCommunicator, kib: usize, stop: &AtomicBool) {
    let rank = world.rank();
    let partner = if rank % 2 == 0 { rank + 1 } else { rank - 1 };
    if partner >= world.size() {
        return;
    }
    let payload = vec![0xA5u8; kib * 1024];
    let peer = world.process_at_rank(partner);
    loop {
        if rank < partner {
            if stop.load(Ordering::Relaxed) {
                peer.send_with_tag(&[0u8], TAG_STOP);
                return;
            }
            peer.send_with_tag(&payload[..], TAG_NOISE);
            peer.receive_vec_with_tag::<u8>(TAG_NOISE);
        } else {
            let (_, status) = peer.receive_vec::<u8>();
            if status.tag() == TAG_STOP {
                return;
            }
            peer.send_with_tag(&payload[..], TAG_NOISE);
        }
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let mode = args.next().unwrap_or_else(|| "safe".into());
    let messages: u64 = args.next().and_then(|s| s.parse().ok()).unwrap_or(20_000);
    let payload: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(256);
    let depth: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(8);
    let noise_kib: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(0);
    let pace_ns: u64 = args.next().and_then(|s| s.parse().ok()).unwrap_or(0);
    // Echoed straight into the output so repeats of one point stay distinguishable.
    let rep: u32 = args.next().and_then(|s| s.parse().ok()).unwrap_or(0);
    assert!(
        payload >= STAMP,
        "payload must hold the origin and sequence"
    );
    let safe = match mode.as_str() {
        "safe" => true,
        "raw" => false,
        other => panic!("mode must be safe or raw, got {other:?}"),
    };

    let (universe, provided) =
        mpi::initialize_with_threading(Threading::Multiple).expect("MPI must initialize once");
    assert_eq!(provided, Threading::Multiple);
    let world = universe.world();
    let rank = world.rank();
    let size = world.size();
    assert!(size >= 2, "soak needs at least 2 ranks");

    // All-to-all: every ordered pair of distinct ranks gets a lane.
    let mut lanes = Vec::new();
    for source in 0..size {
        for destination in 0..size {
            if source != destination {
                lanes.push((source, destination, depth, payload));
            }
        }
    }
    let ring = Arc::new(if safe {
        Ring::safe(&world, &lanes).unwrap()
    } else {
        Ring::raw(&world, &lanes).unwrap()
    });

    let stop = Arc::new(AtomicBool::new(false));
    let done = Arc::new(AtomicBool::new(false));
    let count = Arc::new(AtomicU64::new(0));
    let peers = (size - 1) as u64;
    let sent = messages * peers;

    world.barrier();
    let started = std::time::Instant::now();

    let outcome = std::thread::scope(|scope| {
        if noise_kib > 0 {
            let stop = Arc::clone(&stop);
            let shared = Shared(&world);
            scope.spawn(move || {
                // Name the wrapper as a whole first. Closures capture the
                // narrowest place they use, and `shared.0` alone would be
                // captured as a bare reference, losing the `Send` impl.
                let shared = shared;
                noise(shared.0, noise_kib, &stop);
            });
        }

        let poller = {
            let (ring, done, count) = (Arc::clone(&ring), Arc::clone(&done), Arc::clone(&count));
            scope.spawn(move || -> Result<Vec<u64>, String> {
                let mut seen = vec![0u64; size as usize];
                let mut total = 0u64;
                loop {
                    // If true, this poll follows the global send barrier.
                    let finished = done.load(Ordering::Acquire);
                    let batch = ring.poll().map_err(|e| format!("poll: {e}"))?;
                    let empty = batch.is_empty();
                    let mut furthest: HashMap<Rank, u64> = HashMap::new();
                    for m in &batch {
                        verify(m)?;
                        let previous = seen[m.origin as usize];
                        if m.sequence <= previous {
                            return Err(format!(
                                "sequence from {} went backwards: {} after {previous}",
                                m.origin, m.sequence
                            ));
                        }
                        if safe && m.sequence != previous + 1 {
                            return Err(format!(
                                "safe lane from {} skipped {} to {}",
                                m.origin,
                                previous + 1,
                                m.sequence
                            ));
                        }
                        seen[m.origin as usize] = m.sequence;
                        let f = furthest.entry(m.origin).or_insert(0);
                        *f = (*f).max(m.sequence);
                        total += 1;
                    }
                    for (origin, sequence) in furthest {
                        ring.ack(origin, sequence)
                            .map_err(|e| format!("ack to {origin}: {e}"))?;
                    }
                    count.store(total, Ordering::Relaxed);
                    if finished && empty {
                        return Ok(seen);
                    }
                    if empty {
                        std::thread::yield_now();
                    }
                }
            })
        };

        // Round-robin over peers so every outgoing lane stays hot at once.
        let mut buf = vec![0u8; payload];
        let pace = std::time::Duration::from_nanos(pace_ns);
        let opened = std::time::Instant::now();
        for sequence in 1..=messages {
            for destination in (0..size).filter(|&r| r != rank) {
                paint(rank, sequence, &mut buf);
                ring.send(destination, &buf)
                    .unwrap_or_else(|e| panic!("rank {rank}: send to {destination}: {e}"));
            }
            // Absolute deadlines, so a slow round is not paid for twice and the
            // offered rate stays the one that was asked for.
            if pace_ns > 0 {
                let due = opened + pace * sequence as u32;
                while std::time::Instant::now() < due {
                    std::hint::spin_loop();
                }
            }
        }
        // Every rank's sends have completed at their targets before this
        // returns, so after the barrier nothing new can appear in a slot.
        world.barrier();
        done.store(true, Ordering::Release);

        let seen = poller.join().expect("poller thread panicked");
        stop.store(true, Ordering::Relaxed);
        seen
    });

    let elapsed = started.elapsed().as_secs_f64();
    let fail = |why: String| -> ! {
        eprintln!("[soak] FAIL rank {rank}: {why}");
        world.abort(1)
    };
    let seen = outcome.unwrap_or_else(|why| fail(why));

    let got = count.load(Ordering::Relaxed);
    let lost = ring.lost();
    let corrupt = ring.corrupt();

    // The last message on a lane is never overwritten, so every lane must end
    // exactly at the sender's high-water mark in both modes.
    for origin in (0..size).filter(|&r| r != rank) {
        if seen[origin as usize] != messages {
            fail(format!(
                "lane from {origin} ended at {} not {messages}",
                seen[origin as usize]
            ));
        }
    }
    if safe && lost != 0 {
        fail(format!("safe ring lost {lost}"));
    }
    if got + lost != sent {
        fail(format!("received {got} + lost {lost} != sent {sent}"));
    }

    // Gather the counters and let rank 0 write every row. Interleaving the
    // ranks' own stdout would scatter the header into the middle of the table:
    // mpirun orders a rank's output against itself and nothing else.
    let mine = [
        got,
        lost,
        corrupt,
        ring.max_lag(),
        ring.waits(),
        ring.wait_ns(),
    ];
    let mut all = vec![0u64; mine.len() * size as usize];
    world.all_gather_into(&mine[..], &mut all[..]);
    if rank == 0 {
        println!(
            "mode\tranks\tdepth\tpayload\tnoise_kib\tpace_ns\trep\trank\tsent\treceived\tlost\tcorrupt\tmax_lag\twaits\twait_s\tseconds"
        );
        for (r, row) in all.chunks_exact(mine.len()).enumerate() {
            println!(
                "{mode}\t{size}\t{depth}\t{payload}\t{noise_kib}\t{pace_ns}\t{rep}\t{r}\t{sent}\t{}\t{}\t{}\t{}\t{}\t{:.6}\t{elapsed:.6}",
                row[0],
                row[1],
                row[2],
                row[3],
                row[4],
                row[5] as f64 * 1e-9,
            );
        }
    }

    world.barrier();
    Arc::try_unwrap(ring)
        .unwrap_or_else(|_| panic!("ring still shared at close"))
        .close()
        .unwrap();
    if rank == 0 {
        eprintln!("[soak] ok: {mode}, {size} ranks, {sent} messages/rank in {elapsed:.2}s");
    }
}
