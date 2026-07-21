// RMA put/get vs p2p send, quiet and under background noise.
//
//   mpirun -n <2+> bench [iters=1000] [max_kib=1024] [noise_kib=16]
//
// Ranks 0 and 1 are the measured pair. Ranks >= 2 pair up for continuous p2p
// ping-pong during "noisy" phases (an odd one out idles). Every safe put/get
// includes its flush: this is the crate's real cost profile, not a batched
// best case.
//
// Window layout: 64-byte flag region per phase (quiet at 0, noisy at 64),
// payload at [128, 128 + max). Flags per phase: PING +0, PONG +8, per-size
// DONE at +16 + idx*8 (idx < 6).

use mpi::collective::CommunicatorCollectives;
use mpi::point_to_point::{Destination, Source};
use mpi::topology::Communicator;
use mpi::{Rank, Threading};

use mpi_rma::Window;
use mpi_rma::traits::*;

const DATA: usize = 128;
const FLAG_PING: usize = 0;
const FLAG_PONG: usize = 8;
const FLAG_DONE: usize = 16;
const TAG_NOISE: i32 = 7;
const TAG_STOP: i32 = 99;

// Handshake between the measured pair only. Noise ranks never participate:
// a world barrier here would deadlock against ranks still in the noise loop.
fn sync01(world: &dyn Communicator, rank: Rank) {
    if rank == 0 {
        world.process_at_rank(1).send(&[0u8]);
        world.process_at_rank(1).receive_vec::<u8>();
    } else if rank == 1 {
        world.process_at_rank(0).receive_vec::<u8>();
        world.process_at_rank(0).send(&[0u8]);
    }
}

fn wait_flag(win: &Window<u8>, me: Rank, off: usize, seq: u64) {
    let mut buf = [0u8; 8];
    loop {
        win.get(me, off, &mut buf).unwrap();
        if u64::from_le_bytes(buf) == seq {
            return;
        }
        std::thread::yield_now();
    }
}

// 8-byte ping-pong; returns seconds for `iters` round trips on the driver.
fn rma_pingpong(win: &Window<u8>, rank: Rank, iters: u64, base: usize) -> f64 {
    let t0 = std::time::Instant::now();
    for i in 1..=iters {
        if rank == 0 {
            win.put(1, base + FLAG_PING, &i.to_le_bytes()).unwrap();
            wait_flag(win, 0, base + FLAG_PONG, i);
        } else {
            wait_flag(win, 1, base + FLAG_PING, i);
            win.put(0, base + FLAG_PONG, &i.to_le_bytes()).unwrap();
        }
    }
    t0.elapsed().as_secs_f64()
}

fn p2p_pingpong(world: &dyn Communicator, rank: Rank, iters: u64) -> f64 {
    let msg = [0u8; 8];
    let t0 = std::time::Instant::now();
    for _ in 0..iters {
        if rank == 0 {
            world.process_at_rank(1).send(&msg[..]);
            world.process_at_rank(1).receive_vec::<u8>();
        } else {
            world.process_at_rank(0).receive_vec::<u8>();
            world.process_at_rank(0).send(&msg[..]);
        }
    }
    t0.elapsed().as_secs_f64()
}

fn rma_put_stream(win: &Window<u8>, rank: Rank, payload: &[u8], n: u64, done: usize) -> f64 {
    let t0 = std::time::Instant::now();
    if rank == 0 {
        for _ in 0..n {
            win.put(1, DATA, payload).unwrap();
        }
        win.put(1, done, &1u64.to_le_bytes()).unwrap();
    } else {
        wait_flag(win, 1, done, 1);
    }
    t0.elapsed().as_secs_f64()
}

fn rma_get_stream(win: &Window<u8>, payload: &mut [u8], n: u64) -> f64 {
    let t0 = std::time::Instant::now();
    for _ in 0..n {
        win.get(1, DATA, payload).unwrap();
    }
    t0.elapsed().as_secs_f64()
}

// n messages plus a 1-byte ack so the driver sees full drain.
fn p2p_send_stream(world: &dyn Communicator, rank: Rank, payload: &[u8], n: u64) -> f64 {
    let t0 = std::time::Instant::now();
    if rank == 0 {
        for _ in 0..n {
            world.process_at_rank(1).send(payload);
        }
        world.process_at_rank(1).receive_vec::<u8>();
    } else {
        for _ in 0..n {
            world.process_at_rank(0).receive_vec::<u8>();
        }
        world.process_at_rank(0).send(&[0u8]);
    }
    t0.elapsed().as_secs_f64()
}

// Continuous p2p ping-pong with `partner` until a STOP message arrives.
// STOP is checked while waiting for noise and forwarded to the partner,
// so a rank never blocks on one that already left.
fn noise(world: &dyn Communicator, rank: Rank, partner: Rank, kib: usize) {
    let payload = vec![0u8; kib * 1024];
    let seen_stop = |world: &dyn Communicator| -> bool {
        if world
            .any_process()
            .immediate_probe_with_tag(TAG_STOP)
            .is_some()
        {
            world.any_process().receive_vec_with_tag::<u8>(TAG_STOP);
            return true;
        }
        false
    };
    let wait_noise = |world: &dyn Communicator| -> bool {
        loop {
            if seen_stop(world) {
                return true;
            }
            if world
                .any_process()
                .immediate_probe_with_tag(TAG_NOISE)
                .is_some()
            {
                world.any_process().receive_vec_with_tag::<u8>(TAG_NOISE);
                return false;
            }
            std::thread::yield_now();
        }
    };
    loop {
        if rank < partner {
            world
                .process_at_rank(partner)
                .send_with_tag(&payload[..], TAG_NOISE);
            if wait_noise(world) {
                break;
            }
        } else {
            if wait_noise(world) {
                break;
            }
            world
                .process_at_rank(partner)
                .send_with_tag(&payload[..], TAG_NOISE);
        }
        if seen_stop(world) {
            break;
        }
    }
    // Forward STOP and drain in-flight noise so the partner's blocking send
    // always matches before we leave.
    world
        .process_at_rank(partner)
        .send_with_tag(&[0u8], TAG_STOP);
    for _ in 0..1000 {
        while world
            .any_process()
            .immediate_probe_with_tag(TAG_NOISE)
            .is_some()
        {
            world.any_process().receive_vec_with_tag::<u8>(TAG_NOISE);
        }
        std::thread::yield_now();
    }
}

fn report(size: usize, n: u64, dt: f64) -> String {
    format!(
        "{:>8.1} MB/s {:>9.0} msg/s {:>8.2} us/msg",
        n as f64 * size as f64 / dt / (1 << 20) as f64,
        n as f64 / dt,
        dt * 1e6 / n as f64,
    )
}

fn main() {
    let mut args = std::env::args().skip(1);
    let iters: u64 = args.next().and_then(|s| s.parse().ok()).unwrap_or(1000);
    let max_kib: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(1024);
    let noise_kib: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(16);
    let max = max_kib * 1024;

    let (universe, provided) =
        mpi::initialize_with_threading(Threading::Multiple).expect("MPI already initialized");
    assert_eq!(provided, Threading::Multiple);
    let world = universe.world();
    let rank = world.rank();
    let size = world.size();
    if size < 2 {
        if rank == 0 {
            eprintln!("[bench] needs at least 2 ranks");
        }
        return;
    }
    let world_ref: &dyn Communicator = &world;

    let win = world.allocate_window::<u8>(DATA + max).unwrap();
    win.lock_all().unwrap();

    // Noise ranks pair up (2,3), (4,5), ...; an odd one out idles.
    let noise_last = 2 + (size - 2) / 2 * 2; // ranks 2..noise_last are paired
    let partner = |r: Rank| -> Option<Rank> {
        if r < 2 || r >= noise_last {
            return None;
        }
        Some(if (r - 2) % 2 == 0 { r + 1 } else { r - 1 })
    };

    for noisy in [false, true] {
        let base = if noisy { 64 } else { 0 };
        world.barrier();
        let noise_partner = if noisy { partner(rank) } else { None };
        if let Some(p) = noise_partner {
            noise(world_ref, rank, p, noise_kib);
        } else {
            if rank == 0 {
                use std::io::Write;
                if noisy && noise_last > 2 {
                    println!(
                        "\n[bench] noisy ({} noise ranks, {noise_kib} KiB)",
                        noise_last - 2
                    );
                } else if !noisy {
                    println!("[bench] quiet ({size} ranks, {iters} iters)");
                } else {
                    println!("\n[bench] noisy (no noise ranks)");
                }
                let _ = std::io::stdout().flush();
            }
            if rank <= 1 {
                let dt = rma_pingpong(&win, rank, iters, base);
                if rank == 0 {
                    print!(
                        "  latency rma rtt/2 {:>7.2}us",
                        dt * 1e6 / (2.0 * iters as f64)
                    );
                }
            }
            sync01(world_ref, rank);
            if rank <= 1 {
                let dt = p2p_pingpong(world_ref, rank, iters);
                if rank == 0 {
                    println!("  | p2p rtt/2 {:>7.2}us", dt * 1e6 / (2.0 * iters as f64));
                }
            }
            sync01(world_ref, rank);

            for (idx, &s) in [64, 4096, 65536, 1048576]
                .iter()
                .filter(|&&s| s <= max)
                .enumerate()
            {
                let n = ((32 << 20) / s).clamp(20, 20_000) as u64;
                let done = base + FLAG_DONE + idx * 8;
                if rank == 0 {
                    print!(
                        "  {s:>7} B  put {}",
                        report(s, n, rma_put_stream(&win, rank, &vec![0xAB; s], n, done))
                    );
                } else if rank == 1 {
                    rma_put_stream(&win, rank, &[], n, done);
                }
                sync01(world_ref, rank);
                if rank == 0 {
                    print!(
                        "  | get {}",
                        report(s, n, rma_get_stream(&win, &mut vec![0; s], n))
                    );
                }
                sync01(world_ref, rank);
                if rank <= 1 {
                    let dt = p2p_send_stream(world_ref, rank, &vec![0xAB; s], n);
                    if rank == 0 {
                        println!("  | send {}", report(s, n, dt));
                    }
                }
                sync01(world_ref, rank);
            }
        }
        if noisy && noise_last > 2 {
            if rank == 0 {
                for r in 2..noise_last {
                    world.process_at_rank(r).send_with_tag(&[0u8], TAG_STOP);
                }
            }
            world.barrier();
        }
    }

    world.barrier();
    win.close().unwrap();
}
