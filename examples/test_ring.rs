// Ring test: safe and raw modes on a 2-rank pair.
//   mpirun -n 2 test_ring

use std::thread;
use std::time::{Duration, Instant};

use mpi::Threading;
use mpi::collective::CommunicatorCollectives;
use mpi::topology::{Communicator, Rank, SimpleCommunicator};

use mpi_rma::{Error, Message, Ring};

fn pair(depth: usize, capacity: usize) -> Vec<(Rank, Rank, usize, usize)> {
    vec![(0, 1, depth, capacity), (1, 0, depth, capacity)]
}

fn oneway(depth: usize, capacity: usize) -> Vec<(Rank, Rank, usize, usize)> {
    vec![(0, 1, depth, capacity)]
}

fn safe_basics(world: &SimpleCommunicator, rank: Rank, next: Rank, prev: Rank) {
    // Depth 4 against 3 messages: this covers ordering and acknowledge
    // validation with the overwrite gate deliberately out of the way.
    // Backpressure is its own test, on a one-way lane.
    let ring = Ring::safe(world, &pair(4, 8)).unwrap();
    assert!(ring.is_safe());
    assert_eq!(ring.depth(next), Some(4));
    assert_eq!(ring.capacity(next), Some(8));

    assert!(matches!(
        ring.send(next, &[0; 9]),
        Err(Error::Payload {
            len: 9,
            capacity: 8
        })
    ));
    assert!(matches!(
        ring.ack(prev, 1),
        Err(Error::Ack {
            sequence: 1,
            received: 0,
            ..
        })
    ));

    for s in 1..=3u8 {
        assert_eq!(ring.send(next, &[rank as u8, s]).unwrap(), u64::from(s));
    }
    world.barrier();

    let messages = ring.poll().unwrap();
    assert_eq!(
        messages,
        (1..=3)
            .map(|s| Message {
                origin: prev,
                sequence: s,
                data: vec![prev as u8, s as u8],
            })
            .collect::<Vec<_>>()
    );
    assert!(ring.poll().unwrap().is_empty());
    assert_eq!(ring.lost(), 0);
    assert_eq!(ring.max_lag(), 3);

    assert!(matches!(
        ring.ack(prev, 4),
        Err(Error::Ack {
            sequence: 4,
            received: 3,
            ..
        })
    ));
    ring.ack(prev, 3).unwrap();
    ring.ack(prev, 3).unwrap();
    ring.ack(prev, 2).unwrap();

    world.barrier();
    ring.close().unwrap();
}

/// A safe sender with no free slot blocks until the receiver acknowledges.
///
/// One-way, so the ack timing belongs to the test. On a symmetric pair each
/// rank's ack races the other's gate check: whichever receiver drains first
/// opens its peer's gate before that peer ever evaluates it, and neither side
/// can be relied on to block.
fn backpressure(world: &SimpleCommunicator, rank: Rank) {
    let ring = Ring::safe(world, &oneway(2, 8)).unwrap();
    if rank == 0 {
        assert_eq!(ring.send(1, &[1]).unwrap(), 1);
        assert_eq!(ring.send(1, &[2]).unwrap(), 2);
    }
    world.barrier();

    if rank == 0 {
        let sent = thread::scope(|scope| {
            let sender = &ring;
            let handle = scope.spawn(move || sender.send(1, &[3]));
            // Depth is 2 and nothing has been acknowledged, so there is no slot
            // for the third message and the send has to wait.
            let deadline = Instant::now() + Duration::from_secs(5);
            while ring.waits() == 0 {
                assert!(Instant::now() < deadline, "safe sender never blocked");
                thread::yield_now();
            }
            // The receiver is parked here, so it cannot have drained early.
            world.barrier();
            handle.join().expect("sender thread panicked").unwrap()
        });
        assert_eq!(sent, 3);
        assert!(ring.waits() > 0);
        assert!(ring.wait_ns() > 0);
    } else {
        world.barrier();
        let messages = ring.poll().unwrap();
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].data, vec![1]);
        assert_eq!(messages[1].data, vec![2]);
        assert!(matches!(
            ring.ack(0, 3),
            Err(Error::Ack {
                sequence: 3,
                received: 2,
                ..
            })
        ));
        ring.ack(0, 2).unwrap();
        ring.ack(0, 2).unwrap();
    }

    world.barrier();
    if rank == 1 {
        let messages = ring.poll().unwrap();
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].sequence, 3);
        assert_eq!(messages[0].data, vec![3]);
        ring.ack(0, 3).unwrap();
    }
    world.barrier();
    ring.close().unwrap();
}

fn oneway_basics(world: &SimpleCommunicator, rank: Rank) {
    let ring = Ring::safe(world, &oneway(2, 8)).unwrap();
    if rank == 0 {
        assert_eq!(ring.send(1, &[7]).unwrap(), 1);
    }
    world.barrier();
    if rank == 1 {
        let messages = ring.poll().unwrap();
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].origin, 0);
        assert_eq!(messages[0].data, vec![7]);
        ring.ack(0, 1).unwrap();
    }
    world.barrier();
    ring.close().unwrap();
}

fn raw_basics(world: &SimpleCommunicator, rank: Rank, next: Rank, prev: Rank) {
    let ring = Ring::raw(world, &pair(2, 8)).unwrap();
    assert!(!ring.is_safe());

    for s in 1..=3u8 {
        assert_eq!(ring.send(next, &[rank as u8, s]).unwrap(), u64::from(s));
    }
    world.barrier();

    let messages = ring.poll().unwrap();
    assert_eq!(
        messages,
        vec![
            Message {
                origin: prev,
                sequence: 2,
                data: vec![prev as u8, 2]
            },
            Message {
                origin: prev,
                sequence: 3,
                data: vec![prev as u8, 3]
            },
        ]
    );
    assert_eq!(ring.lost(), 1);

    // Raw acks are no-ops.
    ring.ack(prev, u64::MAX).unwrap();
    world.barrier();
    ring.close().unwrap();
}

fn main() {
    let (universe, provided) =
        mpi::initialize_with_threading(Threading::Multiple).expect("MPI must initialize once");
    assert_eq!(provided, Threading::Multiple);
    let world = universe.world();
    let rank = world.rank();
    let size = world.size();
    assert_eq!(size, 2);
    let next = (rank + 1) % size;
    let prev = (rank + size - 1) % size;

    // Configuration must agree across ranks.
    if rank == 0 {
        assert!(matches!(
            Ring::safe(&world, &pair(1, 8)),
            Err(Error::Ring("configuration differs between ranks"))
        ));
    } else {
        assert!(matches!(
            Ring::safe(&world, &pair(2, 8)),
            Err(Error::Ring("configuration differs between ranks"))
        ));
    }
    assert!(matches!(
        Ring::raw(&world, &pair(0, 8)),
        Err(Error::Ring("depth must be positive"))
    ));

    safe_basics(&world, rank, next, prev);
    oneway_basics(&world, rank);
    backpressure(&world, rank);
    raw_basics(&world, rank, next, prev);

    if rank == 0 {
        println!("test_ring: ok");
    }
}
