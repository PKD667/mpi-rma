// Multi-lane layout test: rank 2 packs two incoming lanes into one window
// (offsets 0 and slot bytes) and two ack counters. Run under mpirun:
//   mpirun -n 3 test_layout

use mpi::Threading;
use mpi::collective::CommunicatorCollectives;
use mpi::topology::{Communicator, Rank, SimpleCommunicator};

use mpi_rma::{Message, Ring};

fn lanes(depth: usize, capacity: usize) -> Vec<(Rank, Rank, usize, usize)> {
    vec![
        (0, 2, depth, capacity),
        (1, 2, depth, capacity),
        (2, 0, depth, capacity),
        (2, 1, depth, capacity),
    ]
}

fn run(world: &SimpleCommunicator, safe: bool) {
    let rank = world.rank();
    let ring = if safe {
        Ring::safe(world, &lanes(2, 8)).unwrap()
    } else {
        Ring::raw(world, &lanes(2, 8)).unwrap()
    };

    if rank < 2 {
        ring.send(2, &[rank as u8, 1]).unwrap();
    }
    world.barrier();

    if rank == 2 {
        let messages = ring.poll().unwrap();
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].origin, 0);
        assert_eq!(messages[0].data, vec![0, 1]);
        assert_eq!(messages[1].origin, 1);
        assert_eq!(messages[1].data, vec![1, 1]);
        ring.ack(0, 1).unwrap();
        ring.ack(1, 1).unwrap();
        ring.send(0, &[2, 1]).unwrap();
        ring.send(1, &[2, 1]).unwrap();
    }
    world.barrier();

    if rank < 2 {
        let messages = ring.poll().unwrap();
        assert_eq!(
            messages,
            vec![Message {
                origin: 2,
                sequence: 1,
                data: vec![2, 1]
            }]
        );
        ring.ack(2, 1).unwrap();
    }
    world.barrier();

    ring.close().unwrap();
}

fn main() {
    let (universe, provided) =
        mpi::initialize_with_threading(Threading::Multiple).expect("MPI must initialize once");
    assert_eq!(provided, Threading::Multiple);
    let world = universe.world();
    assert_eq!(world.size(), 3);

    run(&world, true);
    run(&world, false);

    if world.rank() == 0 {
        println!("test_layout: ok");
    }
}
