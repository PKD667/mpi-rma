use mpi::Threading;
use mpi::collective::CommunicatorCollectives;
use mpi::topology::Communicator;

use mpi_rma::Indexed;
use mpi_rma::traits::*;

fn main() {
    let (universe, provided) =
        mpi::initialize_with_threading(Threading::Multiple).expect("MPI already initialized");
    assert_eq!(provided, Threading::Multiple);
    let world = universe.world();
    let rank = world.rank();
    let size = world.size();
    let next = (rank + 1) % size;
    let prev = (rank + size - 1) % size;

    let win = world.allocate_window::<u64>(8).unwrap();
    win.lock_all().unwrap();
    win.put(next, 0, &[rank as u64]).unwrap();
    let scatter = Indexed::new(&[2, 5], win.len()).unwrap();
    win.put_indexed(next, &[rank as u64 + 10, rank as u64 + 20], &scatter)
        .unwrap();
    world.barrier();

    let mut got = [0; 8];
    win.get(rank, 0, &mut got).unwrap();
    assert_eq!(got[0], prev as u64);
    assert_eq!(got[2], prev as u64 + 10);
    assert_eq!(got[5], prev as u64 + 20);

    world.barrier();
    win.close().unwrap();
}
