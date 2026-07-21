use mpi::Threading;
use mpi::collective::CommunicatorCollectives;
use mpi::topology::Communicator;

use mpi_rma::traits::*;
use mpi_rma::{Error, Indexed};

#[test]
fn contiguous_and_indexed_roundtrip() {
    let (universe, provided) =
        mpi::initialize_with_threading(Threading::Multiple).expect("MPI must initialize once");
    assert_eq!(provided, Threading::Multiple);
    let world = universe.world();
    let rank = world.rank();
    let size = world.size();
    let next = (rank + 1) % size;
    let prev = (rank + size - 1) % size;

    let win = world.allocate_window::<u64>(8).unwrap();
    assert!(matches!(
        win.get(rank, 0, &mut [0]),
        Err(Error::Epoch("RMA operation outside lock_all epoch"))
    ));
    win.lock_all().unwrap();

    win.put(next, 0, &[rank as u64 + 10, rank as u64 + 20])
        .unwrap();
    world.barrier();

    let mut got = [0; 8];
    win.get(rank, 0, &mut got).unwrap();
    assert_eq!(got[0], prev as u64 + 10);
    assert_eq!(got[1], prev as u64 + 20);

    let layout = Indexed::new(&[2, 5], win.len()).unwrap();
    win.put_indexed(next, &[rank as u64 + 100, rank as u64 + 200], &layout)
        .unwrap();
    world.barrier();

    win.get(rank, 0, &mut got).unwrap();
    assert_eq!(got[2], prev as u64 + 100);
    assert_eq!(got[5], prev as u64 + 200);
    assert!(matches!(
        win.put(rank, win.len(), &[1]),
        Err(Error::Range { .. })
    ));
    assert!(matches!(
        win.put_indexed(rank, &[1], &layout),
        Err(Error::Layout("origin and target signatures differ"))
    ));

    world.barrier();
    win.close().unwrap();
}
