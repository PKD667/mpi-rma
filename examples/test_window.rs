// Window API test. Run under mpirun:
//   mpirun -n 2 test_window

use mpi::Threading;
use mpi::collective::CommunicatorCollectives;
use mpi::topology::Communicator;

use mpi_rma::Error;
use mpi_rma::traits::*;

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

    // Element type must agree across ranks.
    if rank == 0 {
        assert!(matches!(
            world.allocate_window::<u8>(1),
            Err(Error::Window("element type differs between ranks"))
        ));
    } else {
        assert!(matches!(
            world.allocate_window::<u64>(2),
            Err(Error::Window("element type differs between ranks"))
        ));
    }

    // Self put/get, bounds, fetch_add; relies on Drop for shutdown.
    {
        let win = world.allocate_window::<u64>(8).unwrap();
        win.put(rank, 0, &[rank as u64 + 10, rank as u64 + 20])
            .unwrap();
        let mut got = [0; 8];
        win.get(rank, 0, &mut got).unwrap();
        assert_eq!(got[0], rank as u64 + 10);
        assert_eq!(got[1], rank as u64 + 20);

        assert!(matches!(
            win.put(rank, win.len(), &[1]),
            Err(Error::Range { .. })
        ));
        assert!(matches!(win.put(-1, 0, &[1]), Err(Error::Rank(-1))));
        assert!(matches!(win.put(size, 0, &[1]), Err(Error::Rank(_))));
        assert!(matches!(
            win.get(rank, win.len(), &mut [0]),
            Err(Error::Range { .. })
        ));

        assert_eq!(win.fetch_add(rank, 7, rank as u64 + 1).unwrap(), 0);
        assert_eq!(win.fetch_add(rank, 7, 0).unwrap(), rank as u64 + 1);
        world.barrier();
    }

    // Cross-rank exchange.
    {
        let win = world.allocate_window::<u64>(8).unwrap();
        win.put(next, 0, &[rank as u64 + 100]).unwrap();
        world.barrier();
        let mut got = [0u64; 1];
        win.get(rank, 0, &mut got).unwrap();
        assert_eq!(got[0], prev as u64 + 100);
        world.barrier();
        win.close().unwrap();
    }

    if rank == 0 {
        println!("test_window: ok");
    }
}
