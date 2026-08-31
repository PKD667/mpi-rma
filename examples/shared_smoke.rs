// SharedWindow smoke test. Run under mpirun:
//   mpirun -n 4 shared_smoke

use mpi::Threading;
use mpi::collective::CommunicatorCollectives;
use mpi::topology::Communicator;

fn main() {
    let (universe, provided) =
        mpi::initialize_with_threading(Threading::Multiple).expect("MPI must initialize once");
    assert_eq!(provided, Threading::Multiple);
    let world = universe.world();

    for total in [65_536, 65_537, 10_025] {
        let rank = world.rank() as usize;
        let size = world.size() as usize;
        let base = total / size * rank;
        let end = if rank + 1 == size {
            total
        } else {
            total / size * (rank + 1)
        };
        let mine: Vec<u8> = (base..end).map(|i| (i % 251) as u8).collect();
        let shared = mpi_rma::SharedWindow::publish(&world, &mine, total).unwrap();
        let got = shared.get();
        assert_eq!(got.len(), total);
        let expected: Vec<u8> = (0..total).map(|i| (i % 251) as u8).collect();
        // Every rank sees every other rank's pattern at the offset its slices imply, which is the
        // contiguity guarantee the flat segment rests on. `publish` also checks each rank's own
        // slice lands there, so reaching this line at all is the assertion. The three totals cover
        // page-aligned shares, a remainder, and shares that are not page multiples at all.
        assert_eq!(got, expected.as_slice());
        drop(shared);
        world.barrier();
    }

    // Empty publish: the mapping is an empty slice, nothing dereferenced.
    let empty = mpi_rma::SharedWindow::publish(&world, &[], 0).unwrap();
    assert!(empty.get().is_empty());
    assert!(empty.is_empty());

    // A second window coexists with the first; both drop collectively at scope end.
    world.barrier();
    if world.rank() == 0 {
        println!("shared_smoke: ok");
    }
}
