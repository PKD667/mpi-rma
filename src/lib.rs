//! Typed, safe MPI one-sided communication (RMA) for Rust, on top of
//! [rsmpi](https://docs.rs/mpi).
//!
//! rsmpi does not expose RMA windows. This crate contains the raw RMA calls,
//! extends every rsmpi communicator through [`CommunicatorRmaExt`], and builds
//! fixed-slot [`Ring`] transport on those windows.
//!
//! [`Ring::safe`] provides backpressure with a cumulative-acknowledgement
//! gate. [`Ring::raw`] never blocks and reports overwritten messages instead.
//! Polling reads local memory; each safe [`Ring::ack`] call uses one atomic
//! MPI operation.
//!
//! ```no_run
//! use mpi::topology::Communicator;
//! use mpi_rma::Ring;
//!
//! # let (universe, _) = mpi::initialize_with_threading(mpi::Threading::Multiple).unwrap();
//! let world = universe.world();
//! // One lane from rank 0 to rank 1: 8 slots of 64 KiB each.
//! let ring = Ring::safe(&world, &[(0, 1, 8, 64 * 1024)]).unwrap();
//! if world.rank() == 0 {
//!     ring.send(1, b"hello").unwrap();
//! } else if world.rank() == 1 {
//!     'receive: loop {
//!         for message in ring.poll().unwrap() {
//!             ring.ack(message.origin, message.sequence).unwrap();
//!             break 'receive;
//!         }
//!         std::thread::yield_now();
//!     }
//! }
//! ring.close().unwrap();
//! ```
//!
//! See `design.md` for the slot layout, sequencing rules, and the
//! acknowledge-gating model behind the ring transport, and `BENCHMARKS.md`
//! for measured throughput and latency against MPI point-to-point.

mod error;
mod ring;
mod window;

pub use error::Error;
pub use ring::{Message, Ring};
pub use window::{CommunicatorRmaExt, MemoryModel, RmaElement, Window};

/// rsmpi-style trait prelude. Importing this extends the original
/// communicator types in place; no communicator wrapper is involved.
pub mod traits {
    pub use crate::CommunicatorRmaExt;
}
