//! Safe, typed MPI one-sided communication on top of rsmpi.
//!
//! rsmpi does not expose RMA windows. This crate contains the raw RMA calls
//! and extends every rsmpi communicator through [`CommunicatorRmaExt`].
//!
//! ```no_run
//! use mpi_rma::CommunicatorRmaExt;
//!
//! # let universe = mpi::initialize().unwrap();
//! let world = universe.world();
//! let win = world.allocate_window::<u64>(1024).unwrap();
//! ```
//!
//! Derived target layouts describe fixed memory maps. They do not execute
//! target-side code: MPI cannot insert into `BTreeMap`, follow pointers,
//! allocate nodes, compare keys, or preserve Rust collection invariants.

mod error;
mod layout;
mod window;

pub use error::Error;
pub use layout::Indexed;
pub use window::{CommunicatorRmaExt, MemoryModel, RmaElement, Window};

/// rsmpi-style trait prelude. Importing this extends the original
/// communicator types; no communicator wrapper is involved.
pub mod traits {
    pub use crate::CommunicatorRmaExt;
}
