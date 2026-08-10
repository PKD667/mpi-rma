//! Error types for the RMA layer and the fixed-slot ring transport.
//!
//! - `Mpi`, `Threading`: environment conditions (returned by the MPI
//!   error handler or the threading check).
//! - `Intercommunicator`, `SizeOverflow`, `CountOverflow`, `Rank`, `Range`,
//!   `Window`: window API failures.
//! - `Ring`, `Payload`, `Ack`, `Lapped`: ring transport failures.

use mpi::Threading;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    Mpi(i32),
    Threading(Threading),
    Intercommunicator,
    SizeOverflow,
    CountOverflow,
    Rank(i32),
    Range {
        start: usize,
        len: usize,
        window: usize,
    },
    Window(&'static str),
    Ring(&'static str),
    Payload {
        len: usize,
        capacity: usize,
    },
    Ack {
        origin: i32,
        sequence: u64,
        received: u64,
    },
    Lapped {
        origin: i32,
        expected: u64,
        found: u64,
    },
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Mpi(code) => write!(f, "MPI RMA error {code}"),
            Error::Threading(got) => {
                write!(f, "MPI RMA requires Threading::Multiple, got {got:?}")
            }
            Error::Intercommunicator => write!(f, "MPI RMA requires an intracommunicator"),
            Error::SizeOverflow => write!(f, "RMA window size does not fit MPI_Aint"),
            Error::CountOverflow => write!(f, "RMA transfer count does not fit MPI Count"),
            Error::Rank(rank) => write!(f, "rank {rank} is outside the window communicator"),
            Error::Range { start, len, window } => write!(
                f,
                "RMA range {start}..{} exceeds window length {window}",
                start.saturating_add(*len)
            ),
            Error::Window(msg) => write!(f, "invalid RMA window: {msg}"),
            Error::Ring(msg) => write!(f, "invalid RMA ring state: {msg}"),
            Error::Payload { len, capacity } => {
                write!(f, "ring payload length {len} exceeds capacity {capacity}")
            }
            Error::Ack {
                origin,
                sequence,
                received,
            } => write!(
                f,
                "cannot acknowledge sequence {sequence} from rank {origin}; received through {received}"
            ),
            Error::Lapped {
                origin,
                expected,
                found,
            } => write!(
                f,
                "safe lane from rank {origin} was overwritten: expected sequence {expected}, found {found}"
            ),
        }
    }
}

impl std::error::Error for Error {}
