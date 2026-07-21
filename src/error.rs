use mpi::Threading;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    Mpi(i32),
    Threading(Threading),
    EmptyWindow,
    SizeOverflow,
    CountOverflow,
    Rank(i32),
    Range {
        start: usize,
        len: usize,
        window: usize,
    },
    Layout(&'static str),
    Epoch(&'static str),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Mpi(code) => write!(f, "MPI RMA error {code}"),
            Error::Threading(got) => {
                write!(f, "MPI RMA requires Threading::Multiple, got {got:?}")
            }
            Error::EmptyWindow => write!(f, "RMA window cannot be empty"),
            Error::SizeOverflow => write!(f, "RMA window size does not fit MPI_Aint"),
            Error::CountOverflow => write!(f, "RMA transfer count does not fit MPI Count"),
            Error::Rank(rank) => write!(f, "rank {rank} is outside the window communicator"),
            Error::Range { start, len, window } => write!(
                f,
                "RMA range {start}..{} exceeds window length {window}",
                start.saturating_add(*len)
            ),
            Error::Layout(msg) => write!(f, "invalid RMA target layout: {msg}"),
            Error::Epoch(msg) => write!(f, "invalid RMA epoch: {msg}"),
        }
    }
}

impl std::error::Error for Error {}
