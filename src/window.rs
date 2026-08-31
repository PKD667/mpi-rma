use std::ffi::c_void;
use std::marker::PhantomData;

use mpi::collective::CommunicatorCollectives;
use mpi::datatype::Equivalence;
use mpi::ffi;
use mpi::raw::AsRaw;
use mpi::topology::{Communicator, Rank, SimpleCommunicator};

use crate::Error;

mod sealed {
    pub trait Sealed {}
}

/// Plain scalar values whose in-memory representation is safe for RMA.
///
/// The trait is sealed: arbitrary `Equivalence` implementations may describe
/// datatypes whose extent exceeds the Rust object and cannot safely back a
/// contiguous window.
pub trait RmaElement: sealed::Sealed + Equivalence + Copy + Send + Sync + 'static {
    #[doc(hidden)]
    const TYPE_ID: u8;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryModel {
    Unified,
    Separate,
}

macro_rules! elements {
    ($($t:ty => $id:literal),* $(,)?) => {$(
        impl sealed::Sealed for $t {}
        impl RmaElement for $t {
            const TYPE_ID: u8 = $id;
        }
    )*};
}

elements!(
    u8 => 1,
    u16 => 2,
    u32 => 3,
    u64 => 4,
    usize => 5,
    i8 => 6,
    i16 => 7,
    i32 => 8,
    i64 => 9,
    isize => 10,
    f32 => 11,
    f64 => 12,
);

fn check(code: i32) -> Result<(), Error> {
    if code == ffi::MPI_SUCCESS as i32 {
        Ok(())
    } else {
        Err(Error::Mpi(code))
    }
}

/// Enter the passive-target epoch for `win`, zero its storage, and report
/// the memory model. The caller owns the handle and must free it on error.
///
/// # Safety
/// `win` must be a freshly allocated handle owning `base` for `len`
/// elements of `T`, with no access epoch started.
unsafe fn init_epoch<T>(win: ffi::MPI_Win, base: *mut T, len: usize) -> Result<MemoryModel, Error> {
    if len > 0 && base.is_null() {
        return Err(Error::Mpi(ffi::MPI_ERR_WIN as i32));
    }
    let mut value: *mut c_void = std::ptr::null_mut();
    let mut flag = 0;
    unsafe {
        check(ffi::MPI_Win_get_attr(
            win,
            ffi::MPI_WIN_MODEL as i32,
            &mut value as *mut *mut c_void as *mut c_void,
            &mut flag,
        ))?;
    }
    if flag == 0 || value.is_null() {
        return Err(Error::Mpi(ffi::MPI_ERR_WIN as i32));
    }
    let model = unsafe {
        if *(value as *const i32) == ffi::MPI_WIN_UNIFIED as i32 {
            MemoryModel::Unified
        } else {
            MemoryModel::Separate
        }
    };
    // NOCHECK: no conflicting lock can exist yet, since all ranks are still
    // inside the collective constructor.
    unsafe {
        check(ffi::MPI_Win_lock_all(ffi::MPI_MODE_NOCHECK as i32, win))?;
    }
    // RmaElement is sealed to numeric scalars, for which all-zero is a valid
    // value. Win_sync publishes it under the separate model.
    if len > 0 {
        unsafe { std::ptr::write_bytes(base, 0, len) };
    }
    if model == MemoryModel::Separate {
        unsafe { check(ffi::MPI_Win_sync(win))? };
    }
    Ok(model)
}

/// Method-style RMA extension for every rsmpi communicator.
pub trait CommunicatorRmaExt: Communicator {
    fn allocate_window<T: RmaElement>(&self, len: usize) -> Result<Window<T>, Error> {
        Window::allocate(self, len)
    }
}

impl<C: Communicator + ?Sized> CommunicatorRmaExt for C {}

/// MPI-allocated homogeneous memory exposed to the communicator.
///
/// `Send + Sync` once constructed. Construction requires
/// `MPI_THREAD_MULTIPLE`; the access epoch is entered inside `allocate` and
/// left in [`close`](Self::close) or `Drop`, and no public API exposes
/// epoch transitions. All reads and writes are issued through the
/// `put`/`get`/`fetch_add` family, each completed at the target before
/// returning; no Rust reference into remotely mutable storage is ever
/// exposed.
pub struct Window<T: RmaElement> {
    win: ffi::MPI_Win,
    base: *mut T,
    len: usize,
    lengths: Vec<usize>,
    rank: Rank,
    ranks: Rank,
    model: MemoryModel,
    closed: bool,
    _element: PhantomData<T>,
}

// SAFETY: the only constructor rejects anything below MPI_THREAD_MULTIPLE.
// Storage is reached only through MPI calls, never via Rust references.
unsafe impl<T: RmaElement> Send for Window<T> {}
unsafe impl<T: RmaElement> Sync for Window<T> {}

impl<T: RmaElement> Window<T> {
    fn allocate<C: Communicator + ?Sized>(comm: &C, len: usize) -> Result<Self, Error> {
        if comm.test_inter() {
            return Err(Error::Intercommunicator);
        }
        let threading = mpi::environment::threading_support();
        if threading != mpi::Threading::Multiple {
            return Err(Error::Threading(threading));
        }
        let width = std::mem::size_of::<T>();
        let config = [
            u64::try_from(len).map_err(|_| Error::SizeOverflow)?,
            u64::try_from(width).map_err(|_| Error::SizeOverflow)?,
            T::TYPE_ID as u64,
        ];
        let ranks = usize::try_from(comm.size()).map_err(|_| Error::SizeOverflow)?;
        let configs_len = ranks.checked_mul(config.len()).ok_or(Error::SizeOverflow)?;
        if configs_len > i32::MAX as usize {
            return Err(Error::CountOverflow);
        }
        let mut configs = vec![0; configs_len];
        comm.all_gather_into(&config[..], &mut configs[..]);
        if configs
            .chunks_exact(config.len())
            .any(|c| c[1..] != config[1..])
        {
            return Err(Error::Window("element type differs between ranks"));
        }
        let lengths = configs
            .chunks_exact(config.len())
            .map(|c| usize::try_from(c[0]).map_err(|_| Error::SizeOverflow))
            .collect::<Result<Vec<_>, _>>()?;
        if lengths.iter().any(|&len| {
            len.checked_mul(width)
                .and_then(|bytes| ffi::MPI_Aint::try_from(bytes).ok())
                .is_none()
        }) {
            return Err(Error::SizeOverflow);
        }
        let bytes = len.checked_mul(width).ok_or(Error::SizeOverflow)?;
        let bytes = ffi::MPI_Aint::try_from(bytes).map_err(|_| Error::SizeOverflow)?;
        let width = i32::try_from(width).map_err(|_| Error::SizeOverflow)?;
        let mut base: *mut c_void = std::ptr::null_mut();
        // SAFETY: read-only MPI null handle used as an out-parameter seed.
        let mut win = unsafe { ffi::RSMPI_WIN_NULL };
        unsafe {
            check(ffi::MPI_Win_allocate(
                bytes,
                width,
                ffi::RSMPI_INFO_NULL,
                comm.as_raw(),
                &mut base as *mut *mut c_void as *mut c_void,
                &mut win,
            ))?;
        }
        // The window handle is live from here on; any failure must free it.
        let model = match unsafe { init_epoch(win, base as *mut T, len) } {
            Ok(model) => model,
            Err(error) => {
                unsafe {
                    let _ = ffi::MPI_Win_free(&mut win);
                }
                return Err(error);
            }
        };
        comm.barrier();
        Ok(Window {
            win,
            base: base as *mut T,
            len,
            lengths,
            rank: comm.rank(),
            ranks: comm.size(),
            model,
            closed: false,
            _element: PhantomData,
        })
    }

    /// Number of locally allocated elements.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether this rank allocated an empty window.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Memory model reported by `MPI_WIN_MODEL`.
    pub fn memory_model(&self) -> MemoryModel {
        self.model
    }

    /// Make remote updates visible to local memory accesses.
    ///
    /// Required after a remote put on a separate-model window before any
    /// local read can observe the new contents.
    pub fn sync(&self) -> Result<(), Error> {
        unsafe { check(ffi::MPI_Win_sync(self.win)) }
    }

    /// Copy a region from this rank's local window storage.
    ///
    /// Concurrent remote updates to the same region yield undefined
    /// contents; the caller is responsible for the ordering. Volatile
    /// reads of single scalars are provided by
    /// [`read_local_volatile`](Self::read_local_volatile).
    pub fn read_local(&self, disp: usize, out: &mut [T]) -> Result<(), Error> {
        self.validate(self.rank, disp, out.len())?;
        if out.is_empty() {
            return Ok(());
        }
        unsafe {
            std::ptr::copy_nonoverlapping(self.base.add(disp), out.as_mut_ptr(), out.len());
        }
        Ok(())
    }

    /// Volatile local copy for scalars concurrently mutated by an RMA operation.
    pub fn read_local_volatile(&self, disp: usize, out: &mut [T]) -> Result<(), Error> {
        self.validate(self.rank, disp, out.len())?;
        for (i, value) in out.iter_mut().enumerate() {
            unsafe {
                *value = std::ptr::read_volatile(self.base.add(disp + i));
            }
        }
        Ok(())
    }

    /// Put a contiguous region and complete the transfer at the target before return.
    ///
    /// The transfer reaches remote completion (target's public window
    /// visible to subsequent `get` from any process) before this returns.
    ///
    /// # Errors
    /// - [`Error::Rank`] if `dest` is outside the window communicator.
    /// - [`Error::Range`] if `disp + data.len()` exceeds the target's
    ///   window length.
    /// - [`Error::CountOverflow`] if `data.len()` does not fit an `MPI Count`.
    /// - [`Error::Mpi`] if the MPI call fails.
    pub fn put(&self, dest: Rank, disp: usize, data: &[T]) -> Result<(), Error> {
        self.validate(dest, disp, data.len())?;
        let count = i32::try_from(data.len()).map_err(|_| Error::CountOverflow)?;
        let datatype = T::equivalent_datatype();
        unsafe {
            check(ffi::MPI_Put(
                data.as_ptr() as *const c_void,
                count,
                datatype.as_raw(),
                dest,
                disp as ffi::MPI_Aint,
                count,
                datatype.as_raw(),
                self.win,
            ))?;
            check(ffi::MPI_Win_flush(dest, self.win))
        }
    }

    /// Get a contiguous region and complete the transfer before return.
    ///
    /// # Errors
    /// - [`Error::Rank`] if `source` is outside the window communicator.
    /// - [`Error::Range`] if `disp + out.len()` exceeds the source's
    ///   window length.
    /// - [`Error::CountOverflow`] if `out.len()` does not fit an `MPI Count`.
    /// - [`Error::Mpi`] if the MPI call fails.
    pub fn get(&self, source: Rank, disp: usize, out: &mut [T]) -> Result<(), Error> {
        self.validate(source, disp, out.len())?;
        let count = i32::try_from(out.len()).map_err(|_| Error::CountOverflow)?;
        let datatype = T::equivalent_datatype();
        unsafe {
            check(ffi::MPI_Get(
                out.as_mut_ptr() as *mut c_void,
                count,
                datatype.as_raw(),
                source,
                disp as ffi::MPI_Aint,
                count,
                datatype.as_raw(),
                self.win,
            ))?;
            check(ffi::MPI_Win_flush(source, self.win))
        }
    }

    /// Atomically add `value` at the target and return the previous value.
    ///
    /// # Errors
    /// - [`Error::Rank`] if `dest` is outside the window communicator.
    /// - [`Error::Range`] if `disp` is outside the target's window length.
    /// - [`Error::Mpi`] if the MPI call fails.
    pub fn fetch_add(&self, dest: Rank, disp: usize, value: T) -> Result<T, Error> {
        self.validate(dest, disp, 1)?;
        let datatype = T::equivalent_datatype();
        let mut previous = std::mem::MaybeUninit::<T>::uninit();
        unsafe {
            check(ffi::MPI_Fetch_and_op(
                &value as *const T as *const c_void,
                previous.as_mut_ptr() as *mut c_void,
                datatype.as_raw(),
                dest,
                disp as ffi::MPI_Aint,
                ffi::RSMPI_SUM,
                self.win,
            ))?;
            check(ffi::MPI_Win_flush(dest, self.win))?;
            Ok(previous.assume_init())
        }
    }

    /// Close the window. Collective over the window group.
    ///
    /// Prefer this over relying on `Drop`: the destructor's shutdown runs
    /// the same steps but the collective boundary is implicit.
    pub fn close(mut self) -> Result<(), Error> {
        let result = self.finish();
        if result.is_err() {
            self.closed = true;
        }
        result
    }

    fn validate(&self, rank: Rank, start: usize, len: usize) -> Result<(), Error> {
        if rank < 0 || rank >= self.ranks {
            return Err(Error::Rank(rank));
        }
        let end = start.checked_add(len).ok_or(Error::SizeOverflow)?;
        let window = self.lengths[rank as usize];
        if end > window {
            return Err(Error::Range { start, len, window });
        }
        Ok(())
    }

    fn finish(&mut self) -> Result<(), Error> {
        if self.closed {
            return Ok(());
        }
        unsafe {
            check(ffi::MPI_Win_unlock_all(self.win))?;
            check(ffi::MPI_Win_free(&mut self.win))?;
        }
        self.closed = true;
        Ok(())
    }
}

impl<T: RmaElement> Drop for Window<T> {
    fn drop(&mut self) {
        // MPI_Win_free is collective. Well-structured MPI programs drop
        // windows symmetrically; explicit `close` makes that boundary visible.
        let _ = self.finish();
    }
}

/// One read-only shared-memory segment per node, assembled from one slice per
/// node-local rank. MPI places the slices consecutively when
/// `alloc_shared_noncontig` is not set, so every rank can expose the node's
/// whole segment as one flat slice without a leader allocation.
pub struct SharedWindow {
    win: ffi::MPI_Win,
    base: *const u8,
    len: usize,
    _comm: SimpleCommunicator,
}

// SAFETY: the segment bytes are written once, inside `publish`, before the
// node barrier, and nothing in this crate ever exposes a mutable view of
// them, so concurrent reads through `get` observe memory nobody writes.
unsafe impl Send for SharedWindow {}
unsafe impl Sync for SharedWindow {}

impl SharedWindow {
    /// Publish each rank's slice into one flat shared segment per node.
    ///
    /// `mine.len()` is this rank's allocation and `total` is the complete
    /// segment length. Every rank on a node contributes one disjoint slice.
    pub fn publish<C: Communicator + ?Sized>(
        participants: &C,
        mine: &[u8],
        total: usize,
    ) -> Result<Self, Error> {
        if participants.test_inter() {
            return Err(Error::Intercommunicator);
        }
        let threading = mpi::environment::threading_support();
        if threading != mpi::Threading::Multiple {
            return Err(Error::Threading(threading));
        }
        let comm = participants.split_shared(0);
        let node_size = usize::try_from(comm.size()).map_err(|_| Error::SizeOverflow)?;
        let node_rank = usize::try_from(comm.rank()).map_err(|_| Error::SizeOverflow)?;
        let local = u64::try_from(mine.len()).map_err(|_| Error::SizeOverflow)?;
        let mut lengths = vec![0u64; node_size];
        comm.all_gather_into(&local, &mut lengths);
        let sum = lengths.iter().try_fold(0usize, |sum, &length| {
            sum.checked_add(usize::try_from(length).map_err(|_| Error::SizeOverflow)?)
                .ok_or(Error::SizeOverflow)
        })?;
        if sum != total {
            return Err(Error::Window("shared slices do not cover the segment"));
        }
        if total > 0 && lengths[0] == 0 {
            return Err(Error::Window("rank zero must own a non-empty shared slice"));
        }
        let expected_offset = lengths[..node_rank]
            .iter()
            .try_fold(0usize, |sum, &length| {
                sum.checked_add(usize::try_from(length).map_err(|_| Error::SizeOverflow)?)
                    .ok_or(Error::SizeOverflow)
            })?;
        let local_bytes = ffi::MPI_Aint::try_from(mine.len()).map_err(|_| Error::SizeOverflow)?;

        let mut win = unsafe { ffi::RSMPI_WIN_NULL };
        let mut mapped: *mut c_void = std::ptr::null_mut();
        let outcome: Result<(), Error> = (|| {
            unsafe {
                check(ffi::MPI_Win_allocate_shared(
                    local_bytes,
                    1,
                    ffi::RSMPI_INFO_NULL,
                    comm.as_raw(),
                    &mut mapped as *mut *mut c_void as *mut c_void,
                    &mut win,
                ))?;
            }
            // The segment is read by plain loads, which the standard orders
            // only through window synchronization: lock_all opens the epoch,
            // Win_sync by the writer publishes its stores, the barrier orders
            // the ranks in time, Win_sync by every reader orders its later
            // loads, unlock_all closes the epoch.
            let mut model: *mut c_void = std::ptr::null_mut();
            let mut flag = 0;
            unsafe {
                check(ffi::MPI_Win_get_attr(
                    win,
                    ffi::MPI_WIN_MODEL as i32,
                    &mut model as *mut *mut c_void as *mut c_void,
                    &mut flag,
                ))?;
            }
            // Load/store semantics on a shared window are defined only in the
            // unified model; refuse the other rather than read undefined bytes.
            if flag == 0
                || model.is_null()
                || unsafe { *(model as *const i32) } != ffi::MPI_WIN_UNIFIED as i32
            {
                return Err(Error::Window(
                    "shared window is not in the unified memory model",
                ));
            }
            unsafe {
                check(ffi::MPI_Win_lock_all(ffi::MPI_MODE_NOCHECK as i32, win))?;
            }
            if !mine.is_empty() {
                unsafe {
                    std::ptr::copy_nonoverlapping(mine.as_ptr(), mapped as *mut u8, mine.len());
                    check(ffi::MPI_Win_sync(win))?;
                }
            }
            comm.barrier();
            let mut size = 0;
            let mut disp_unit = 0;
            unsafe {
                check(ffi::MPI_Win_shared_query(
                    win,
                    0,
                    &mut size,
                    &mut disp_unit,
                    &mut mapped as *mut *mut c_void as *mut c_void,
                ))?;
            }
            if disp_unit != 1 || (total > 0 && size <= 0) {
                return Err(Error::Window("shared segment has no rank-zero storage"));
            }
            if total > 0 {
                let whole = mapped as *const u8;
                let local_base = unsafe { (whole as *const u8).add(expected_offset) };
                let mut queried = std::ptr::null_mut();
                let mut queried_size = 0;
                let mut queried_disp = 0;
                unsafe {
                    check(ffi::MPI_Win_shared_query(
                        win,
                        node_rank as Rank,
                        &mut queried_size,
                        &mut queried_disp,
                        &mut queried as *mut *mut c_void as *mut c_void,
                    ))?;
                }
                if queried_disp != 1 || queried_size != mine.len() as ffi::MPI_Aint {
                    return Err(Error::Window("shared slice size changed during query"));
                }
                if !mine.is_empty() && queried as *const u8 != local_base {
                    return Err(Error::Window("shared slices are not contiguous"));
                }
            }
            unsafe {
                check(ffi::MPI_Win_sync(win))?;
                check(ffi::MPI_Win_unlock_all(win))?;
            }
            Ok(())
        })();
        if outcome.is_err() {
            unsafe {
                let _ = check(ffi::MPI_Win_free(&mut win));
            }
        }
        outcome?;
        Ok(SharedWindow {
            win,
            base: mapped as *const u8,
            len: total,
            _comm: comm,
        })
    }

    /// Zero-copy view of the published bytes.
    #[inline]
    pub fn get(&self) -> &[u8] {
        if self.len == 0 {
            &[]
        } else {
            // SAFETY: `base` is the node's mapped segment, live as long as
            // the window, and nothing writes it after `publish` returns.
            unsafe { std::slice::from_raw_parts(self.base, self.len) }
        }
    }

    /// Number of published bytes.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether `publish` was given an empty buffer.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl Drop for SharedWindow {
    fn drop(&mut self) {
        // MPI_Win_free is collective over the window's group (one node), so
        // every rank of a node must drop its window at the same point. See
        // the type documentation.
        unsafe {
            let _ = check(ffi::MPI_Win_free(&mut self.win));
        }
    }
}
