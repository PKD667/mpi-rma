use std::ffi::c_void;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicBool, Ordering};

use mpi::collective::CommunicatorCollectives;
use mpi::datatype::{Equivalence, UserDatatype};
use mpi::ffi;
use mpi::raw::AsRaw;
use mpi::topology::{Communicator, Rank};

use crate::{Error, Indexed};

mod sealed {
    pub trait Sealed {}
}

/// Plain scalar values whose in-memory representation is safe for RMA.
///
/// The trait is sealed: arbitrary `Equivalence` implementations may describe
/// datatypes whose extent exceeds the Rust object and cannot safely back a
/// contiguous window.
pub trait RmaElement: sealed::Sealed + Equivalence + Copy + Send + Sync + 'static {}

macro_rules! elements {
    ($($t:ty),* $(,)?) => {$(
        impl sealed::Sealed for $t {}
        impl RmaElement for $t {}
    )*};
}

elements!(u8, u16, u32, u64, usize, i8, i16, i32, i64, isize, f32, f64);

fn check(code: i32) -> Result<(), Error> {
    if code == ffi::MPI_SUCCESS as i32 {
        Ok(())
    } else {
        Err(Error::Mpi(code))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryModel {
    Unified,
    Separate,
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
/// Construction requires `MPI_THREAD_MULTIPLE`, making shared references safe
/// for concurrent MPI calls. No Rust references into remotely mutable storage
/// are exposed; all reads use `get` and all writes use `put`, each completed at
/// the target before returning.
pub struct Window<T: RmaElement> {
    win: ffi::MPI_Win,
    len: usize,
    ranks: Rank,
    model: MemoryModel,
    locked: AtomicBool,
    closed: bool,
    _element: PhantomData<T>,
}

// SAFETY: the only constructor rejects anything below MPI_THREAD_MULTIPLE.
// Storage is accessed through completed MPI operations, never Rust references.
unsafe impl<T: RmaElement> Send for Window<T> {}
unsafe impl<T: RmaElement> Sync for Window<T> {}

impl<T: RmaElement> Window<T> {
    fn allocate<C: Communicator + ?Sized>(comm: &C, len: usize) -> Result<Self, Error> {
        let threading = mpi::environment::threading_support();
        if threading != mpi::Threading::Multiple {
            return Err(Error::Threading(threading));
        }
        if len == 0 {
            return Err(Error::EmptyWindow);
        }
        let width = std::mem::size_of::<T>();
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
        if base.is_null() {
            return Err(Error::Mpi(ffi::MPI_ERR_WIN as i32));
        }
        unsafe {
            check(ffi::MPI_Win_allocate(
                bytes,
                width,
                ffi::RSMPI_INFO_NULL,
                comm.as_raw(),
                &mut base as *mut *mut c_void as *mut c_void,
                &mut win,
            ))?;
            // RmaElement is sealed to numeric scalars, for which all-zero is a
            // valid value. Counters and slots therefore start deterministic.
            std::ptr::write_bytes(base as *mut T, 0, len);
        }
        // Do not let one rank begin RMA while another is still zeroing its
        // local allocation.
        comm.barrier();
        let model = unsafe {
            let mut value: *mut c_void = std::ptr::null_mut();
            let mut flag = 0;
            check(ffi::MPI_Win_get_attr(
                win,
                ffi::MPI_WIN_MODEL as i32,
                &mut value as *mut *mut c_void as *mut c_void,
                &mut flag,
            ))?;
            if flag == 0 || value.is_null() {
                return Err(Error::Mpi(ffi::MPI_ERR_WIN as i32));
            }
            if *(value as *const i32) == ffi::MPI_WIN_UNIFIED as i32 {
                MemoryModel::Unified
            } else {
                MemoryModel::Separate
            }
        };
        Ok(Window {
            win,
            len,
            ranks: comm.size(),
            model,
            locked: AtomicBool::new(false),
            closed: false,
            _element: PhantomData,
        })
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        false
    }

    pub fn memory_model(&self) -> MemoryModel {
        self.model
    }

    pub fn lock_all(&self) -> Result<(), Error> {
        self.locked
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|_| Error::Epoch("lock_all called twice"))?;
        let result = unsafe {
            check(ffi::MPI_Win_lock_all(
                ffi::MPI_MODE_NOCHECK as i32,
                self.win,
            ))
        };
        if result.is_err() {
            self.locked.store(false, Ordering::Release);
        }
        result
    }

    pub fn unlock_all(&self) -> Result<(), Error> {
        self.locked
            .compare_exchange(true, false, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|_| Error::Epoch("unlock_all without lock_all"))?;
        let result = unsafe { check(ffi::MPI_Win_unlock_all(self.win)) };
        if result.is_err() {
            self.locked.store(true, Ordering::Release);
        }
        result
    }

    /// Put a contiguous region and complete it at the target before return.
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

    /// Get a contiguous region and complete it before return.
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

    /// Scatter contiguous values into a validated noncontiguous target map.
    pub fn put_indexed(&self, dest: Rank, values: &[T], target: &Indexed<T>) -> Result<(), Error> {
        if values.len() != target.len {
            return Err(Error::Layout("origin and target signatures differ"));
        }
        if target.bound > self.len {
            return Err(Error::Range {
                start: target.bound - 1,
                len: 1,
                window: self.len,
            });
        }
        self.validate(dest, 0, 0)?;
        let count = i32::try_from(values.len()).map_err(|_| Error::CountOverflow)?;
        let datatype = T::equivalent_datatype();
        let target_datatype =
            UserDatatype::heterogeneous_indexed_block(1, &target.displacements, &datatype);
        unsafe {
            check(ffi::MPI_Put(
                values.as_ptr() as *const c_void,
                count,
                datatype.as_raw(),
                dest,
                0,
                1,
                target_datatype.as_raw(),
                self.win,
            ))?;
            check(ffi::MPI_Win_flush(dest, self.win))
        }
    }

    /// Collective over the window group. Prefer this over relying on Drop.
    pub fn close(mut self) -> Result<(), Error> {
        self.finish()
    }

    fn validate(&self, rank: Rank, start: usize, len: usize) -> Result<(), Error> {
        if !self.locked.load(Ordering::Acquire) {
            return Err(Error::Epoch("RMA operation outside lock_all epoch"));
        }
        if rank < 0 || rank >= self.ranks {
            return Err(Error::Rank(rank));
        }
        let end = start.checked_add(len).ok_or(Error::SizeOverflow)?;
        if end > self.len {
            return Err(Error::Range {
                start,
                len,
                window: self.len,
            });
        }
        Ok(())
    }

    fn finish(&mut self) -> Result<(), Error> {
        if self.closed {
            return Ok(());
        }
        if self.locked.load(Ordering::Acquire) {
            unsafe { check(ffi::MPI_Win_unlock_all(self.win))? };
            self.locked.store(false, Ordering::Release);
        }
        unsafe { check(ffi::MPI_Win_free(&mut self.win))? };
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
