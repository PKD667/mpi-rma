//! The leader's published SGR segment for the host backends: one POSIX
//! shared-memory object per publication, written once by its creator and mapped
//! read-only by the node's workers. OS calls only: no MPI, because the segment
//! is a fact about the node, not the job.

use std::ffi::{CStr, CString};
use std::io;
use std::num::NonZeroU64;
use std::os::raw::c_int;
use std::sync::atomic::{AtomicU64, Ordering};

/// Length of the fixed header before the payload.
const HEADER: usize = 64;
/// Format tag at object offset 0. `from_le_bytes` so the file bytes are b"trameseg".
const MAGIC: u64 = u64::from_le_bytes(*b"trameseg");

/// One POSIX shared-memory segment: its mapping, its name, and who may unlink it.
///
/// `base` is null once the mapping has been detached. `owner` is true only for the
/// `create` handle. `failed` records the syscall and errno of an explicit cleanup
/// that was refused, so `Drop` can report it rather than retry.
pub struct Segment {
    base: *mut u8,
    size: usize,
    name: CString,
    owner: bool,
    failed: Option<(&'static str, i32)>,
}

fn close_or_abort(name: &CStr, fd: c_int) {
    if unsafe { libc::close(fd) } != 0 {
        let e = io::Error::last_os_error();
        eprintln!("mpi-rma segment {:?}: cleanup close failed: {e}", name);
        std::process::abort();
    }
}

fn munmap_or_abort(name: &CStr, base: *mut u8, size: usize) {
    if unsafe { libc::munmap(base.cast(), size) } != 0 {
        let e = io::Error::last_os_error();
        eprintln!("mpi-rma segment {:?}: cleanup munmap failed: {e}", name);
        std::process::abort();
    }
}

fn shm_unlink_or_abort(name: &CStr) {
    if unsafe { libc::shm_unlink(name.as_ptr()) } != 0 {
        let e = io::Error::last_os_error();
        eprintln!("mpi-rma segment {:?}: cleanup shm_unlink failed: {e}", name);
        std::process::abort();
    }
}

impl Segment {
    /// Create `name` (O_CREAT|O_EXCL), write `payload`, and publish `revision` last.
    ///
    /// The whole object is reserved before the copy, so an exhausted `/dev/shm` is
    /// refused with its errno (e.g. `ENOSPC`) rather than faulting (`SIGBUS`) during
    /// the copy.
    pub fn create(name: &CStr, revision: NonZeroU64, payload: &[u8]) -> io::Result<Self> {
        let size = HEADER.checked_add(payload.len()).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidInput, "segment size overflows usize")
        })?;
        let length = u64::try_from(payload.len()).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "payload length does not fit u64",
            )
        })?;
        let size_off = libc::off_t::try_from(size).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "segment size does not fit off_t",
            )
        })?;

        let fd = unsafe {
            libc::shm_open(
                name.as_ptr(),
                libc::O_CREAT | libc::O_EXCL | libc::O_RDWR,
                0o600,
            )
        };
        if fd < 0 {
            return Err(io::Error::last_os_error());
        }

        if unsafe { libc::ftruncate(fd, size_off) } != 0 {
            let e = io::Error::last_os_error();
            close_or_abort(name, fd);
            shm_unlink_or_abort(name);
            return Err(e);
        }

        // `posix_fallocate` returns the errno directly instead of setting `errno`, so a
        // nonzero return value is that error. Reserve the whole object before mapping: an
        // unsupported or exhausted reservation is refused here rather than faulting during
        // the copy below.
        let reservation = unsafe { libc::posix_fallocate(fd, 0, size_off) };
        if reservation != 0 {
            let e = io::Error::from_raw_os_error(reservation);
            close_or_abort(name, fd);
            shm_unlink_or_abort(name);
            return Err(e);
        }

        let mapped = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                fd,
                0,
            )
        };
        if mapped == libc::MAP_FAILED {
            let e = io::Error::last_os_error();
            close_or_abort(name, fd);
            shm_unlink_or_abort(name);
            return Err(e);
        }
        let base = mapped.cast::<u8>();

        if unsafe { libc::close(fd) } != 0 {
            let e = io::Error::last_os_error();
            munmap_or_abort(name, base, size);
            shm_unlink_or_abort(name);
            return Err(e);
        }

        unsafe {
            std::ptr::write(base as *mut u64, MAGIC.to_le());
            std::ptr::write(base.add(8) as *mut u64, length.to_le());
            std::ptr::copy_nonoverlapping(payload.as_ptr(), base.add(HEADER), payload.len());
            // Release: the payload and header stores are visible before the revision is.
            AtomicU64::from_ptr(base.add(16) as *mut u64).store(revision.get(), Ordering::Release);
        }

        Ok(Segment {
            base,
            size,
            name: name.to_owned(),
            owner: true,
            failed: None,
        })
    }

    /// Map the published segment `name` read-only and confirm `revision` and `length`.
    ///
    /// # Safety
    /// The creator does not write the object while this mapping lives.
    pub unsafe fn open(name: &CStr, revision: NonZeroU64, length: usize) -> io::Result<Self> {
        let size = HEADER.checked_add(length).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidInput, "segment size overflows usize")
        })?;
        let size_off = libc::off_t::try_from(size).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "segment size does not fit off_t",
            )
        })?;

        let fd = unsafe { libc::shm_open(name.as_ptr(), libc::O_RDONLY, 0) };
        if fd < 0 {
            return Err(io::Error::last_os_error());
        }

        let mut stat: libc::stat = unsafe { std::mem::zeroed() };
        if unsafe { libc::fstat(fd, &mut stat) } != 0 {
            let e = io::Error::last_os_error();
            close_or_abort(name, fd);
            return Err(e);
        }
        if stat.st_size != size_off {
            close_or_abort(name, fd);
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "segment size disagrees with the handle",
            ));
        }

        let mapped = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                size,
                libc::PROT_READ,
                libc::MAP_SHARED,
                fd,
                0,
            )
        };
        if mapped == libc::MAP_FAILED {
            let e = io::Error::last_os_error();
            close_or_abort(name, fd);
            return Err(e);
        }
        let base = mapped.cast::<u8>();

        if unsafe { libc::close(fd) } != 0 {
            let e = io::Error::last_os_error();
            munmap_or_abort(name, base, size);
            return Err(e);
        }

        // The first mapped read acquires the published revision, before any other
        // header field or payload byte. A 64-bit aligned atomic load on read-only
        // memory is sound on the supported targets (std::sync::atomic, "Atomic
        // accesses to read-only memory").
        let published =
            unsafe { AtomicU64::from_ptr(base.add(16) as *mut u64).load(Ordering::Acquire) };
        if published != revision.get() {
            munmap_or_abort(name, base, size);
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "segment revision disagrees with the handle",
            ));
        }

        let magic = u64::from_le(unsafe { std::ptr::read(base as *const u64) });
        let stored = u64::from_le(unsafe { std::ptr::read(base.add(8) as *const u64) });
        if magic != MAGIC || stored != length as u64 {
            munmap_or_abort(name, base, size);
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "segment format disagrees with the handle",
            ));
        }

        Ok(Segment {
            base,
            size,
            name: name.to_owned(),
            owner: false,
            failed: None,
        })
    }

    /// The published bytes. A detached segment has no bytes: this asserts rather
    /// than manufacturing a slice from a null base.
    pub fn payload(&self) -> &[u8] {
        assert!(
            !self.base.is_null(),
            "mpi_rma::Segment::payload after detach"
        );
        // SAFETY: `base` maps `size` bytes; the assert above rejects the detached
        // (null) state before any pointer arithmetic.
        unsafe { std::slice::from_raw_parts(self.base.add(HEADER), self.size - HEADER) }
    }

    /// Unmap. On a refused `munmap` the mapping is kept and the errno recorded.
    /// A recorded failure is sticky: further detaches refuse with that errno and
    /// issue no syscall.
    pub fn detach(&mut self) -> io::Result<()> {
        if let Some((_, errno)) = self.failed {
            return Err(io::Error::from_raw_os_error(errno));
        }
        if self.base.is_null() {
            return Ok(());
        }
        if unsafe { libc::munmap(self.base.cast(), self.size) } != 0 {
            let errno = io::Error::last_os_error()
                .raw_os_error()
                .expect("munmap failure reports an errno");
            self.failed = Some(("munmap", errno));
            return Err(io::Error::from_raw_os_error(errno));
        }
        self.base = std::ptr::null_mut();
        Ok(())
    }

    /// Detach, then unlink the name. Only the owner may call this. On a refused
    /// `shm_unlink` the owner and name are kept and the errno recorded; that
    /// failure is sticky, so a later retire issues no syscall.
    pub fn retire(&mut self) -> io::Result<()> {
        if let Some((_, errno)) = self.failed {
            return Err(io::Error::from_raw_os_error(errno));
        }
        assert!(self.owner, "mpi_rma::Segment::retire on a reader");
        self.detach()?;
        if unsafe { libc::shm_unlink(self.name.as_ptr()) } != 0 {
            let errno = io::Error::last_os_error()
                .raw_os_error()
                .expect("shm_unlink failure reports an errno");
            self.failed = Some(("shm_unlink", errno));
            return Err(io::Error::from_raw_os_error(errno));
        }
        self.owner = false;
        Ok(())
    }
}

impl Drop for Segment {
    fn drop(&mut self) {
        // A refused explicit cleanup is reported once and never retried.
        if let Some((syscall, errno)) = self.failed {
            eprintln!(
                "mpi-rma segment {:?}: {syscall} failed (errno {errno}); refusing to retry",
                self.name
            );
            std::process::abort();
        }
        if !self.base.is_null() {
            if unsafe { libc::munmap(self.base.cast(), self.size) } != 0 {
                let errno = io::Error::last_os_error()
                    .raw_os_error()
                    .expect("munmap failure reports an errno");
                eprintln!(
                    "mpi-rma segment {:?}: munmap failed (errno {errno})",
                    self.name
                );
                std::process::abort();
            }
            self.base = std::ptr::null_mut();
        }
        if self.owner {
            if unsafe { libc::shm_unlink(self.name.as_ptr()) } != 0 {
                let errno = io::Error::last_os_error()
                    .raw_os_error()
                    .expect("shm_unlink failure reports an errno");
                eprintln!(
                    "mpi-rma segment {:?}: shm_unlink failed (errno {errno})",
                    self.name
                );
                std::process::abort();
            }
            self.owner = false;
        }
    }
}
