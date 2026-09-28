//! Integration tests for the OS-only segment: byte identity past one frame, two
//! live revisions, refusal of a wrong handle, and a reader that outlives its creator.

use std::ffi::CString;
use std::io::ErrorKind;
use std::num::NonZeroU64;

use mpi_rma::Segment;

const LEN: usize = 65_545;

fn payload() -> Vec<u8> {
    (0..LEN).map(|i| (i * 7 % 251) as u8).collect()
}

fn name(tag: &str) -> CString {
    CString::new(format!("/mpi-rma-test-{}-{tag}", std::process::id())).expect("no NUL in a name")
}

fn revision(value: u64) -> NonZeroU64 {
    NonZeroU64::new(value).expect("a nonzero revision")
}

/// The error from opening a handle the object does not match. `Segment` has no
/// `Debug`, so this avoids `expect_err`/`unwrap_err`.
fn refuse(name: &std::ffi::CStr, revision: NonZeroU64, length: usize) -> std::io::Error {
    match unsafe { Segment::open(name, revision, length) } {
        Ok(_) => panic!("open of a disagreeing handle succeeded"),
        Err(e) => e,
    }
}

#[test]
fn a_segment_past_one_frame_is_bit_identical() {
    let name = name("bit");
    let bytes = payload();
    let mut creator = Segment::create(&name, revision(1), &bytes).expect("create");
    let mut reader = unsafe { Segment::open(&name, revision(1), LEN) }.expect("open");
    assert_eq!(reader.payload(), &bytes[..]);
    reader.detach().expect("detach");
    creator.retire().expect("retire");
}

#[test]
fn two_revisions_coexist() {
    let one = name("rev-one");
    let two = name("rev-two");
    let first: Vec<u8> = (0..LEN).map(|i| (i * 3 % 251) as u8).collect();
    let second: Vec<u8> = (0..LEN).map(|i| (i * 5 % 241) as u8).collect();
    let mut creator_one = Segment::create(&one, revision(1), &first).expect("create rev one");
    let mut creator_two = Segment::create(&two, revision(2), &second).expect("create rev two");
    let mut reader_one = unsafe { Segment::open(&one, revision(1), LEN) }.expect("open rev one");
    let mut reader_two = unsafe { Segment::open(&two, revision(2), LEN) }.expect("open rev two");
    assert_eq!(reader_one.payload(), &first[..]);
    assert_eq!(reader_two.payload(), &second[..]);
    reader_one.detach().expect("detach rev one");
    reader_two.detach().expect("detach rev two");
    creator_one.retire().expect("retire rev one");
    creator_two.retire().expect("retire rev two");
}

#[test]
fn a_wrong_revision_or_length_is_refused() {
    let name = name("wrong");
    let bytes = payload();
    let mut creator = Segment::create(&name, revision(5), &bytes).expect("create");
    let wrong_revision = refuse(&name, revision(6), LEN);
    assert_eq!(wrong_revision.kind(), ErrorKind::InvalidData);
    let wrong_length = refuse(&name, revision(5), LEN + 1);
    assert_eq!(wrong_length.kind(), ErrorKind::InvalidData);
    creator.retire().expect("retire");
}

#[test]
fn a_reader_outlives_the_creator_and_the_name_goes_with_it() {
    let name = name("reader");
    let bytes = payload();
    let mut creator = Segment::create(&name, revision(1), &bytes).expect("create");
    let mut reader = unsafe { Segment::open(&name, revision(1), LEN) }.expect("open");
    creator.retire().expect("retire");
    assert_eq!(reader.payload(), &bytes[..]);
    let gone = refuse(&name, revision(1), LEN);
    assert_eq!(gone.kind(), ErrorKind::NotFound);
    assert_eq!(gone.raw_os_error(), Some(libc::ENOENT));
    reader.detach().expect("detach");
}

#[test]
fn misuse_is_refused_not_hidden() {
    let name = name("misuse");
    let bytes = payload();
    let mut creator = Segment::create(&name, revision(1), &bytes).expect("create");

    // `payload` after `detach` must panic, not build a slice from a null base.
    let mut reader = unsafe { Segment::open(&name, revision(1), LEN) }.expect("open");
    reader.detach().expect("detach reader");
    let payload_after_detach = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        reader.payload();
    }));
    assert!(payload_after_detach.is_err(), "payload after detach must panic");

    // `retire` on a reader must panic before detaching or unlinking anything.
    // Its Drop then munmaps normally, so the unwind leaks no mapping.
    let mut reader = unsafe { Segment::open(&name, revision(1), LEN) }.expect("open reader");
    let retire_on_reader = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        reader.retire()
    }));
    assert!(retire_on_reader.is_err(), "retire on a reader must panic");

    creator.retire().expect("retire creator");
}
