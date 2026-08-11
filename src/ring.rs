//! Fixed-slot message transport over RMA windows.
//!
//! See `design.md` for the slot layout, sequencing rules, and acknowledge
//! gating. Construction is collective over the configured communicator and
//! requires `MPI_THREAD_MULTIPLE` plus a unified window memory model.

use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use crc32fast::Hasher;
use mpi::collective::CommunicatorCollectives;
use mpi::topology::{Communicator, Rank};

use crate::{CommunicatorRmaExt, Error, MemoryModel, Window};

const WORD: usize = size_of::<u64>();
const CHECKSUM: usize = size_of::<u32>();
const HEADER: usize = 2 * WORD + CHECKSUM;

/// One message observed in a ring.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Message {
    /// Rank that sent the message.
    pub origin: Rank,
    /// Monotonic sequence number within the directed lane.
    pub sequence: u64,
    /// Message payload copied from the slot.
    pub data: Vec<u8>,
}

#[derive(Clone, Copy)]
struct Lane {
    offset: usize,
    depth: usize,
    capacity: usize,
    slot: usize,
    ack: usize,
}

struct Frame {
    sequence: u64,
    data: Vec<u8>,
}

/// Per-lane sender state
struct Outgoing {
    /// Highest sequence put on the wire.
    sent: u64,
    /// Cumulative acknowledgement, as last read from the counter window.
    acked: u64,
    /// Slot-sized scratch for the image, reused across sends.
    /// Only the header, the payload and the guard are written.
    /// The padding between payload and guard is never zeroed because nothing reads it.
    /// The checksum covers the declared length only, and `length` bounds what a receiver copies out.
    image: Vec<u8>,
}

/// Sparse directed message rings packed into collective RMA windows.
///
/// Safe to share between threads on the same process.
/// Lifetime is the underlying MPI window
/// To drop the ring use [`Self::close`] (collective) or let the destructor run symmetrically on every rank.
pub struct Ring {
    slots: Window<u8>,
    acks: Option<Window<u64>>,
    ranks: usize,
    /// Outgoing lanes, indexed by destination
    to: Vec<Option<Lane>>,
    /// Incoming lanes, indexed by origin
    from: Vec<Option<Lane>>,
    // Ranks with incoming lanes
    incoming: Vec<Rank>,
    sent: Vec<Mutex<Outgoing>>,
    seen: Mutex<Vec<u64>>,
    acked: Mutex<Vec<u64>>,
    lost: AtomicU64,
    corrupt: AtomicU64,
    max_lag: AtomicU64,
    waits: AtomicU64,
    wait_ns: AtomicU64,
}

impl Ring {
    /// Construct a safe (overwrite-gated) ring. Collective over `comm`.
    ///
    /// `rings` names the active directed lanes, one `(source, destination,
    /// depth, capacity)` tuple per lane: `depth` is the number of fixed slots
    /// the lane holds, `capacity` the payload bytes one slot fits. Every rank
    /// passes the same list.
    ///
    /// Each source lane gets a cumulative-acknowledgement counter at the source
    /// The senders spin on `yield_now` until the receiver has acked
    /// enough earlier messages to make room in the slot ring. Requires a
    /// unified window memory model otherwise contruction fails
    ///
    /// # Errors
    /// - [`Error::Intercommunicator`] if `comm` is an intercommunicator.
    /// - [`Error::Ring`] for configuration disagreement, invalid or repeated
    ///   lanes, or zero depth or capacity.
    /// - [`Error::Window`] if the window uses a separate memory model, which
    ///   cannot support the local polling the ring relies on.
    /// - Plus whatever [`CommunicatorRmaExt::allocate_window`] returns for
    ///   the slot and counter windows.
    pub fn safe<C: Communicator + ?Sized>(
        comm: &C,
        rings: &[(Rank, Rank, usize, usize)],
    ) -> Result<Self, Error> {
        Self::new(comm, rings, true)
    }

    /// Construct a raw (overwrite-capable) ring. Collective over `comm`.
    ///
    /// `rings` has the same shape as for [`Self::safe`]. No acknowledgement
    /// counter: unread slots are silently overwritten when the depth is
    /// exhausted, and the receiver reports the gap via [`Self::lost`]. The
    /// sender never blocks on a slow peer. Requires a unified window memory
    /// model.
    ///
    /// # Errors
    /// Same conditions as [`Self::safe`].
    pub fn raw<C: Communicator + ?Sized>(
        comm: &C,
        // (Source: Rank, Destination: Rank, Depth: usize, Capacity: usize)
        rings: &[(Rank, Rank, usize, usize)],
    ) -> Result<Self, Error> {
        Self::new(comm, rings, false)
    }

    fn new<C: Communicator + ?Sized>(
        comm: &C,
        rings: &[(Rank, Rank, usize, usize)],
        safe: bool,
    ) -> Result<Self, Error> {
        if comm.test_inter() {
            return Err(Error::Intercommunicator);
        }
        let ranks = usize::try_from(comm.size()).map_err(|_| Error::SizeOverflow)?;
        let me = usize::try_from(comm.rank()).map_err(|_| Error::SizeOverflow)?;

        let mut config: Vec<_> = rings.to_vec();
        config.sort_unstable();
        Self::agree(comm, safe, &config)?;
        for pair in config.windows(2) {
            if pair[0].0 == pair[1].0 && pair[0].1 == pair[1].1 {
                return Err(Error::Ring("a lane is configured twice"));
            }
        }
        for &(source, destination, depth, capacity) in &config {
            if source < 0
                || destination < 0
                || source as usize >= ranks
                || destination as usize >= ranks
            {
                return Err(Error::Ring("lane rank is outside the communicator"));
            }
            if source == destination {
                return Err(Error::Ring("self lanes are not transport"));
            }
            if depth == 0 {
                return Err(Error::Ring("depth must be positive"));
            }
            if capacity == 0 {
                return Err(Error::Ring("capacity must be positive"));
            }
        }

        let mut lengths = vec![0usize; ranks];
        let mut ack_lengths = vec![0usize; ranks];
        let mut to = vec![None; ranks];
        let mut from = vec![None; ranks];
        for &(source, destination, depth, capacity) in &config {
            let slot = capacity
                .checked_add(HEADER + WORD)
                .ok_or(Error::SizeOverflow)?;
            if slot > i32::MAX as usize {
                return Err(Error::CountOverflow);
            }
            let bytes = depth.checked_mul(slot).ok_or(Error::SizeOverflow)?;
            let target = destination as usize;
            let offset = lengths[target];
            lengths[target] = offset.checked_add(bytes).ok_or(Error::SizeOverflow)?;
            // The ack counter index counts the source's outgoing lanes in
            // configuration order. Every rank derives it the same way, even
            // for lanes it does not own: the `from` table needs it to target
            // the counter on the origin's window.
            let ack = ack_lengths[source as usize];
            ack_lengths[source as usize] = ack.checked_add(1).ok_or(Error::SizeOverflow)?;
            let lane = Lane {
                offset,
                depth,
                capacity,
                slot,
                ack,
            };
            // Only lanes where this rank is an endpoint belong to its tables.
            if source as usize == me {
                to[target] = Some(lane);
            }
            if target == me {
                from[source as usize] = Some(lane);
            }
        }

        let slots = comm.allocate_window::<u8>(lengths[me])?;
        let acks = safe
            .then(|| comm.allocate_window::<u64>(ack_lengths[me]))
            .transpose()?;
        let unified = slots.memory_model() == MemoryModel::Unified
            && acks
                .as_ref()
                .is_none_or(|window| window.memory_model() == MemoryModel::Unified);
        let mut models = vec![0u8; ranks];
        comm.all_gather_into(&(unified as u8), &mut models[..]);
        if models.contains(&0) {
            return Err(Error::Window("local ring access requires unified memory"));
        }

        let mut incoming: Vec<_> = config
            .iter()
            .filter_map(|&(source, destination, _, _)| {
                (destination == comm.rank()).then_some(source)
            })
            .collect();
        incoming.sort_unstable();

        Ok(Ring {
            slots,
            acks,
            ranks,
            to,
            from,
            incoming,
            sent: (0..ranks)
                .map(|_| {
                    Mutex::new(Outgoing {
                        sent: 0,
                        acked: 0,
                        image: Vec::new(),
                    })
                })
                .collect(),
            seen: Mutex::new(vec![0; ranks]),
            acked: Mutex::new(vec![0; ranks]),
            lost: AtomicU64::new(0),
            corrupt: AtomicU64::new(0),
            max_lag: AtomicU64::new(0),
            waits: AtomicU64::new(0),
            wait_ns: AtomicU64::new(0),
        })
    }

    fn agree<C: Communicator + ?Sized>(
        comm: &C,
        safe: bool,
        config: &[(Rank, Rank, usize, usize)],
    ) -> Result<(), Error> {
        let ranks = usize::try_from(comm.size()).map_err(|_| Error::SizeOverflow)?;
        let mut counts = vec![0u64; ranks];
        comm.all_gather_into(&(config.len() as u64), &mut counts[..]);
        if counts.iter().any(|&count| count != config.len() as u64) {
            return Err(Error::Ring("configuration differs between ranks"));
        }
        let mut local = Vec::with_capacity(1 + config.len() * 4);
        local.push(safe as u64);
        for &(source, destination, depth, capacity) in config {
            local.extend_from_slice(&[
                source as u64,
                destination as u64,
                depth as u64,
                capacity as u64,
            ]);
        }
        let mut all = vec![0u64; local.len() * ranks];
        comm.all_gather_into(&local[..], &mut all[..]);
        if all.chunks_exact(local.len()).any(|row| row != local) {
            return Err(Error::Ring("configuration differs between ranks"));
        }
        Ok(())
    }

    /// Whether this ring gates overwrites with cumulative acknowledgements.
    pub fn is_safe(&self) -> bool {
        self.acks.is_some()
    }

    /// Slot count configured for the outgoing lane to `destination`, if any.
    pub fn depth(&self, destination: Rank) -> Option<usize> {
        self.peer(destination)
            .ok()
            .and_then(|i| self.to[i])
            .map(|lane| lane.depth)
    }

    /// Per-slot payload capacity configured for the outgoing lane to `destination`, if any.
    pub fn capacity(&self, destination: Rank) -> Option<usize> {
        self.peer(destination)
            .ok()
            .and_then(|i| self.to[i])
            .map(|lane| lane.capacity)
    }

    /// Total messages observed as lost (raw mode only).
    pub fn lost(&self) -> u64 {
        self.lost.load(Ordering::Relaxed)
    }

    /// Total corrupt slot reads: torn header/footer, bad length, or CRC mismatch.
    pub fn corrupt(&self) -> u64 {
        self.corrupt.load(Ordering::Relaxed)
    }

    /// Greatest sequence advance one [`Self::poll`] made on one lane.
    ///
    /// In raw mode this includes gaps counted as lost. In safe mode it is the
    /// number of messages drained.
    pub fn max_lag(&self) -> u64 {
        self.max_lag.load(Ordering::Relaxed)
    }

    /// Number of times a safe-mode sender actually blocked on acknowledgements.
    ///
    /// Refreshing the cached counter does not count; only a sender that
    /// found no headroom after refreshing and had to spin does.
    pub fn waits(&self) -> u64 {
        self.waits.load(Ordering::Relaxed)
    }

    /// Cumulative nanoseconds spent in safe-mode wait spins.
    pub fn wait_ns(&self) -> u64 {
        self.wait_ns.load(Ordering::Relaxed)
    }

    /// Send one message and return its sequence number.
    ///
    /// Completes the put at the target before returning. In safe mode,
    /// spins on `yield_now` if the destination has not yet acknowledged
    /// enough earlier messages to make room in the slot ring; the wait is
    /// recorded in [`Self::waits`] and [`Self::wait_ns`].
    ///
    /// # Errors
    /// - [`Error::Rank`] if `destination` is outside the communicator.
    /// - [`Error::Ring`] if no lane to `destination` is configured, the
    ///   sequence number would overflow, the acknowledgement counter
    ///   regresses or exceeds what was sent, or the send state is poisoned.
    /// - [`Error::Payload`] if `data` does not fit the lane's capacity.
    /// - Whatever the underlying `MPI_Put` fails with (wrapped), e.g.
    ///   [`Error::Mpi`] or [`Error::Range`].
    pub fn send(&self, destination: Rank, data: &[u8]) -> Result<u64, Error> {
        let destination_index = self.peer(destination)?;
        let lane =
            self.to[destination_index].ok_or(Error::Ring("directed lane is not configured"))?;
        if data.len() > lane.capacity {
            return Err(Error::Payload {
                len: data.len(),
                capacity: lane.capacity,
            });
        }

        let mut guard = self.sent[destination_index]
            .lock()
            .map_err(|_| Error::Ring("send state poisoned"))?;
        let Outgoing { sent, acked, image } = &mut *guard;
        let sequence = sent
            .checked_add(1)
            .ok_or(Error::Ring("sequence number exhausted"))?;
        if self.is_safe() && sequence - *acked > lane.depth as u64 {
            // The cached counter only moves when read here, so it is stale
            // roughly once per `depth` sends. Refresh before calling this a
            // wait: otherwise `waits` counts cache misses, not blocked senders.
            *acked = self.acknowledged(lane, *sent, *acked)?;
            if sequence - *acked > lane.depth as u64 {
                self.waits.fetch_add(1, Ordering::Relaxed);
                let started = Instant::now();
                while sequence - *acked > lane.depth as u64 {
                    std::thread::yield_now();
                    *acked = self.acknowledged(lane, *sent, *acked)?;
                }
                self.wait_ns.fetch_add(
                    started.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64,
                    Ordering::Relaxed,
                );
            }
        }

        // One buffer per lane, grown once. A slot-sized allocation and memset
        // per send costs more than the transfer itself on a wide lane.
        if image.len() != lane.slot {
            image.resize(lane.slot, 0);
        }
        let len = data.len() as u64;
        image[..WORD].copy_from_slice(&sequence.to_le_bytes());
        image[WORD..2 * WORD].copy_from_slice(&len.to_le_bytes());
        image[HEADER..HEADER + data.len()].copy_from_slice(data);
        let checksum = Self::checksum(sequence, len, data);
        image[2 * WORD..HEADER].copy_from_slice(&checksum.to_le_bytes());
        image[lane.slot - WORD..].copy_from_slice(&(!sequence).to_le_bytes());

        let position = ((sequence - 1) % lane.depth as u64) as usize;
        self.slots
            .put(destination, lane.offset + position * lane.slot, image)?;
        *sent = sequence;
        Ok(sequence)
    }

    /// Drain every incoming lane and return newly observed messages.
    ///
    /// Does not call MPI. Safe mode delivers in-order; raw mode scans all
    /// slots when one is lapped, sorts the survivors, and counts the gaps
    /// into [`Self::lost`]. The per-origin `seen` cursor only advances.
    ///
    /// # Errors
    /// - [`Error::Lapped`] if a safe lane was overwritten before consumption,
    ///   which the acknowledge gate makes impossible and so indicates a
    ///   corrupted counter or window rather than backpressure.
    /// - Any window read error from the local slot memory.
    pub fn poll(&self) -> Result<Vec<Message>, Error> {
        let mut seen = self
            .seen
            .lock()
            .map_err(|_| Error::Ring("receive state poisoned"))?;
        let mut messages = Vec::new();

        for &origin in &self.incoming {
            let origin_index = origin as usize;
            let lane =
                self.from[origin_index].ok_or(Error::Ring("directed lane is not configured"))?;
            let behind = seen[origin_index];
            let mut read = 0;
            while read < lane.depth {
                let Some(expected) = seen[origin_index].checked_add(1) else {
                    break;
                };
                let position = ((expected - 1) % lane.depth as u64) as usize;
                let Some(found) = self.probe(lane, position)? else {
                    break;
                };
                if found < expected {
                    break;
                }
                if found > expected {
                    // A safe lane cannot lap: the sender cannot reach this slot
                    // again until `expected` has been acknowledged, and an ack
                    // only follows consumption. Reaching here means the counter
                    // or the window has been corrupted, and continuing would
                    // silently turn a guaranteed lane into a lossy one.
                    if self.is_safe() {
                        return Err(Error::Lapped {
                            origin,
                            expected,
                            found,
                        });
                    }
                    let frames = self.recover(lane, seen[origin_index])?;
                    let mut next = expected;
                    for frame in frames {
                        if frame.sequence < next {
                            continue;
                        }
                        self.lost
                            .fetch_add(frame.sequence - next, Ordering::Relaxed);
                        next = frame.sequence.saturating_add(1);
                        seen[origin_index] = frame.sequence;
                        messages.push(Message {
                            origin,
                            sequence: frame.sequence,
                            data: frame.data,
                        });
                    }
                    break;
                }

                let Some(frame) = self.frame(lane, position)? else {
                    break;
                };
                if frame.sequence != expected {
                    break;
                }
                seen[origin_index] = expected;
                messages.push(Message {
                    origin,
                    sequence: expected,
                    data: frame.data,
                });
                read += 1;
            }
            self.max_lag
                .fetch_max(seen[origin_index] - behind, Ordering::Relaxed);
        }
        Ok(messages)
    }

    /// Acknowledge all messages from `origin` through `sequence`.
    ///
    /// Cumulative and idempotent. In safe mode performs one atomic
    /// `MPI_Fetch_and_op` against `origin`'s counter window. In raw mode
    /// this is a no-op (returns `Ok`).
    ///
    /// # Errors
    /// - [`Error::Rank`] if `origin` is outside the communicator.
    /// - [`Error::Ring`] if no incoming lane from `origin` is configured,
    ///   or the counter diverges from the locally tracked value.
    /// - [`Error::Ack`] if `sequence` is past the last received message.
    pub fn ack(&self, origin: Rank, sequence: u64) -> Result<(), Error> {
        let origin_index = self.peer(origin)?;
        let Some(acks) = &self.acks else {
            return Ok(());
        };
        let lane = self.from[origin_index].ok_or(Error::Ring("directed lane is not configured"))?;

        let mut acked = self
            .acked
            .lock()
            .map_err(|_| Error::Ring("acknowledgement state poisoned"))?;
        if sequence <= acked[origin_index] {
            return Ok(());
        }
        let seen = self
            .seen
            .lock()
            .map_err(|_| Error::Ring("receive state poisoned"))?;
        if sequence > seen[origin_index] {
            return Err(Error::Ack {
                origin,
                sequence,
                received: seen[origin_index],
            });
        }
        drop(seen);

        let delta = sequence - acked[origin_index];
        let previous = acks.fetch_add(origin, lane.ack, delta)?;
        if previous != acked[origin_index] {
            return Err(Error::Ring("acknowledgement counter diverged"));
        }
        acked[origin_index] = sequence;
        Ok(())
    }

    /// Close the underlying windows. Collective over the ring communicator.
    ///
    /// Prefer this over relying on `Drop`: the destructor's shutdown runs
    /// the same steps but the collective boundary is implicit.
    pub fn close(self) -> Result<(), Error> {
        let slots = self.slots.close();
        let acks = self.acks.map_or(Ok(()), Window::close);
        slots.and(acks)
    }

    fn probe(&self, lane: Lane, position: usize) -> Result<Option<u64>, Error> {
        let offset = lane.offset + position * lane.slot;
        let mut head = [0; WORD];
        let mut tail = [0; WORD];
        self.slots.read_local_volatile(offset, &mut head)?;
        self.slots
            .read_local_volatile(offset + lane.slot - WORD, &mut tail)?;
        let sequence = Self::word(&head);
        if sequence == 0 {
            return Ok(None);
        }
        if Self::word(&tail) != !sequence {
            self.corrupt.fetch_add(1, Ordering::Relaxed);
            return Ok(None);
        }
        Ok(Some(sequence))
    }

    fn frame(&self, lane: Lane, position: usize) -> Result<Option<Frame>, Error> {
        let offset = lane.offset + position * lane.slot;
        let mut image = vec![0; lane.slot];
        self.slots.read_local(offset, &mut image)?;

        let sequence = Self::word(&image[..WORD]);
        let guard = Self::word(&image[lane.slot - WORD..]);
        if sequence == 0 {
            return Ok(None);
        }
        if guard != !sequence {
            self.corrupt.fetch_add(1, Ordering::Relaxed);
            return Ok(None);
        }
        let len = Self::word(&image[WORD..2 * WORD]);
        let Ok(len) = usize::try_from(len) else {
            self.corrupt.fetch_add(1, Ordering::Relaxed);
            return Ok(None);
        };
        if len > lane.capacity {
            self.corrupt.fetch_add(1, Ordering::Relaxed);
            return Ok(None);
        }
        let checksum = u32::from_le_bytes(image[2 * WORD..HEADER].try_into().unwrap());
        if checksum != Self::checksum(sequence, len as u64, &image[HEADER..HEADER + len]) {
            self.corrupt.fetch_add(1, Ordering::Relaxed);
            return Ok(None);
        }
        Ok(Some(Frame {
            sequence,
            data: image[HEADER..HEADER + len].to_vec(),
        }))
    }

    fn recover(&self, lane: Lane, seen: u64) -> Result<Vec<Frame>, Error> {
        let mut frames = Vec::with_capacity(lane.depth);
        for position in 0..lane.depth {
            if self
                .probe(lane, position)?
                .is_some_and(|sequence| sequence > seen)
                && let Some(frame) = self.frame(lane, position)?
                && frame.sequence > seen
            {
                frames.push(frame);
            }
        }
        frames.sort_unstable_by_key(|frame| frame.sequence);
        frames.dedup_by_key(|frame| frame.sequence);
        Ok(frames)
    }

    /// Re-read `lane`'s cumulative acknowledgement, rejecting a counter that
    /// went backwards or ran past what this rank has sent.
    fn acknowledged(&self, lane: Lane, sent: u64, acked: u64) -> Result<u64, Error> {
        let mut value = [0];
        self.acks
            .as_ref()
            .expect("safe ring has acknowledgements")
            .read_local_volatile(lane.ack, &mut value)?;
        if value[0] < acked {
            return Err(Error::Ring("acknowledgement counter regressed"));
        }
        if value[0] > sent {
            return Err(Error::Ring("acknowledgement exceeds sent sequence"));
        }
        Ok(value[0])
    }

    fn checksum(sequence: u64, len: u64, data: &[u8]) -> u32 {
        let mut checksum = Hasher::new();
        checksum.update(&sequence.to_le_bytes());
        checksum.update(&len.to_le_bytes());
        checksum.update(data);
        checksum.finalize()
    }

    fn word(bytes: &[u8]) -> u64 {
        u64::from_le_bytes(bytes.try_into().unwrap())
    }

    fn peer(&self, rank: Rank) -> Result<usize, Error> {
        if rank < 0 || rank as usize >= self.ranks {
            Err(Error::Rank(rank))
        } else {
            Ok(rank as usize)
        }
    }
}
