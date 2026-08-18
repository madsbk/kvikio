/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <string>

#include <kvikio/observation.hpp>
#include <kvikio/shim/utils.hpp>

namespace KVIKIO_EXPORT kvikio {
namespace statistics {

/**
 * @brief What KvikIO has spent on itself, rather than on I/O.
 *
 * Pinned buffers get allocated, thread pools get built, files get registered with cuFile. These
 * are rare and expensive, and they land at the start of a run, which is where a throughput curve
 * looks worst. The observation facility shows the symptom, operations that are slower than the
 * steady state, and these counters say which of these costs caused it.
 *
 * Remote reads carry a second set, splitting what a transfer spent waiting for the endpoint,
 * receiving bytes, and queued behind one of KvikIO's own limits. Disappointing bandwidth reads
 * differently in each.
 *
 * Every field is a running total for the process, so the cost of an interval is the difference
 * between two readings.
 */
struct Internals {
  /// Pinned bounce buffers handed out.
  std::uint64_t bounce_buffer_acquisitions{};
  /// Of those, the ones that found nothing to reuse and had to allocate.
  std::uint64_t bounce_buffer_misses{};
  /// Time spent allocating them, which is a call into CUDA for pinned memory.
  Duration bounce_buffer_allocating{};
  /// Bytes allocated, and bytes given back. Buffers are only freed when the configured size
  /// changes, so on a whole-process reading the difference is what KvikIO is holding pinned.
  ///
  /// A buffer is a whole `defaults::bounce_buffer_size()` however little the caller asked for, so
  /// a program that read a kilobyte through one still holds the full buffer.
  std::uint64_t bounce_buffer_bytes_allocated{};
  std::uint64_t bounce_buffer_bytes_freed{};

  /// Reads and writes issued to the file system, the bytes they moved, and the time inside the
  /// call. Compare the two against the staging copies below to see which of the two a read
  /// without GDS was limited by.
  std::uint64_t posix_calls{};
  std::uint64_t posix_bytes{};
  Duration posix_transferring{};
  /// Of those, the ones that came back with less than was asked for. A read at the end of a file
  /// legitimately does, so only a large share of the calls says anything.
  std::uint64_t posix_short_calls{};

  /// Copies between a bounce buffer and device memory, the bytes they moved, and the time they
  /// took. This is what a device read pays on top of the file system when GDS is not in use.
  std::uint64_t staging_copies{};
  std::uint64_t staging_bytes{};
  Duration staging_copying{};

  /// Copies made because the caller's host buffer was not page aligned, which Direct I/O requires.
  /// Every byte is copied an extra time, and a caller who aligns their buffer pays none of it.
  std::uint64_t alignment_copies{};
  std::uint64_t alignment_bytes{};
  /// Page-aligned host memory held for those copies. Ordinary host memory, unlike the pinned
  /// bounce buffers above.
  std::uint64_t alignment_buffer_bytes_allocated{};
  std::uint64_t alignment_buffer_bytes_freed{};

  /// Files registered with cuFile.
  std::uint64_t cufile_registrations{};
  /// Time spent registering them.
  Duration cufile_registering{};

  /// Remote file sizes asked for, which is an HTTP round trip each.
  std::uint64_t remote_size_probes{};
  /// Time spent waiting for them.
  Duration remote_size_probing{};

  /// Connections libcurl opened, as opposed to reused.
  std::uint64_t http_connections{};
  /// Time those connections spent resolving, connecting, and shaking hands. libcurl measures
  /// these whether or not anybody asks, so reading them costs nothing.
  Duration http_dns{};
  Duration http_connecting{};
  Duration http_tls{};

  /// Requests that finished, and the bytes they brought back. A file size probe is a request like
  /// any other and is counted here as well as on its own line above.
  std::uint64_t http_transfers{};
  std::uint64_t http_bytes{};
  /// Time between the request going out and the first byte coming back, which is the endpoint
  /// thinking rather than the network moving anything.
  Duration http_waiting{};
  /// Time from the first byte to the last, which is the bytes actually moving. Divide the bytes
  /// by this for what one connection got, with KvikIO's own queueing left out.
  Duration http_receiving{};

  /// Requests the endpoint turned away with a retryable error, and the time spent sleeping before
  /// trying again. A request that was retried twice counts twice. Nothing else records this, since
  /// a retry that eventually succeeds is reported as a success.
  std::uint64_t http_retries{};
  Duration http_retry_backoff{};

  /// Remote transfers that were held back for want of a bounce buffer, and how long they were
  /// held. Time here is network capacity that was available and went unused.
  ///
  /// A transfer that queued behind both limits is counted under both, over the same stretch of
  /// time, so the two waits overlap and adding them says nothing.
  std::uint64_t remote_deferred_for_buffer{};
  Duration remote_waiting_for_buffer{};
  /// Remote transfers that were held back for want of a concurrency slot, and how long they were
  /// held. Time here is KvikIO keeping to the request limit it was configured with.
  std::uint64_t remote_deferred_for_slot{};
  Duration remote_waiting_for_slot{};

  /**
   * @brief The difference between this reading and an earlier one.
   *
   * @param previous An earlier reading.
   * @return What was spent in between, saturating at zero.
   */
  [[nodiscard]] Internals since(Internals const& previous) const noexcept;

  /**
   * @brief Bytes of pinned memory KvikIO is holding.
   *
   * On a whole-process reading this is what it has pinned. On an interval it is how much that
   * grew, since buffers are only freed when the configured buffer size changes.
   *
   * @return Allocated minus freed.
   */
  [[nodiscard]] std::uint64_t bounce_buffer_bytes_held() const noexcept;

  /**
   * @brief Whether anything was counted at all.
   *
   * @return True if every counter is zero.
   */
  [[nodiscard]] bool empty() const noexcept;

  /**
   * @brief Serialise to JSON.
   *
   * @return A JSON object as a string.
   */
  [[nodiscard]] std::string to_json() const;

  /**
   * @brief What `report()` prints.
   */
  enum class Rows : std::uint8_t {
    /// The groups the run touched. A zero within such a group is an answer, so it is printed.
    USED,
    /// Every row, whatever the run did, which is one way to see what is recorded at all.
    ALL,
  };

  /**
   * @brief Format a human-readable report.
   *
   * Grouped by subsystem, and a group the run never touched is left out, so an empty reading
   * formats as an empty string.
   *
   * @code
   *   bounce buffer        4096 acquisitions, 3 allocated in 12.40 ms, 48 MiB held
   *   unaligned buffers    0 copies, 0 B, 0 B held
   * @endcode
   *
   * @param rows Which rows to print.
   * @return The report, newline-terminated, or empty.
   */
  [[nodiscard]] std::string report(Rows rows = Rows::USED) const;
};

/**
 * @brief What KvikIO has spent on itself so far.
 *
 * @return The running totals.
 */
[[nodiscard]] Internals internals() noexcept;

}  // namespace statistics

namespace detail {

/**
 * @brief Record a bounce buffer acquisition. Called wherever one is handed out.
 *
 * @param pinned Whether the pool hands out pinned memory, as opposed to the page-aligned host
 * memory that the alignment copies use.
 * @param allocated Whether it had to be allocated rather than reused.
 * @param allocating How long the allocation took, or zero.
 * @param bytes How big it is, counted only when it was allocated.
 */
void count_bounce_buffer(bool pinned,
                         bool allocated,
                         Duration allocating,
                         std::uint64_t bytes) noexcept;

/**
 * @brief Record a read or write issued to the file system.
 *
 * @param bytes What it moved, which is what it returned rather than what it was asked for.
 * @param short_call Whether it moved less than it was asked for.
 * @param transferring How long the call took.
 */
void count_posix_io(std::uint64_t bytes, bool short_call, Duration transferring) noexcept;

/**
 * @brief Record a copy between a bounce buffer and device memory.
 *
 * @param bytes What it moved.
 * @param copying How long it took, including the wait for the stream.
 */
void count_staging_copy(std::uint64_t bytes, Duration copying) noexcept;

/**
 * @brief Record a copy made to give Direct I/O a page-aligned buffer.
 *
 * @param bytes What it moved.
 */
void count_alignment_copy(std::uint64_t bytes) noexcept;

/**
 * @brief Record a file being registered with cuFile.
 *
 * @param registering How long the registration took.
 */
void count_cufile_registration(Duration registering) noexcept;

/**
 * @brief Record bounce buffer memory being given back.
 *
 * @param pinned Whether the pool hands out pinned memory.
 * @param bytes How much was freed.
 */
void count_bounce_buffer_freed(bool pinned, std::uint64_t bytes) noexcept;

/**
 * @brief Record asking a remote endpoint how big a file is.
 *
 * @param probing How long the round trip took.
 */
void count_remote_size_probe(Duration probing) noexcept;

/**
 * @brief Record what a finished HTTP transfer spent getting connected.
 *
 * @param connections Connections opened, which is zero when one was reused.
 * @param dns Time resolving the name.
 * @param connecting Time establishing the connection.
 * @param tls Time shaking hands, or zero without TLS.
 */
void count_http_connection(std::uint64_t connections,
                           Duration dns,
                           Duration connecting,
                           Duration tls) noexcept;

/**
 * @brief Record what a finished HTTP transfer brought back and how long the two halves took.
 *
 * @param bytes Bytes received.
 * @param waiting Time until the first byte arrived.
 * @param receiving Time from the first byte to the last.
 */
void count_http_transfer(std::uint64_t bytes, Duration waiting, Duration receiving) noexcept;

/**
 * @brief Record an HTTP request that hit a retryable error and will be tried again.
 *
 * @param backoff How long the next attempt waits before it goes out.
 */
void count_http_retry(Duration backoff) noexcept;

/**
 * @brief Record a remote transfer that was held back for want of a bounce buffer.
 *
 * @param waiting How long it was held.
 */
void count_remote_deferred_for_buffer(Duration waiting) noexcept;

/**
 * @brief Record a remote transfer that was held back for want of a concurrency slot.
 *
 * @param waiting How long it was held.
 */
void count_remote_deferred_for_slot(Duration waiting) noexcept;

}  // namespace detail
}  // namespace KVIKIO_EXPORT kvikio
