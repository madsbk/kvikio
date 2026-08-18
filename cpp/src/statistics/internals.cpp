/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <atomic>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <locale>
#include <sstream>
#include <string>

#include <kvikio/detail/string_utils.hpp>
#include <kvikio/statistics/internals.hpp>

namespace kvikio {

namespace {

/**
 * @brief The counters themselves.
 *
 * Relaxed throughout. Each is a running total that nothing else is ordered against, and a reading
 * taken while another thread is incrementing is a snapshot of a moving target either way.
 */
struct Counters {
  std::atomic<std::uint64_t> bounce_buffer_acquisitions{0};
  std::atomic<std::uint64_t> bounce_buffer_misses{0};
  std::atomic<std::int64_t> bounce_buffer_allocating_ns{0};
  std::atomic<std::uint64_t> bounce_buffer_bytes_allocated{0};
  std::atomic<std::uint64_t> bounce_buffer_bytes_freed{0};
  std::atomic<std::uint64_t> posix_calls{0};
  std::atomic<std::uint64_t> posix_bytes{0};
  std::atomic<std::int64_t> posix_transferring_ns{0};
  std::atomic<std::uint64_t> posix_short_calls{0};
  std::atomic<std::uint64_t> staging_copies{0};
  std::atomic<std::uint64_t> staging_bytes{0};
  std::atomic<std::int64_t> staging_copying_ns{0};
  std::atomic<std::uint64_t> alignment_copies{0};
  std::atomic<std::uint64_t> alignment_bytes{0};
  std::atomic<std::uint64_t> alignment_buffer_bytes_allocated{0};
  std::atomic<std::uint64_t> alignment_buffer_bytes_freed{0};
  std::atomic<std::uint64_t> cufile_registrations{0};
  std::atomic<std::int64_t> cufile_registering_ns{0};
  std::atomic<std::uint64_t> remote_size_probes{0};
  std::atomic<std::int64_t> remote_size_probing_ns{0};
  std::atomic<std::uint64_t> http_connections{0};
  std::atomic<std::int64_t> http_dns_ns{0};
  std::atomic<std::int64_t> http_connecting_ns{0};
  std::atomic<std::int64_t> http_tls_ns{0};
  std::atomic<std::uint64_t> http_transfers{0};
  std::atomic<std::uint64_t> http_bytes{0};
  std::atomic<std::int64_t> http_waiting_ns{0};
  std::atomic<std::int64_t> http_receiving_ns{0};
  std::atomic<std::uint64_t> http_retries{0};
  std::atomic<std::int64_t> http_retry_backoff_ns{0};
  std::atomic<std::uint64_t> remote_deferred_for_buffer{0};
  std::atomic<std::int64_t> remote_waiting_for_buffer_ns{0};
  std::atomic<std::uint64_t> remote_deferred_for_slot{0};
  std::atomic<std::int64_t> remote_waiting_for_slot_ns{0};
};

/**
 * @brief The one set of counters.
 *
 * Intentionally leaked, since a thread pool can be torn down during static destruction and must
 * still find somewhere to count.
 */
Counters& counters() noexcept
{
  static auto* instance = new Counters{};
  return *instance;
}

/// Add a duration to a counter of nanoseconds.
void add(std::atomic<std::int64_t>& counter, Duration duration) noexcept
{
  counter.fetch_add(duration.count(), std::memory_order_relaxed);
}

}  // namespace

namespace detail {

void count_bounce_buffer(bool pinned,
                         bool allocated,
                         Duration allocating,
                         std::uint64_t bytes) noexcept
{
  auto& all = counters();
  // The unpinned pool exists only to give Direct I/O an aligned buffer, and its use is reported
  // with the copies it was fetched for.
  if (!pinned) {
    if (allocated) {
      all.alignment_buffer_bytes_allocated.fetch_add(bytes, std::memory_order_relaxed);
    }
    return;
  }
  all.bounce_buffer_acquisitions.fetch_add(1, std::memory_order_relaxed);
  if (!allocated) { return; }
  all.bounce_buffer_misses.fetch_add(1, std::memory_order_relaxed);
  add(all.bounce_buffer_allocating_ns, allocating);
  all.bounce_buffer_bytes_allocated.fetch_add(bytes, std::memory_order_relaxed);
}

void count_bounce_buffer_freed(bool pinned, std::uint64_t bytes) noexcept
{
  auto& all = counters();
  if (pinned) {
    all.bounce_buffer_bytes_freed.fetch_add(bytes, std::memory_order_relaxed);
  } else {
    all.alignment_buffer_bytes_freed.fetch_add(bytes, std::memory_order_relaxed);
  }
}

void count_posix_io(std::uint64_t bytes, bool short_call, Duration transferring) noexcept
{
  auto& all = counters();
  all.posix_calls.fetch_add(1, std::memory_order_relaxed);
  all.posix_bytes.fetch_add(bytes, std::memory_order_relaxed);
  add(all.posix_transferring_ns, transferring);
  if (short_call) { all.posix_short_calls.fetch_add(1, std::memory_order_relaxed); }
}

void count_staging_copy(std::uint64_t bytes, Duration copying) noexcept
{
  auto& all = counters();
  all.staging_copies.fetch_add(1, std::memory_order_relaxed);
  all.staging_bytes.fetch_add(bytes, std::memory_order_relaxed);
  add(all.staging_copying_ns, copying);
}

void count_alignment_copy(std::uint64_t bytes) noexcept
{
  auto& all = counters();
  all.alignment_copies.fetch_add(1, std::memory_order_relaxed);
  all.alignment_bytes.fetch_add(bytes, std::memory_order_relaxed);
}

void count_remote_size_probe(Duration probing) noexcept
{
  auto& all = counters();
  all.remote_size_probes.fetch_add(1, std::memory_order_relaxed);
  add(all.remote_size_probing_ns, probing);
}

void count_http_connection(std::uint64_t connections,
                           Duration dns,
                           Duration connecting,
                           Duration tls) noexcept
{
  auto& all = counters();
  all.http_connections.fetch_add(connections, std::memory_order_relaxed);
  add(all.http_dns_ns, dns);
  add(all.http_connecting_ns, connecting);
  add(all.http_tls_ns, tls);
}

void count_http_transfer(std::uint64_t bytes, Duration waiting, Duration receiving) noexcept
{
  auto& all = counters();
  all.http_transfers.fetch_add(1, std::memory_order_relaxed);
  all.http_bytes.fetch_add(bytes, std::memory_order_relaxed);
  add(all.http_waiting_ns, waiting);
  add(all.http_receiving_ns, receiving);
}

void count_http_retry(Duration backoff) noexcept
{
  auto& all = counters();
  all.http_retries.fetch_add(1, std::memory_order_relaxed);
  add(all.http_retry_backoff_ns, backoff);
}

void count_remote_deferred_for_buffer(Duration waiting) noexcept
{
  auto& all = counters();
  all.remote_deferred_for_buffer.fetch_add(1, std::memory_order_relaxed);
  add(all.remote_waiting_for_buffer_ns, waiting);
}

void count_remote_deferred_for_slot(Duration waiting) noexcept
{
  auto& all = counters();
  all.remote_deferred_for_slot.fetch_add(1, std::memory_order_relaxed);
  add(all.remote_waiting_for_slot_ns, waiting);
}

void count_cufile_registration(Duration registering) noexcept
{
  auto& all = counters();
  all.cufile_registrations.fetch_add(1, std::memory_order_relaxed);
  add(all.cufile_registering_ns, registering);
}

}  // namespace detail

namespace statistics {

Internals internals() noexcept
{
  auto& all = counters();
  Internals ret;
  ret.bounce_buffer_acquisitions = all.bounce_buffer_acquisitions.load(std::memory_order_relaxed);
  ret.bounce_buffer_misses       = all.bounce_buffer_misses.load(std::memory_order_relaxed);
  ret.bounce_buffer_allocating =
    Duration{all.bounce_buffer_allocating_ns.load(std::memory_order_relaxed)};
  ret.posix_calls        = all.posix_calls.load(std::memory_order_relaxed);
  ret.posix_bytes        = all.posix_bytes.load(std::memory_order_relaxed);
  ret.posix_transferring = Duration{all.posix_transferring_ns.load(std::memory_order_relaxed)};
  ret.posix_short_calls  = all.posix_short_calls.load(std::memory_order_relaxed);
  ret.staging_copies     = all.staging_copies.load(std::memory_order_relaxed);
  ret.staging_bytes      = all.staging_bytes.load(std::memory_order_relaxed);
  ret.staging_copying    = Duration{all.staging_copying_ns.load(std::memory_order_relaxed)};
  ret.alignment_copies   = all.alignment_copies.load(std::memory_order_relaxed);
  ret.alignment_bytes    = all.alignment_bytes.load(std::memory_order_relaxed);
  ret.alignment_buffer_bytes_allocated =
    all.alignment_buffer_bytes_allocated.load(std::memory_order_relaxed);
  ret.alignment_buffer_bytes_freed =
    all.alignment_buffer_bytes_freed.load(std::memory_order_relaxed);
  ret.cufile_registrations = all.cufile_registrations.load(std::memory_order_relaxed);
  ret.cufile_registering   = Duration{all.cufile_registering_ns.load(std::memory_order_relaxed)};
  ret.bounce_buffer_bytes_allocated =
    all.bounce_buffer_bytes_allocated.load(std::memory_order_relaxed);
  ret.bounce_buffer_bytes_freed = all.bounce_buffer_bytes_freed.load(std::memory_order_relaxed);
  ret.remote_size_probes        = all.remote_size_probes.load(std::memory_order_relaxed);
  ret.remote_size_probing = Duration{all.remote_size_probing_ns.load(std::memory_order_relaxed)};
  ret.http_connections    = all.http_connections.load(std::memory_order_relaxed);
  ret.http_dns            = Duration{all.http_dns_ns.load(std::memory_order_relaxed)};
  ret.http_connecting     = Duration{all.http_connecting_ns.load(std::memory_order_relaxed)};
  ret.http_tls            = Duration{all.http_tls_ns.load(std::memory_order_relaxed)};
  ret.http_transfers      = all.http_transfers.load(std::memory_order_relaxed);
  ret.http_bytes          = all.http_bytes.load(std::memory_order_relaxed);
  ret.http_waiting        = Duration{all.http_waiting_ns.load(std::memory_order_relaxed)};
  ret.http_receiving      = Duration{all.http_receiving_ns.load(std::memory_order_relaxed)};
  ret.http_retries        = all.http_retries.load(std::memory_order_relaxed);
  ret.http_retry_backoff  = Duration{all.http_retry_backoff_ns.load(std::memory_order_relaxed)};
  ret.remote_deferred_for_buffer = all.remote_deferred_for_buffer.load(std::memory_order_relaxed);
  ret.remote_waiting_for_buffer =
    Duration{all.remote_waiting_for_buffer_ns.load(std::memory_order_relaxed)};
  ret.remote_deferred_for_slot = all.remote_deferred_for_slot.load(std::memory_order_relaxed);
  ret.remote_waiting_for_slot =
    Duration{all.remote_waiting_for_slot_ns.load(std::memory_order_relaxed)};
  return ret;
}

Internals Internals::since(Internals const& previous) const noexcept
{
  auto const sub = [](auto lhs, auto rhs) { return lhs > rhs ? lhs - rhs : decltype(lhs){}; };

  Internals ret;
  ret.bounce_buffer_acquisitions =
    sub(bounce_buffer_acquisitions, previous.bounce_buffer_acquisitions);
  ret.bounce_buffer_misses     = sub(bounce_buffer_misses, previous.bounce_buffer_misses);
  ret.bounce_buffer_allocating = sub(bounce_buffer_allocating, previous.bounce_buffer_allocating);
  ret.posix_calls              = sub(posix_calls, previous.posix_calls);
  ret.posix_bytes              = sub(posix_bytes, previous.posix_bytes);
  ret.posix_transferring       = sub(posix_transferring, previous.posix_transferring);
  ret.posix_short_calls        = sub(posix_short_calls, previous.posix_short_calls);
  ret.staging_copies           = sub(staging_copies, previous.staging_copies);
  ret.staging_bytes            = sub(staging_bytes, previous.staging_bytes);
  ret.staging_copying          = sub(staging_copying, previous.staging_copying);
  ret.alignment_copies         = sub(alignment_copies, previous.alignment_copies);
  ret.alignment_bytes          = sub(alignment_bytes, previous.alignment_bytes);
  ret.alignment_buffer_bytes_allocated =
    sub(alignment_buffer_bytes_allocated, previous.alignment_buffer_bytes_allocated);
  ret.alignment_buffer_bytes_freed =
    sub(alignment_buffer_bytes_freed, previous.alignment_buffer_bytes_freed);
  ret.cufile_registrations = sub(cufile_registrations, previous.cufile_registrations);
  ret.cufile_registering   = sub(cufile_registering, previous.cufile_registering);
  ret.bounce_buffer_bytes_allocated =
    sub(bounce_buffer_bytes_allocated, previous.bounce_buffer_bytes_allocated);
  ret.bounce_buffer_bytes_freed =
    sub(bounce_buffer_bytes_freed, previous.bounce_buffer_bytes_freed);
  ret.remote_size_probes  = sub(remote_size_probes, previous.remote_size_probes);
  ret.remote_size_probing = sub(remote_size_probing, previous.remote_size_probing);
  ret.http_connections    = sub(http_connections, previous.http_connections);
  ret.http_dns            = sub(http_dns, previous.http_dns);
  ret.http_connecting     = sub(http_connecting, previous.http_connecting);
  ret.http_tls            = sub(http_tls, previous.http_tls);
  ret.http_transfers      = sub(http_transfers, previous.http_transfers);
  ret.http_bytes          = sub(http_bytes, previous.http_bytes);
  ret.http_waiting        = sub(http_waiting, previous.http_waiting);
  ret.http_receiving      = sub(http_receiving, previous.http_receiving);
  ret.http_retries        = sub(http_retries, previous.http_retries);
  ret.http_retry_backoff  = sub(http_retry_backoff, previous.http_retry_backoff);
  ret.remote_deferred_for_buffer =
    sub(remote_deferred_for_buffer, previous.remote_deferred_for_buffer);
  ret.remote_waiting_for_buffer =
    sub(remote_waiting_for_buffer, previous.remote_waiting_for_buffer);
  ret.remote_deferred_for_slot = sub(remote_deferred_for_slot, previous.remote_deferred_for_slot);
  ret.remote_waiting_for_slot  = sub(remote_waiting_for_slot, previous.remote_waiting_for_slot);
  return ret;
}

std::uint64_t Internals::bounce_buffer_bytes_held() const noexcept
{
  return bounce_buffer_bytes_allocated > bounce_buffer_bytes_freed
           ? bounce_buffer_bytes_allocated - bounce_buffer_bytes_freed
           : 0;
}

bool Internals::empty() const noexcept
{
  return bounce_buffer_acquisitions == 0 && posix_calls == 0 && staging_copies == 0 &&
         alignment_copies == 0 && cufile_registrations == 0 && remote_size_probes == 0 &&
         http_connections == 0 && http_transfers == 0 && http_retries == 0 &&
         remote_deferred_for_buffer == 0 && remote_deferred_for_slot == 0;
}

std::string Internals::to_json() const
{
  std::ostringstream os;
  os.imbue(std::locale::classic());
  os << "{\"bounce_buffer_acquisitions\": " << bounce_buffer_acquisitions
     << ", \"bounce_buffer_misses\": " << bounce_buffer_misses
     << ", \"bounce_buffer_allocating_ns\": " << bounce_buffer_allocating.count()
     << ", \"posix_calls\": " << posix_calls << ", \"posix_bytes\": " << posix_bytes
     << ", \"posix_transferring_ns\": " << posix_transferring.count()
     << ", \"posix_short_calls\": " << posix_short_calls
     << ", \"staging_copies\": " << staging_copies << ", \"staging_bytes\": " << staging_bytes
     << ", \"staging_copying_ns\": " << staging_copying.count()
     << ", \"alignment_copies\": " << alignment_copies
     << ", \"alignment_bytes\": " << alignment_bytes
     << ", \"alignment_buffer_bytes_allocated\": " << alignment_buffer_bytes_allocated
     << ", \"alignment_buffer_bytes_freed\": " << alignment_buffer_bytes_freed
     << ", \"cufile_registrations\": " << cufile_registrations
     << ", \"cufile_registering_ns\": " << cufile_registering.count()
     << ", \"bounce_buffer_bytes_allocated\": " << bounce_buffer_bytes_allocated
     << ", \"bounce_buffer_bytes_freed\": " << bounce_buffer_bytes_freed
     << ", \"remote_size_probes\": " << remote_size_probes
     << ", \"remote_size_probing_ns\": " << remote_size_probing.count()
     << ", \"http_connections\": " << http_connections << ", \"http_dns_ns\": " << http_dns.count()
     << ", \"http_connecting_ns\": " << http_connecting.count()
     << ", \"http_tls_ns\": " << http_tls.count() << ", \"http_transfers\": " << http_transfers
     << ", \"http_bytes\": " << http_bytes << ", \"http_waiting_ns\": " << http_waiting.count()
     << ", \"http_receiving_ns\": " << http_receiving.count()
     << ", \"http_retries\": " << http_retries
     << ", \"http_retry_backoff_ns\": " << http_retry_backoff.count()
     << ", \"remote_deferred_for_buffer\": " << remote_deferred_for_buffer
     << ", \"remote_waiting_for_buffer_ns\": " << remote_waiting_for_buffer.count()
     << ", \"remote_deferred_for_slot\": " << remote_deferred_for_slot
     << ", \"remote_waiting_for_slot_ns\": " << remote_waiting_for_slot.count() << "}";
  return os.str();
}

std::string Internals::report(Rows rows) const
{
  std::ostringstream os;
  // The same width the summary's rows use, so the two read as one report.
  auto const row = [&os](char const* label, std::string const& value) {
    os << "  " << std::left << std::setw(21) << label << value << "\n";
  };
  auto const nbytes = [](std::uint64_t value) { return static_cast<std::int64_t>(value); };

  // By category, so a group appears whole or not at all. A zero within a group the run did touch
  // is an answer, `0 retries` or nothing deferred for want of a slot, while a group it never
  // touched says nothing worth a row.
  auto const everything = rows == Rows::ALL;
  auto const buffers    = everything || bounce_buffer_acquisitions != 0 || alignment_copies != 0;
  auto const local =
    everything || posix_calls != 0 || staging_copies != 0 || cufile_registrations != 0;
  auto const remote = everything || remote_size_probes != 0 || http_connections != 0 ||
                      http_transfers != 0 || http_retries != 0 || remote_deferred_for_buffer != 0 ||
                      remote_deferred_for_slot != 0;

  if (buffers) {
    {
      std::ostringstream value;
      // The time is the allocations', not the acquisitions', since a buffer that came from the
      // pool took none of it.
      value << bounce_buffer_acquisitions << " acquisitions, " << bounce_buffer_misses
            << " allocated in " << detail::format_duration(bounce_buffer_allocating) << ", "
            << detail::format_nbytes(static_cast<std::int64_t>(bounce_buffer_bytes_held()))
            << " held";
      row("bounce buffer", value.str());
    }
    {
      std::ostringstream value;
      auto const held = alignment_buffer_bytes_allocated > alignment_buffer_bytes_freed
                          ? alignment_buffer_bytes_allocated - alignment_buffer_bytes_freed
                          : 0;
      value << alignment_copies << " copies, " << detail::format_nbytes(nbytes(alignment_bytes))
            << ", " << detail::format_nbytes(nbytes(held)) << " held";
      row("unaligned buffers", value.str());
    }
  }

  if (local) {
    {
      std::ostringstream value;
      value << posix_calls << " calls, " << detail::format_nbytes(nbytes(posix_bytes)) << ", "
            << detail::format_duration(posix_transferring) << ", " << posix_short_calls << " short";
      row("file system", value.str());
    }
    {
      std::ostringstream value;
      value << staging_copies << " copies, " << detail::format_nbytes(nbytes(staging_bytes)) << ", "
            << detail::format_duration(staging_copying);
      row("device staging", value.str());
    }
    {
      std::ostringstream value;
      value << cufile_registrations << " files, " << detail::format_duration(cufile_registering);
      row("cufile", value.str());
    }
  }

  if (remote) {
    {
      std::ostringstream value;
      value << remote_size_probes << " remote file sizes, "
            << detail::format_duration(remote_size_probing);
      row("size probes", value.str());
    }
    {
      std::ostringstream value;
      value << http_connections << " connections, " << detail::format_duration(http_dns) << " dns, "
            << detail::format_duration(http_connecting) << " connect, "
            << detail::format_duration(http_tls) << " tls";
      row("http", value.str());
    }
    {
      std::ostringstream value;
      value << http_transfers << " requests, "
            << detail::format_nbytes(static_cast<std::int64_t>(http_bytes)) << ", "
            << detail::format_duration(http_waiting) << " waiting, "
            << detail::format_duration(http_receiving) << " receiving";
      // What the connections themselves managed, which is the ceiling the run was working under.
      if (http_receiving > Duration::zero()) {
        auto const seconds =
          std::chrono::duration_cast<std::chrono::duration<double>>(http_receiving).count();
        value << ", " << detail::format_rate(static_cast<double>(http_bytes) / seconds);
      }
      row("http transfers", value.str());
    }
    {
      std::ostringstream value;
      value << http_retries << " retries, " << detail::format_duration(http_retry_backoff)
            << " backoff";
      row("http retries", value.str());
    }
    {
      std::ostringstream value;
      value << remote_deferred_for_buffer << " transfers, "
            << detail::format_duration(remote_waiting_for_buffer);
      row("waiting for buffers", value.str());
    }
    {
      std::ostringstream value;
      value << remote_deferred_for_slot << " transfers, "
            << detail::format_duration(remote_waiting_for_slot);
      row("waiting for slots", value.str());
    }
  }
  return os.str();
}

}  // namespace statistics
}  // namespace kvikio
