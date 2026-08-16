/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <utility>
#include <vector>

#include <kvikio/observation.hpp>
#include <kvikio/shim/utils.hpp>
#include <kvikio/statistics/summary.hpp>

namespace KVIKIO_EXPORT kvikio {
namespace statistics {

/**
 * @brief What KvikIO did during one interval.
 */
struct Sample {
  /**
   * @brief The interval's totals, the same `Summary` a whole run has.
   *
   * Counted rather than sampled, so exact whatever the operations' duration. Bytes belong to the
   * interval an operation *completed* in, since that is when they became known. `busy` is the
   * exception, being the part of the interval something was in flight across, however much of the
   * operation lay outside it.
   */
  Summary totals{};

  /**
   * @brief What was in flight at the end of the interval.
   *
   * An operation open across a `take()` is in flight in every interval it covers on both sides of
   * it, and its bytes are outstanding in each. Its `totals` land whole in the interval it
   * completed in, while its `busy` is shared out by time across the intervals it covered.
   *
   * The rest of a sample is counted as operations complete, so an operation longer than the
   * interval lands whole in one sample and nothing in the ones it spanned. This is what those
   * intervals have instead: the bytes are outstanding in every sample the operation covers.
   */
  InFlight in_flight{};
};

/**
 * @brief How a `SamplingMonitor` behaves.
 */
struct SamplingOptions {
  /**
   * @brief Time one sample covers.
   *
   * The totals are counted rather than sampled, so they are exact whatever this is set to. What
   * it sets is how precisely a completion is placed on the time axis, since an operation's bytes
   * go to the interval it completed in. A bandwidth derived from one interval is an average only
   * where several operations completed in it: a 30 ms transfer landing whole in a 1 ms interval
   * reads as a rate nothing ran at. `Sample::in_flight` is what those intervals have instead.
   *
   * Memory follows from it rather than from what the workload does, so a run of seconds to
   * minutes wants the default and nothing finer than a plot can show.
   */
  Duration resolution{std::chrono::milliseconds{10}};

  /**
   * @brief Intervals held before the callback is invoked, or recording stops keeping them.
   *
   * A sample is about three hundred bytes, so the default holds a quarter of an hour at the
   * default resolution for thirty megabytes. With no callback the buffer simply fills and every
   * later interval is dropped and counted, which keeps the beginning of the run. Anyone wanting
   * something else, the recent past or the samples on disk, hands the monitor a callback.
   */
  std::size_t capacity{100'000};
};

/**
 * @brief What a `SamplingMonitor` has done.
 */
struct SamplingStats {
  /// Intervals recorded.
  std::uint64_t samples_taken{};
  /// Intervals dropped because nobody collected them and the buffer was full.
  std::uint64_t samples_dropped{};
};

/**
 * @brief Records what KvikIO is doing interval by interval, and keeps it.
 *
 * A `Summary` per interval of the run rather than one for the whole of it, so its shape over time
 * is visible and not only its totals. The cost is a fixed number of bytes per second, whatever
 * the workload does.
 *
 * Operations are filed into the interval they completed in, as they are reported, so nothing runs
 * on a clock and there is no thread. `Sample::in_flight` is what an interval has instead of the
 * bytes of an operation that has not landed yet.
 *
 * A report can be separated from the timestamp it carries by scheduling, so one can arrive after
 * the interval it belongs to has already been handed out. It is filed into the earliest interval
 * still held, so its bytes are never lost, and its placement is late by however late the report
 * was.
 *
 * Nothing is written anywhere. The samples are held until `take()` is called, or handed to a
 * callback when the buffer fills.
 *
 * What a sample does not hold is per-operation detail: no latency distribution, no attribution to
 * a file, and no way to spread an operation's bytes over the time it ran.
 *
 * @code
 * kvikio::statistics::SamplingMonitor monitor;
 * run_the_thing();
 * for (auto const& sample : monitor.take()) {
 *   std::cout << sample.totals.busy_bytes_per_sec() << " B/s, " << sample.in_flight.num_ops
 *             << " in flight\n";
 * }
 * @endcode
 *
 * ### Overhead
 *
 * Per operation, what a `SummaryMonitor` costs and no more: 58 ns against no monitor on an 8 B
 * mmap read. `busy` is charged in slices that never overlap, so a whole run costs one pass over
 * its intervals however many operations filled them, `duration / resolution` charges in total.
 *
 * @warning A monitor covers the whole process, not a scope. See `SummaryMonitor` for the rest of
 * the caveats, all of which apply here.
 */
class SamplingMonitor final : private kvikio::Monitor {
 public:
  /**
   * @brief Where samples go when the buffer fills, as an alternative to keeping them.
   *
   * Called with everything held so far, which is then forgotten, on whichever thread reported the
   * operation that filled the buffer. That is a thread doing I/O, and the monitor's lock is held
   * across it, so it must not block and must not call back into this monitor or into KvikIO. Hand
   * the batch to a thread of your own if it is going anywhere slow.
   */
  using Callback = std::function<void(std::vector<Sample>&&)>;

  /**
   * @brief Start recording, keeping what is recorded.
   *
   * @param options See `SamplingOptions`.
   */
  explicit SamplingMonitor(SamplingOptions options = {});

  /**
   * @brief Start recording, handing the samples over as the buffer fills.
   *
   * @param on_full Invoked with the samples whenever `SamplingOptions::capacity` is reached, and
   * once more at `stop()` with whatever is left. Nothing is dropped while it is there. Exceptions
   * it throws are caught and logged.
   * @param options See `SamplingOptions`.
   */
  explicit SamplingMonitor(Callback on_full, SamplingOptions options = {});

  ~SamplingMonitor() override;

  SamplingMonitor(SamplingMonitor const&)            = delete;
  SamplingMonitor& operator=(SamplingMonitor const&) = delete;

  /**
   * @brief Take the samples, leaving the monitor empty.
   *
   * The interval in progress is kept back until it is complete, so consecutive calls partition
   * the run rather than sharing an interval between them. A stopped monitor has no interval in
   * progress and hands out everything.
   *
   * Taking repeatedly loses nothing, so a caller writing the series somewhere can drive it
   * itself, provided it takes at least once every `SamplingOptions::capacity` intervals of wall
   * clock. Beyond that intervals are dropped, and `stats()` says how many.
   *
   * @code
   * while (running) {
   *   for (auto const& sample : monitor.take()) { write(sample); }
   *   sleep_for(a_while);
   * }
   * monitor.stop();
   * for (auto const& sample : monitor.take()) { write(sample); }  // the last of it
   * @endcode
   *
   * @return Every complete interval since the last call, oldest first.
   */
  [[nodiscard]] std::vector<Sample> take();

  /**
   * @brief Stop recording. Idempotent, and called by the destructor.
   *
   * Invokes the callback, if there is one, with whatever is left.
   */
  void stop();

  /**
   * @brief What has been recorded so far.
   *
   * @return The counts. `samples_dropped` is not zero once a callback-less monitor has filled.
   */
  [[nodiscard]] SamplingStats stats() const noexcept;

 private:
  /**
   * @brief The `Monitor` contract.
   *
   * Private overrides, and privately inherited: the registry calls them through the base.
   */
  void on_start(Observation const& observation) noexcept override;
  void on_finish(Observation const& observation) noexcept override;

  /// Charge the time since the last report to the intervals it covers, if anything was in flight
  /// across it, handing each window over as it fills. Call with `_mutex` held.
  void advance_to(TimePoint when);

  /// Add `[from, to)` to the `busy` of every interval it covers. Call with `_mutex` held.
  void charge(TimePoint from, TimePoint to);

  /// Which interval an instant falls in, counted from `_origin`, unbounded by the window's
  /// length. Call with `_mutex` held.
  [[nodiscard]] std::size_t index_of(TimePoint when) const;

  /// Empty the window and open a new one at `origin`, carrying what is in flight into its first
  /// interval. Call with `_mutex` held.
  void reset_window(TimePoint origin);

  /// Open a new window at interval `from`, keeping what it and everything after it hold. Call
  /// with `_mutex` held.
  void carry_window(std::size_t from);

  /// Count the intervals a window has run past its end, once. Call with `_mutex` held.
  void count_overflow(std::size_t at);

  /// Intervals `[0, upto]`, with `in_flight` resolved from the deltas and each interval's span
  /// filled in. Call with `_mutex` held.
  [[nodiscard]] std::vector<Sample> harvest(std::size_t upto);

  /// Hand a batch to the callback. Call with `_mutex` held, so that batches reach it in the
  /// order they were made.
  void hand_over(std::vector<Sample>&& batch) noexcept;

  SamplingOptions _options;
  Callback _on_full;

  mutable std::mutex _mutex;
  ClockAnchor _anchor{};
  std::vector<Sample> _bins;
  /// The change in what is in flight across each interval, summed into a count when read.
  std::vector<std::pair<std::int64_t, std::int64_t>> _in_flight_delta;
  SamplingStats _stats;
  TimePoint _origin{};
  TimePoint _last_event{};
  /**
   * @brief When this monitor registered.
   *
   * What decides whether an operation is ours, which the window's origin cannot, since that moves
   * every time the window rolls over. Until the constructor sets it, no operation is.
   */
  TimePoint _registered{TimePoint::max()};
  /// The furthest interval anything has reached, so a read stops there rather than at capacity.
  std::size_t _reached{0};
  /// The furthest past the window this one has run, so a second overflow adds to the first
  /// rather than replacing it.
  std::size_t _dropped_high_water{0};
  std::uint64_t _in_flight{0};
  /// What the operations in flight asked for, carried into each new window.
  std::uint64_t _in_flight_bytes{0};
  std::uint64_t _registration{0};
  /// Whether anything has happened since the last `take()`, an operation or the passing of an
  /// interval, so that a monitor with nothing to say returns nothing.
  bool _recorded{false};
  bool _stopped{false};
};

}  // namespace statistics
}  // namespace KVIKIO_EXPORT kvikio
