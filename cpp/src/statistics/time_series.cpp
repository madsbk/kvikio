/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include <kvikio/detail/observation_recorder.hpp>
#include <kvikio/error.hpp>
#include <kvikio/logger.hpp>
#include <kvikio/logger_macros.hpp>
#include <kvikio/statistics/time_series.hpp>

namespace kvikio {
namespace statistics {

TimeSeriesMonitor::TimeSeriesMonitor(TimeSeriesOptions options)
  : TimeSeriesMonitor{Callback{}, options}
{
}

TimeSeriesMonitor::TimeSeriesMonitor(Callback on_full, TimeSeriesOptions options)
  : _options{options}, _on_full{std::move(on_full)}
{
  KVIKIO_EXPECT(_options.resolution > Duration::zero(),
                "the resolution must be positive",
                std::invalid_argument);
  KVIKIO_EXPECT(_options.capacity > 0, "the capacity must be positive", std::invalid_argument);
  _bins.resize(_options.capacity);
  _in_flight_delta.resize(_options.capacity);
  // Registration comes first, and the span is stamped after it, under the lock. `_registered`
  // starts at its maximum, so an operation that starts in between is refused by both callbacks
  // rather than by only one of them, which would leave the in-flight count unbalanced.
  _registration = register_monitor(this);
  std::lock_guard const lock{_mutex};
  auto const t = detail::now();
  _origin      = t;
  _last_event  = t;
  _registered  = t;
  _anchor      = ClockAnchor::now();
}

TimeSeriesMonitor::~TimeSeriesMonitor() { stop(); }

std::size_t TimeSeriesMonitor::index_of(TimePoint when) const
{
  if (when <= _origin) { return 0; }
  return static_cast<std::size_t>((when - _origin).count() / _options.resolution.count());
}

void TimeSeriesMonitor::charge(TimePoint from, TimePoint to)
{
  auto const first = index_of(from);
  auto const last  = std::min(index_of(to), _bins.size() - 1);
  for (auto i = first; i <= last; ++i) {
    auto const bin_start = _origin + _options.resolution * static_cast<std::int64_t>(i);
    auto const begins    = std::max(from, bin_start);
    auto const ends      = std::min(to, bin_start + _options.resolution);
    if (ends > begins) { _bins[i].totals.busy += ends - begins; }
  }
}

void TimeSeriesMonitor::count_overflow(std::size_t at)
{
  if (at < _bins.size()) { return; }
  auto const past = at - (_bins.size() - 1);
  // Against this window's own high-water mark, so a later overflow adds to the total rather than
  // replacing it.
  if (past > _dropped_high_water) {
    _stats.intervals_dropped += past - _dropped_high_water;
    _dropped_high_water = past;
  }
}

void TimeSeriesMonitor::advance_to(TimePoint when)
{
  // An instant older than the last report is ignored: its slice has been charged already, and the
  // stretch it belongs to was open across it either way.
  if (when <= _last_event) { return; }

  // Window by window, so that busy is charged into each one before it is handed over rather than
  // into whichever window happens to be current when the last report arrives.
  while (true) {
    auto const window_end = _origin + _options.resolution * static_cast<std::int64_t>(_bins.size());
    auto const to         = std::min(when, window_end);
    if (_in_flight > 0 && to > _last_event) { charge(_last_event, to); }
    _last_event = std::max(_last_event, to);
    // An instant exactly on the boundary belongs to the next window, which is where `index_of()`
    // would put it.
    if (when < window_end) { break; }

    // Past the end of the window. With nobody to hand it to, everything past it is dropped and
    // counted, which keeps the beginning of the run.
    if (!_on_full) {
      count_overflow(index_of(when));
      _last_event = when;
      break;
    }
    _reached  = _bins.size() - 1;
    _recorded = true;
    // Handed over a window at a time, so a jump across many of them does not build one batch of
    // all of them.
    auto window = harvest(_reached);
    _stats.intervals_taken += window.size();
    reset_window(window_end);
    hand_over(std::move(window));
  }
}

void TimeSeriesMonitor::reset_window(TimePoint origin)
{
  _origin = origin;
  std::fill(_bins.begin(), _bins.end(), Interval{});
  std::fill(
    _in_flight_delta.begin(), _in_flight_delta.end(), std::pair<std::int64_t, std::int64_t>{});
  // Whatever is in flight spans the boundary, so the new window opens with it outstanding.
  _in_flight_delta[0] = {static_cast<std::int64_t>(_in_flight),
                         static_cast<std::int64_t>(_in_flight_bytes)};
  _reached            = 0;
  _recorded           = false;
  _dropped_high_water = 0;
}

void TimeSeriesMonitor::carry_window(std::size_t from)
{
  if (from == 0) { return; }
  // What is in flight entering the first interval kept, which its own delta is measured against.
  std::int64_t ops{0};
  std::int64_t bytes{0};
  for (std::size_t i = 0; i < from; ++i) {
    ops += _in_flight_delta[i].first;
    bytes += _in_flight_delta[i].second;
  }
  auto const kept = _reached >= from ? _reached - from : 0;
  for (std::size_t i = 0; i + from < _bins.size(); ++i) {
    _bins[i]            = _bins[i + from];
    _in_flight_delta[i] = _in_flight_delta[i + from];
  }
  std::fill(_bins.end() - static_cast<std::ptrdiff_t>(from), _bins.end(), Interval{});
  std::fill(_in_flight_delta.end() - static_cast<std::ptrdiff_t>(from),
            _in_flight_delta.end(),
            std::pair<std::int64_t, std::int64_t>{});
  _in_flight_delta[0].first += ops;
  _in_flight_delta[0].second += bytes;
  _origin += _options.resolution * static_cast<std::int64_t>(from);
  _reached  = kept;
  _recorded = false;
  for (std::size_t i = 0; i <= kept; ++i) {
    if (_bins[i].totals.num_ops != 0 || _bins[i].totals.busy != Duration::zero()) {
      _recorded = true;
      break;
    }
  }
}

std::vector<Interval> TimeSeriesMonitor::harvest(std::size_t upto)
{
  upto = std::min(upto, _bins.size() - 1);
  std::vector<Interval> ret;
  ret.reserve(upto + 1);
  std::int64_t ops{0};
  std::int64_t bytes{0};
  for (std::size_t i = 0; i <= upto; ++i) {
    ops += _in_flight_delta[i].first;
    bytes += _in_flight_delta[i].second;
    auto interval          = _bins[i];
    interval.totals.start  = _origin + _options.resolution * static_cast<std::int64_t>(i);
    interval.totals.end    = interval.totals.start + _options.resolution;
    interval.totals.anchor = _anchor;
    interval.in_flight     = InFlight{static_cast<std::uint64_t>(std::max<std::int64_t>(ops, 0)),
                                  static_cast<std::uint64_t>(std::max<std::int64_t>(bytes, 0))};
    ret.push_back(interval);
  }
  return ret;
}

void TimeSeriesMonitor::on_start(Observation const& observation) noexcept
{
  std::lock_guard const lock{_mutex};
  if (_stopped || observation.start < _registered) { return; }
  // Before the counts move, so a window handed over here carries what was in flight without this
  // operation, which is then filed into the new window like any other.
  advance_to(observation.start);
  ++_in_flight;
  _in_flight_bytes += observation.size;
  // Past the window, with nobody to hand it to, means it is one of the dropped.
  if (auto const i = index_of(observation.start); i < _bins.size()) {
    ++_in_flight_delta[i].first;
    _in_flight_delta[i].second += static_cast<std::int64_t>(observation.size);
    _reached  = std::max(_reached, i);
    _recorded = true;
  }
}

void TimeSeriesMonitor::on_finish(Observation const& observation) noexcept
{
  std::lock_guard const lock{_mutex};
  if (_stopped) { return; }
  advance_to(observation.end);
  // Not ours: it began before we registered, so we never counted its start and must not uncount
  // it now.
  if (observation.start < _registered) { return; }
  if (_in_flight > 0) {
    --_in_flight;
    _in_flight_bytes -= std::min(_in_flight_bytes, static_cast<std::uint64_t>(observation.size));
  }
  if (auto const i = index_of(observation.end); i < _bins.size()) {
    _bins[i].totals.add(observation);
    --_in_flight_delta[i].first;
    _in_flight_delta[i].second -= static_cast<std::int64_t>(observation.size);
    _reached  = std::max(_reached, i);
    _recorded = true;
  }
}

void TimeSeriesMonitor::hand_over(std::vector<Interval>&& batch) noexcept
{
  if (batch.empty() || !_on_full) { return; }
  try {
    _on_full(std::move(batch));
  } catch (std::exception const& e) {
    KVIKIO_LOG_ERROR(std::string("the time-series callback threw: ") + e.what());
  } catch (...) {
    KVIKIO_LOG_ERROR("the time-series callback threw an unknown exception");
  }
}

std::vector<Interval> TimeSeriesMonitor::take()
{
  std::lock_guard const lock{_mutex};
  if (_stopped) {
    if (!_recorded) { return {}; }
    // Nothing is in progress once stopped, so everything goes.
    auto ret = harvest(_reached);
    _stats.intervals_taken += ret.size();
    reset_window(_origin + _options.resolution * static_cast<std::int64_t>(_reached + 1));
    return ret;
  }

  auto const now = detail::now();
  advance_to(now);
  auto const raw = index_of(now);
  count_overflow(raw);
  auto const current = std::min(raw, _bins.size() - 1);
  // Intervals that went by with nothing in them are part of the series, so time passing counts as
  // something to report.
  if (current > 0) { _recorded = true; }
  _reached = std::max(_reached, current);
  if (!_recorded) { return {}; }

  // The interval in progress is kept back until it is complete, or the next call would hand out
  // the same interval again holding a different part of its totals. Intervals below the one `now`
  // falls in are the complete ones, which is not the same as those below the last one reached.
  auto const complete = std::min(raw, _bins.size());
  auto const from     = std::min(_reached + 1, complete);
  if (from == 0) { return {}; }
  auto ret = harvest(from - 1);
  _stats.intervals_taken += ret.size();
  if (raw >= _bins.size()) {
    // The window ran past its end while nobody was collecting, so what was dropped is gone and
    // recording begins again at the interval now in progress.
    reset_window(_origin + _options.resolution * static_cast<std::int64_t>(raw));
  } else {
    carry_window(from);
  }
  return ret;
}

void TimeSeriesMonitor::stop()
{
  std::lock_guard const stopping{_stopping};
  if (_registration == 0) { return; }
  // Waits for a notification in progress, so once this returns no thread is inside this object.
  unregister_monitor(_registration);
  _registration = 0;

  std::lock_guard const lock{_mutex};
  auto const now = detail::now();
  advance_to(now);
  auto const raw = index_of(now);
  count_overflow(raw);
  auto const current = std::min(raw, _bins.size() - 1);
  if (current > 0) { _recorded = true; }
  _reached = std::max(_reached, current);
  _stopped = true;
  if (_on_full && _recorded) {
    auto remaining = harvest(_reached);
    _stats.intervals_taken += remaining.size();
    reset_window(_origin + _options.resolution * static_cast<std::int64_t>(_reached + 1));
    hand_over(std::move(remaining));
  }
}

TimeSeriesStats TimeSeriesMonitor::stats() const noexcept
{
  std::lock_guard const lock{_mutex};
  return _stats;
}

}  // namespace statistics
}  // namespace kvikio
