/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include <kvikio/detail/observation_recorder.hpp>
#include <kvikio/file_handle.hpp>
#include <kvikio/observation.hpp>
#include <kvikio/statistics/sampling.hpp>

#include "utils/utils.hpp"

using kvikio::IoBackend;
using kvikio::MemoryKind;
using kvikio::Observation;
using kvikio::TransferDirection;
using kvikio::statistics::Sample;
using kvikio::statistics::SamplingMonitor;
using kvikio::statistics::SamplingOptions;

namespace {

/// Everything the samples counted, which must match what the run did.
[[nodiscard]] std::uint64_t total_ops(std::vector<Sample> const& samples)
{
  return std::accumulate(samples.begin(), samples.end(), std::uint64_t{0}, [](auto sum, auto& s) {
    return sum + s.totals.num_ops;
  });
}

[[nodiscard]] std::uint64_t total_bytes(std::vector<Sample> const& samples)
{
  return std::accumulate(samples.begin(), samples.end(), std::uint64_t{0}, [](auto sum, auto& s) {
    return sum + s.totals.bytes_transferred;
  });
}

/// Report one operation over `[start, end)`, as the recorders do.
void report(kvikio::TimePoint start, kvikio::TimePoint end, std::size_t size)
{
  Observation observation{};
  observation.start             = start;
  observation.end               = end;
  observation.size              = size;
  observation.bytes_transferred = size;
  kvikio::detail::notify_started(observation);
  kvikio::detail::notify_finished(observation);
}

/// The union of the intervals' busy time, and their summed durations.
[[nodiscard]] kvikio::Duration total_busy(std::vector<Sample> const& samples)
{
  return std::accumulate(
    samples.begin(), samples.end(), kvikio::Duration::zero(), [](auto sum, auto& s) {
      return sum + s.totals.busy;
    });
}

[[nodiscard]] kvikio::Duration total_duration(std::vector<Sample> const& samples)
{
  return std::accumulate(
    samples.begin(), samples.end(), kvikio::Duration::zero(), [](auto sum, auto& s) {
      return sum + s.totals.total_duration;
    });
}

class SamplingTest : public testing::Test {
 protected:
  void SetUp() override
  {
    _filepath = _tmp_dir.path() / "test_sampling";
    _data.resize(1024ull * 128ull);
    std::iota(_data.begin(), _data.end(), 0);

    kvikio::FileHandle f{_filepath, "w"};
    f.pwrite(_data.data(), nbytes()).get();
  }

  [[nodiscard]] std::size_t nbytes() const { return _data.size() * sizeof(std::uint64_t); }

  kvikio::test::TempDir _tmp_dir{};
  std::string _filepath;
  std::vector<std::uint64_t> _data;
};

}  // namespace

TEST_F(SamplingTest, the_samples_add_up_to_the_run)
{
  std::vector<std::uint64_t> buffer(_data.size());
  SamplingMonitor monitor{SamplingOptions{.resolution = std::chrono::milliseconds{2}}};
  {
    kvikio::FileHandle f{_filepath, "r"};
    for (int i = 0; i < 8; ++i) {
      f.pread(buffer.data(), nbytes(), 0).get();
      std::this_thread::sleep_for(std::chrono::milliseconds{3});
    }
  }
  monitor.stop();

  auto const samples = monitor.take();
  ASSERT_FALSE(samples.empty());
  // Nothing is lost between the intervals and nothing is counted twice.
  EXPECT_EQ(total_ops(samples), 8);
  EXPECT_EQ(total_bytes(samples), 8 * nbytes());
  EXPECT_EQ(monitor.stats().samples_dropped, 0);
  EXPECT_EQ(monitor.stats().samples_taken, samples.size());
}

TEST_F(SamplingTest, an_operation_lands_in_the_interval_it_completed_in)
{
  // Reports are filed by their own timestamps, so a series can be pinned rather than raced
  // against a clock.
  constexpr auto resolution = std::chrono::milliseconds{10};
  SamplingMonitor monitor{SamplingOptions{.resolution = resolution}};
  auto const base = kvikio::detail::now();

  // Five back to back, each covering one interval's worth of time.
  for (int i = 0; i < 5; ++i) {
    report(base + resolution * i, base + resolution * (i + 1), 1024);
  }
  monitor.stop();

  auto const samples = monitor.take();
  EXPECT_EQ(total_ops(samples), 5);
  EXPECT_EQ(total_bytes(samples), 5 * 1024);
  // One per interval, wherever the window's origin happens to fall against them.
  auto const filled =
    std::count_if(samples.begin(), samples.end(), [](auto& s) { return s.totals.num_ops > 0; });
  EXPECT_EQ(filled, 5);
  for (auto const& sample : samples) {
    EXPECT_LE(sample.totals.num_ops, 1) << "two operations landed in one interval";
    EXPECT_LE(sample.totals.busy, resolution) << "an interval cannot be busy longer than it is";
  }
  // They were back to back, so the union of their spans is the whole 50 ms.
  EXPECT_EQ(total_busy(samples), resolution * 5);
  // The intervals abut, so the series has no holes in it.
  for (std::size_t i = 1; i < samples.size(); ++i) {
    EXPECT_EQ(samples[i].totals.start, samples[i - 1].totals.end);
  }
}

TEST_F(SamplingTest, overlapping_operations_are_busy_once)
{
  // Busy is the union of the spans, so four operations over the same 4 ms are busy for 4 ms
  // however their durations add up.
  constexpr auto resolution = std::chrono::milliseconds{10};
  constexpr auto span       = std::chrono::milliseconds{4};
  SamplingMonitor monitor{SamplingOptions{.resolution = resolution}};
  auto const base = kvikio::detail::now();

  for (int i = 0; i < 4; ++i) {
    Observation observation{};
    observation.start             = base;
    observation.end               = base + span;
    observation.size              = 512;
    observation.bytes_transferred = 512;
    kvikio::detail::notify_started(observation);
  }
  for (int i = 0; i < 4; ++i) {
    Observation observation{};
    observation.start             = base;
    observation.end               = base + span;
    observation.size              = 512;
    observation.bytes_transferred = 512;
    kvikio::detail::notify_finished(observation);
  }
  monitor.stop();

  auto const samples = monitor.take();
  EXPECT_EQ(total_ops(samples), 4);
  EXPECT_EQ(total_busy(samples), span) << "the overlap was counted more than once";
  EXPECT_EQ(total_duration(samples), span * 4) << "the durations should add up unmerged";
}

TEST_F(SamplingTest, an_operation_longer_than_the_interval_is_in_flight_across_samples)
{
  SamplingMonitor monitor{SamplingOptions{.resolution = std::chrono::milliseconds{2}}};
  {
    // Held open across many intervals, which a counted total cannot see and a sample can.
    kvikio::detail::LogicalObservationRecorder const open{
      IoBackend::POSIX, TransferDirection::READ, MemoryKind::HOST, 0, nbytes()};
    std::this_thread::sleep_for(std::chrono::milliseconds{40});
  }
  monitor.stop();

  auto const samples = monitor.take();
  auto const busy =
    std::count_if(samples.begin(), samples.end(), [](auto& s) { return s.in_flight.num_ops > 0; });
  EXPECT_GE(busy, 5) << "the open operation should be in flight in many samples";
  // Its bytes are outstanding in every one of those, and counted in only the one it landed in.
  auto const outstanding = std::count_if(
    samples.begin(), samples.end(), [&](auto& s) { return s.in_flight.bytes == nbytes(); });
  EXPECT_EQ(outstanding, busy);
  // It completed once, in one interval, however long it ran.
  EXPECT_EQ(total_ops(samples), 1);
}

TEST_F(SamplingTest, an_operation_that_spans_a_take_is_still_counted)
{
  // `take()` moves the window's origin, which says nothing about whether an operation is ours.
  constexpr auto resolution = std::chrono::milliseconds{10};
  SamplingMonitor monitor{SamplingOptions{.resolution = resolution}};
  auto const base = kvikio::detail::now();

  Observation observation{};
  observation.start             = base;
  observation.end               = base + resolution * 4;
  observation.size              = 4096;
  observation.bytes_transferred = 4096;
  kvikio::detail::notify_started(observation);

  // A take hands out the intervals that have finished, so give it some.
  std::this_thread::sleep_for(resolution * 3);
  auto const during = monitor.take();
  ASSERT_FALSE(during.empty());
  EXPECT_GT(during.back().in_flight.num_ops, 0) << "it was open across the take";
  EXPECT_EQ(during.back().in_flight.bytes, 4096) << "its bytes are outstanding";

  kvikio::detail::notify_finished(observation);
  monitor.stop();

  auto const after = monitor.take();
  EXPECT_EQ(total_ops(after), 1) << "a completion spanning a take was thrown away";
  EXPECT_EQ(total_bytes(after), 4096);
  // It was in flight until it landed, and not after, so the last interval is clear of it.
  EXPECT_EQ(after.back().in_flight.num_ops, 0);
  EXPECT_EQ(after.back().in_flight.bytes, 0);
}

TEST_F(SamplingTest, a_take_splits_an_operation_s_busy_but_not_its_bytes)
{
  // What happens to an operation open across a `take()`: its busy is divided between the two
  // batches by time, and its counters land whole in the interval it completed in.
  constexpr auto resolution = std::chrono::milliseconds{10};
  SamplingMonitor monitor{SamplingOptions{.resolution = resolution}};
  auto const base = kvikio::detail::now();

  Observation observation{};
  observation.start             = base;
  observation.end               = base + resolution * 4;
  observation.size              = 4096;
  observation.bytes_transferred = 4096;
  kvikio::detail::notify_started(observation);

  // Long enough for whole intervals to go by while it is open, since a take hands out only the
  // intervals that have finished.
  std::this_thread::sleep_for(resolution * 3);
  auto const first = monitor.take();
  kvikio::detail::notify_finished(observation);
  monitor.stop();
  auto const second = monitor.take();

  // Nothing of the operation's counters is in the first batch, and all of it is in the second.
  EXPECT_EQ(total_ops(first), 0);
  EXPECT_EQ(total_bytes(first), 0);
  EXPECT_EQ(total_ops(second), 1);
  EXPECT_EQ(total_bytes(second), 4096);
  EXPECT_EQ(total_duration(second), resolution * 4) << "its duration lands whole, where it ended";
  // Its busy is shared out by time, so the two batches together hold what it ran for.
  EXPECT_GT(total_busy(first), kvikio::Duration::zero())
    << "the part before the take was not charged";
  EXPECT_GT(total_busy(second), kvikio::Duration::zero())
    << "the part after the take was not charged";
  EXPECT_LE(total_busy(first) + total_busy(second), resolution * 4);
}

TEST_F(SamplingTest, what_is_in_flight_survives_a_rollover)
{
  // The window rolls over under both a take and a full buffer, and what is open spans it.
  constexpr auto resolution = std::chrono::milliseconds{10};
  std::vector<Sample> collected;
  SamplingMonitor monitor{[&](std::vector<Sample>&& batch) {
                            collected.insert(collected.end(), batch.begin(), batch.end());
                          },
                          SamplingOptions{.resolution = resolution, .capacity = 4}};
  auto const base = kvikio::detail::now();

  Observation open{};
  open.start             = base;
  open.end               = base + resolution * 20;
  open.size              = 8192;
  open.bytes_transferred = 8192;
  kvikio::detail::notify_started(open);

  // Far enough ahead to roll the window over several times.
  report(base + resolution * 10, base + resolution * 10 + std::chrono::milliseconds{1}, 256);

  auto const samples = monitor.take();
  for (auto const& sample : samples) {
    EXPECT_EQ(sample.in_flight.num_ops, 1) << "the open operation was lost or duplicated";
    EXPECT_EQ(sample.in_flight.bytes, 8192) << "its bytes were forgotten across the rollover";
  }
  EXPECT_FALSE(collected.empty()) << "the windows the jump crossed were never handed over";
  for (auto const& sample : collected) {
    EXPECT_EQ(sample.in_flight.num_ops, 1) << "the open operation was lost across a rollover";
    EXPECT_EQ(sample.in_flight.bytes, 8192);
  }
}

TEST_F(SamplingTest, a_jump_past_the_window_is_accounted_for)
{
  constexpr auto resolution = std::chrono::milliseconds{10};
  std::vector<Sample> collected;
  {
    SamplingMonitor const monitor{[&](std::vector<Sample>&& batch) {
                                    collected.insert(collected.end(), batch.begin(), batch.end());
                                  },
                                  SamplingOptions{.resolution = resolution, .capacity = 4}};
    auto const base = kvikio::detail::now();
    // Twelve intervals apart, three whole windows, with nothing in between.
    report(base, base + std::chrono::milliseconds{1}, 256);
    report(base + resolution * 12, base + resolution * 12 + std::chrono::milliseconds{1}, 256);
  }

  EXPECT_EQ(total_ops(collected), 2) << "an operation was lost across the jump";
  // The intervals between them are empty but real, and none of them went missing.
  for (std::size_t i = 1; i < collected.size(); ++i) {
    EXPECT_EQ(collected[i].totals.start, collected[i - 1].totals.end)
      << "the series has a hole at " << i;
  }
  EXPECT_GE(collected.size(), 12u) << "the intervals between the two were dropped silently";
}

TEST_F(SamplingTest, a_full_buffer_hands_over_to_the_callback)
{
  constexpr auto resolution = std::chrono::milliseconds{10};
  std::mutex mutex;
  std::vector<Sample> collected;
  {
    SamplingMonitor const monitor{[&](std::vector<Sample>&& batch) {
                                    std::lock_guard const lock{mutex};
                                    collected.insert(collected.end(), batch.begin(), batch.end());
                                  },
                                  SamplingOptions{.resolution = resolution, .capacity = 4}};
    auto const base = kvikio::detail::now();
    // Sixteen intervals through a window that holds four.
    for (int i = 0; i < 16; ++i) {
      report(base + resolution * i, base + resolution * i + std::chrono::milliseconds{1}, 256);
    }
  }

  std::lock_guard const lock{mutex};
  // Handed over rather than dropped, so every operation is somewhere in the batches.
  EXPECT_EQ(total_ops(collected), 16);
  EXPECT_EQ(total_bytes(collected), 16 * 256);
  for (std::size_t i = 1; i < collected.size(); ++i) {
    EXPECT_EQ(collected[i].totals.start, collected[i - 1].totals.end)
      << "a batch boundary lost an interval";
  }
}

TEST_F(SamplingTest, a_full_buffer_with_nobody_listening_drops_the_rest)
{
  constexpr auto resolution = std::chrono::milliseconds{10};
  SamplingMonitor monitor{SamplingOptions{.resolution = resolution, .capacity = 8}};
  auto const base = kvikio::detail::now();
  for (int i = 0; i < 40; ++i) {
    report(base + resolution * i, base + resolution * i + std::chrono::milliseconds{1}, 256);
  }
  monitor.stop();

  auto const samples = monitor.take();
  auto const stats   = monitor.stats();
  EXPECT_GT(stats.samples_dropped, 0) << "a tiny buffer should have overflowed";
  EXPECT_LE(samples.size(), 8) << "the buffer grew past its capacity";
  // What is kept is the beginning of the run, and it is a contiguous series.
  EXPECT_GT(total_ops(samples), 0);
  EXPECT_LT(total_ops(samples), 40) << "nothing was dropped";
  for (std::size_t i = 1; i < samples.size(); ++i) {
    EXPECT_EQ(samples[i].totals.start, samples[i - 1].totals.end);
  }
}

TEST_F(SamplingTest, a_drained_monitor_records_again_after_an_overflow)
{
  constexpr auto resolution = std::chrono::milliseconds{2};
  SamplingMonitor monitor{SamplingOptions{.resolution = resolution, .capacity = 4}};

  // Long enough to run past the window twice over, with nobody collecting.
  std::this_thread::sleep_for(resolution * 12);
  auto const first        = monitor.take();
  auto const dropped_once = monitor.stats().samples_dropped;
  EXPECT_GT(dropped_once, 0u) << "running past the window was not counted";

  // Draining should have brought the window back to the present, so this is counted, not dropped.
  std::vector<std::uint64_t> buffer(_data.size());
  {
    kvikio::FileHandle f{_filepath, "r"};
    f.pread(buffer.data(), nbytes(), 0).get();
  }
  monitor.stop();
  auto const second = monitor.take();
  EXPECT_EQ(total_ops(second), 1) << "the monitor was still stuck in the past";

  // And a second overflow adds to the count rather than replacing it.
  EXPECT_GE(monitor.stats().samples_dropped, dropped_once);
}

TEST_F(SamplingTest, overflowing_twice_counts_twice)
{
  constexpr auto resolution = std::chrono::milliseconds{2};
  SamplingMonitor monitor{SamplingOptions{.resolution = resolution, .capacity = 4}};

  std::this_thread::sleep_for(resolution * 10);
  auto const ignored_a   = monitor.take();
  auto const after_first = monitor.stats().samples_dropped;
  std::this_thread::sleep_for(resolution * 10);
  auto const ignored_b    = monitor.take();
  auto const after_second = monitor.stats().samples_dropped;

  EXPECT_GT(after_first, 0u);
  EXPECT_GT(after_second, after_first) << "the second overflow replaced the first rather than "
                                          "adding to it";
}

TEST_F(SamplingTest, a_window_of_one_interval_can_still_be_taken)
{
  // `capacity` of one is accepted, so it has to work: the single interval is handed out once it
  // is complete rather than never.
  constexpr auto resolution = std::chrono::milliseconds{2};
  SamplingMonitor monitor{SamplingOptions{.resolution = resolution, .capacity = 1}};
  std::vector<std::uint64_t> buffer(_data.size());
  {
    kvikio::FileHandle f{_filepath, "r"};
    f.pread(buffer.data(), nbytes(), 0).get();
  }
  std::this_thread::sleep_for(resolution * 3);

  auto const samples = monitor.take();
  EXPECT_EQ(samples.size(), 1u) << "a one-interval window handed out nothing";
  EXPECT_EQ(total_ops(samples), 1);
}

TEST_F(SamplingTest, a_sample_can_be_related_to_the_wall_clock)
{
  SamplingMonitor monitor{SamplingOptions{.resolution = std::chrono::milliseconds{2}}};
  std::this_thread::sleep_for(std::chrono::milliseconds{20});
  monitor.stop();

  auto const samples = monitor.take();
  ASSERT_FALSE(samples.empty());
  // Every sample carries the anchor, or its timestamps mean nothing outside this process.
  auto const now = std::chrono::system_clock::now();
  for (auto const& sample : samples) {
    EXPECT_LT(now - sample.totals.anchor.to_wall_clock(sample.totals.end), std::chrono::seconds{60})
      << "the anchor did not survive the interval's subtraction";
  }
}

TEST_F(SamplingTest, take_leaves_the_monitor_empty)
{
  constexpr auto resolution = std::chrono::milliseconds{10};
  SamplingMonitor monitor{SamplingOptions{.resolution = resolution}};
  auto const base = kvikio::detail::now();
  for (int i = 0; i < 3; ++i) {
    report(base + resolution * i, base + resolution * i + std::chrono::milliseconds{1}, 256);
  }

  // The interval in progress is kept back, so a take holds what has finished and nothing else.
  auto const first = monitor.take();
  auto const again = monitor.take();
  EXPECT_EQ(total_ops(again), 0) << "the same operations were handed out twice";

  for (int i = 4; i < 7; ++i) {
    report(base + resolution * i, base + resolution * i + std::chrono::milliseconds{1}, 256);
  }
  monitor.stop();
  auto const rest = monitor.take();
  // Taking repeatedly and once more after stopping is lossless: every operation appears exactly
  // once across the batches, and the batches abut.
  EXPECT_EQ(total_ops(first) + total_ops(again) + total_ops(rest), 6);
  EXPECT_EQ(total_ops(monitor.take()), 0) << "there was something left after stopping";
  if (!first.empty() && !rest.empty()) {
    EXPECT_EQ(rest.front().totals.start, first.back().totals.end) << "the series has a hole";
  }
}
