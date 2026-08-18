/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstddef>
#include <cstdint>
#include <numeric>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <kvikio/file_handle.hpp>
#include <kvikio/statistics/internals.hpp>

#include "utils/utils.hpp"

using kvikio::statistics::Internals;
using kvikio::statistics::internals;

namespace {

class InternalsTest : public testing::Test {
 protected:
  void SetUp() override
  {
    _filepath = _tmp_dir.path() / "test_internals";
    _data.resize(1024ull * 1024ull);
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

TEST_F(InternalsTest, an_interval_is_the_difference_between_two_readings)
{
  auto const before = internals();
  {
    kvikio::FileHandle f{_filepath, "r"};
    std::vector<std::uint64_t> buffer(_data.size());
    f.pread(buffer.data(), nbytes(), 0).get();
  }
  auto const after = internals();
  auto const spent = after.since(before);

  // Totals only ever grow, so an interval is a subtraction and never a negative.
  EXPECT_GE(after.bounce_buffer_acquisitions, before.bounce_buffer_acquisitions);
  EXPECT_EQ(spent.bounce_buffer_acquisitions,
            after.bounce_buffer_acquisitions - before.bounce_buffer_acquisitions);
  // Saturating, so a reading compared against a later one is empty rather than nonsense.
  EXPECT_TRUE(before.since(after).empty());
}

TEST_F(InternalsTest, a_host_read_is_charged_to_the_file_system)
{
  auto const before = internals();
  {
    kvikio::FileHandle f{_filepath, "r"};
    std::vector<std::uint64_t> buffer(_data.size());
    f.pread(buffer.data(), nbytes(), 0).get();
  }
  auto const spent = internals().since(before);

  EXPECT_GT(spent.posix_calls, 0) << "the read has to reach the file system somehow";
  EXPECT_EQ(spent.posix_bytes, nbytes()) << "every byte read is a byte the file system moved";
  EXPECT_GT(spent.posix_transferring, kvikio::Duration::zero());
  // Host memory, so there is nothing to stage through the device.
  EXPECT_EQ(spent.staging_copies, 0);
}

TEST_F(InternalsTest, held_memory_is_what_was_allocated_and_not_given_back)
{
  Internals reading;
  reading.bounce_buffer_bytes_allocated = 64ULL << 20U;
  reading.bounce_buffer_bytes_freed     = 16ULL << 20U;
  EXPECT_EQ(reading.bounce_buffer_bytes_held(), 48ULL << 20U);

  // Buffers are only freed when the configured size changes, so a run that never changed it
  // reports everything it allocated as held.
  Internals const growing{.bounce_buffer_bytes_allocated = 32ULL << 20U};
  EXPECT_EQ(growing.bounce_buffer_bytes_held(), 32ULL << 20U);

  // An interval that gave back more than it took is a fall in held memory, not a negative.
  Internals shrinking;
  shrinking.bounce_buffer_bytes_freed = 8ULL << 20U;
  EXPECT_EQ(shrinking.bounce_buffer_bytes_held(), 0);
}

TEST_F(InternalsTest, a_reading_formats_only_what_was_paid_for)
{
  Internals nothing;
  EXPECT_TRUE(nothing.empty());
  EXPECT_EQ(nothing.report(), "") << "a reading of nothing should say nothing";

  // A run that reused every connection opened none, and still did work.
  Internals reused;
  reused.http_transfers = 1;
  EXPECT_FALSE(reused.empty());

  // A transfer that retried and then gave up never reached the success path, so the retries are
  // all there is to report.
  Internals gave_up;
  gave_up.http_retries = 3;
  EXPECT_FALSE(gave_up.empty());

  Internals reading;
  reading.bounce_buffer_acquisitions       = 4096;
  reading.bounce_buffer_misses             = 3;
  reading.bounce_buffer_allocating         = std::chrono::milliseconds{12};
  reading.bounce_buffer_bytes_allocated    = 48ULL << 20U;
  reading.posix_calls                      = 640;
  reading.posix_bytes                      = 10ULL << 30U;
  reading.posix_transferring               = std::chrono::milliseconds{900};
  reading.posix_short_calls                = 2;
  reading.staging_copies                   = 640;
  reading.staging_bytes                    = 10ULL << 30U;
  reading.staging_copying                  = std::chrono::milliseconds{300};
  reading.alignment_copies                 = 5;
  reading.alignment_bytes                  = 80ULL << 20U;
  reading.alignment_buffer_bytes_allocated = 16ULL << 20U;
  reading.cufile_registrations             = 8;
  reading.cufile_registering               = std::chrono::microseconds{340};
  reading.remote_size_probes               = 12;
  reading.remote_size_probing              = std::chrono::milliseconds{600};
  reading.http_connections                 = 128;
  reading.http_dns                         = std::chrono::milliseconds{40};
  reading.http_connecting                  = std::chrono::milliseconds{900};
  reading.http_tls                         = std::chrono::milliseconds{1900};
  reading.http_transfers                   = 128;
  reading.http_bytes                       = 1ULL << 30U;
  reading.http_waiting                     = std::chrono::milliseconds{40};
  reading.http_receiving                   = std::chrono::milliseconds{800};
  reading.http_retries                     = 7;
  reading.http_retry_backoff               = std::chrono::milliseconds{1200};
  reading.remote_deferred_for_buffer       = 12;
  reading.remote_waiting_for_buffer        = std::chrono::milliseconds{3400};
  reading.remote_deferred_for_slot         = 4;
  reading.remote_waiting_for_slot          = std::chrono::milliseconds{120};

  auto const text = reading.report();
  EXPECT_NE(text.find("bounce buffer"), std::string::npos);
  EXPECT_NE(text.find("4096 acquisitions, 3 allocated in 12 ms, 48 MiB held"), std::string::npos);
  EXPECT_NE(text.find("640 calls, 10 GiB, 900 ms, 2 short"), std::string::npos);
  EXPECT_NE(text.find("device staging       640 copies, 10 GiB, 300 ms"), std::string::npos);
  // The buffer the copies went through is held, and it is not pinned memory.
  EXPECT_NE(text.find("unaligned buffers    5 copies, 80 MiB, 16 MiB held"), std::string::npos);
  EXPECT_NE(text.find("12 remote file sizes, 600 ms"), std::string::npos);
  EXPECT_NE(text.find("128 connections, 40 ms dns, 900 ms connect, 1.90 s tls"), std::string::npos);
  // The rate is what the connections themselves managed, bytes over the time they spent moving.
  EXPECT_NE(text.find("128 requests, 1 GiB, 40 ms waiting, 800 ms receiving, 1.34 GB/s"),
            std::string::npos);
  EXPECT_NE(text.find("http retries         7 retries, 1.20 s backoff"), std::string::npos);
  EXPECT_NE(text.find("waiting for buffers  12 transfers, 3.40 s"), std::string::npos);
  EXPECT_NE(text.find("waiting for slots    4 transfers, 120 ms"), std::string::npos);

  // A group the run touched prints whole, so plain HTTP still reports what TLS cost it.
  Internals plain;
  plain.http_connections = 3;
  plain.http_dns         = std::chrono::microseconds{66};
  plain.http_connecting  = std::chrono::microseconds{237};
  auto const plain_text  = plain.report();
  EXPECT_NE(plain_text.find("3 connections, 66 us dns, 237 us connect, 0 s tls"),
            std::string::npos);
  EXPECT_EQ(plain_text.find("file system"), std::string::npos) << "a group it never touched";
  // Unless everything is asked for, which is how to see what can be recorded at all.
  EXPECT_NE(plain.report(Internals::Rows::ALL).find("file system"), std::string::npos);
  EXPECT_NE(text.find("8 files, 340 us"), std::string::npos);

  auto const json = reading.to_json();
  EXPECT_NE(json.find("\"bounce_buffer_misses\": 3"), std::string::npos);
  EXPECT_NE(json.find("\"http_retries\": 7"), std::string::npos);
  EXPECT_NE(json.find("\"http_retry_backoff_ns\": 1200000000"), std::string::npos);
  // The JSON schema is the same whatever was used, unlike the report.
  EXPECT_NE(nothing.to_json().find("\"cufile_registrations\": 0"), std::string::npos);
}
