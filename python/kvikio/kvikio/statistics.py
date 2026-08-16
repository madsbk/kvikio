# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import Any, ClassVar, TypedDict

from kvikio._lib import statistics as _statistics  # type: ignore


@dataclass(frozen=True)
class Summary:
    """Totals of the I/O KvikIO has performed

    A snapshot taken from a :class:`SummaryMonitor`, whose values do not change once
    read. This class shouldn't be constructed directly, use :meth:`SummaryMonitor.get`
    or :meth:`SummaryMonitor.since`.

    Everything here describes *logical* operations: one ``read()`` is one operation
    however many reads KvikIO issued underneath.

    Every field named ``_ns`` is nanoseconds, and the two timestamps are nanoseconds
    since the Unix epoch, so comparable with ``time.time_ns()``. They are measured on
    a monotonic clock and mapped through an anchor the monitor took when it was
    constructed, so a stepped system clock cannot corrupt any duration, while a long run
    may drift from the wall clock by whatever NTP did to it.
    """

    class BackendTotals(TypedDict):
        """What one backend carried, a value of :attr:`Summary.by_backend`"""

        num_ops: int
        bytes_transferred: int
        total_duration_ns: int
        num_errors: int

    start_unix_ns: int
    """When counting started, or was last reset"""

    end_unix_ns: int
    """When the summary was read"""

    num_ops: int
    """Number of user-facing operations"""

    num_reads: int
    """Number of operations that were reads"""

    num_writes: int
    """Number of operations that were writes"""

    bytes_requested: int
    """Bytes the operations asked for"""

    bytes_transferred: int
    """Bytes actually transferred

    Differs from :attr:`bytes_requested` on a short or failed read.
    """

    bytes_read: int
    """Of the transferred bytes, how many were read"""

    bytes_written: int
    """Of the transferred bytes, how many were written"""

    num_errors: int
    """Number of operations that failed"""

    busy_ns: int
    """Time during which at least one operation was in flight

    An approximation of the union of the operations' spans: overlapping work is counted
    once, the gaps between calls are counted as idle, and it never exceeds
    :attr:`wall_ns`. An idle gap can be counted as busy when a finish reaches the monitor
    after a start that followed it, which takes two threads and a gap shorter than the
    delay between stamping a report and delivering it.
    """

    total_duration_ns: int
    """The operations' durations added up

    Unlike :attr:`busy_ns`, which counts a stretch of time once however many
    operations filled it, this counts every operation. Only completed operations
    contribute.
    """

    by_backend: dict[str, BackendTotals] = field(hash=False)
    """What each backend carried, keyed by the backend's name

    The totals partition the summary's own, every operation belonging to exactly one
    backend. There is no per-backend busy time, that being a union over wall time which
    two backends running at once would both claim.

    Excluded from :func:`hash` as the only unhashable field, and only from that. It
    still takes part in ``==``, so two summaries that differ here are unequal, they
    merely share a hash bucket.
    """

    wall_ns: int
    """Wall-clock span this summary covers

    Between :attr:`start_unix_ns` and :attr:`end_unix_ns`.
    """

    busy_bytes_per_sec: float
    """Throughput while KvikIO was actually busy, or zero if no time was spent busy

    Dividing the bytes by :attr:`wall_ns` instead would make a program that reads for
    10 ms and then computes for 90 ms look ten times slower than its storage really is.
    Multiply by :attr:`busy_fraction` to recover the whole-span rate.

    Understates while an operation is in flight, since its time counts from the moment it
    starts and its bytes only once it completes.
    """

    busy_fraction: float
    """Fraction of the span during which KvikIO was doing something

    Between 0 and 1. At 0.9 the program is nearly always doing I/O, at 0.03 it was idle
    almost throughout.
    """

    mean_duration_ns: int
    """Average time one operation took, or zero if nothing completed"""

    # The C++ summary the fields were read from, kept for the report and the derived
    # numbers, which are computed in C++.
    _handle: ClassVar[Any] = None

    @classmethod
    def _from_handle(cls, handle: _statistics.Summary) -> "Summary":
        ret = cls(**handle.as_dict())
        object.__setattr__(ret, "_handle", handle)
        return ret

    def since(self, previous: "Summary") -> "Summary":
        """Totals for the interval between an earlier reading and this one

        Reporting periodically wants one reading per tick, differenced against the last.
        Two calls to :meth:`SummaryMonitor.since` would leave a gap between them, and an
        operation that completed in the gap would fall into both intervals::

            baseline = monitor.get()
            while running:
                time.sleep(interval)
                now = monitor.get()
                report(now.since(baseline))
                baseline = now

        Parameters
        ----------
        previous
            An earlier reading of the same span.

        Returns
        -------
        The interval's totals.

        Raises
        ------
        ValueError
            If ``previous`` is not an earlier reading of the same span, which covers an
            interval, a reading from another monitor, and one from before a reset.
        """
        return Summary._from_handle(self._handle.since(previous._handle))

    def to_json(self) -> str:
        """Serialize to JSON

        The timestamps are against the wall clock, so another program can line the
        summary up with its own log.

        Returns
        -------
        A JSON object.
        """
        return self._handle.to_json()

    def serialize(self) -> bytes:
        """Serialize to bytes, exactly

        Everything survives, including the clock anchor, so a summary that has been
        through a pipe is still a valid ``previous`` for :meth:`SummaryMonitor.since`.
        Pickling uses this.

        Returns
        -------
        A fixed-size buffer, which only this version of KvikIO reads back.
        """
        return self._handle.serialize()

    @staticmethod
    def deserialize(data: bytes) -> "Summary":
        """Rebuild a summary from :meth:`serialize`

        Parameters
        ----------
        data
            What :meth:`serialize` produced.

        Returns
        -------
        Summary
            The summary.

        Raises
        ------
        ValueError
            If the bytes are not a summary, are truncated, or carry a version this build
            does not know.
        """
        return Summary._from_handle(_statistics.Summary.deserialize(data))

    def __reduce__(self):
        # The C++ handle cannot be pickled, so a summary travels as its bytes.
        return (Summary.deserialize, (self.serialize(),))

    def report(self) -> str:
        """Format a human-readable report of every field

        Byte counts, durations and rates are scaled to readable units. Use
        :meth:`to_json` instead when the output is going to be parsed.

        Returns
        -------
        The report, one field per line, newline-terminated.
        """
        return self._handle.report()

    def __str__(self) -> str:
        return self.report()


class SummaryMonitor:
    """Turns on I/O statistics for the process and accumulates them while it exists

    The intended use is to create one early, keep it, and read it whenever a report is
    wanted::

        monitor = kvikio.SummaryMonitor()
        ...
        print(monitor.get())

    Or scope it to a phase, and ask for the interval::

        with kvikio.SummaryMonitor() as monitor:
            before = monitor.get()
            run_a_phase()
            print(monitor.since(before).busy_bytes_per_sec)

    KvikIO does no counting at all while no monitor exists. Counting happens entirely in
    C++, where the monitor is told when each operation starts and finishes, so Python
    pays only when a reading is taken, not per operation.

    Notes
    -----
    A monitor measures the whole process, not a scope. It counts every thread's I/O while
    it exists and cannot attribute I/O to a particular call, so wrapping a block in one
    measures that block only if nothing else is doing I/O at the same time.

    Monitors are independent: any number can exist at once, nested or overlapping, and
    resetting one has no effect on the others.
    """

    __slots__ = ("_handle",)

    def __init__(self):
        """Create a monitor and begin counting"""
        self._handle = _statistics.SummaryMonitor()

    def get(self) -> Summary:
        """Read the totals accumulated since construction, or since the last reset

        Safe to call repeatedly, and non-destructive.

        Returns
        -------
        Summary
            The totals.
        """
        return Summary._from_handle(self._handle.get())

    def reset(self) -> None:
        """Zero the totals and restart the span, as if the monitor had just been created"""
        self._handle.reset()

    def since(self, previous: Summary) -> Summary:
        """Totals for the interval since an earlier reading

        Parameters
        ----------
        previous : Summary
            An earlier reading from this monitor.

        Returns
        -------
        Summary
            The interval's totals, spanning ``[previous.end_unix_ns, now)``.

        Raises
        ------
        ValueError
            If ``previous`` is not an earlier reading of this monitor's current span.
            See :meth:`Summary.since`.
        """
        return Summary._from_handle(self._handle.since(previous._handle))

    def stop(self) -> None:
        """Stop counting. Idempotent, and one-way, as there is no resuming

        The end of the measured span is fixed here, so later readings describe the
        interval that was measured rather than growing with the process.
        """
        self._handle.stop()

    def __enter__(self) -> "SummaryMonitor":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.stop()


class SamplingMonitor:
    """Record what KvikIO is doing at a fixed rate, and keep it

    A :class:`SummaryMonitor` reading taken every interval, so a run's shape over time is
    visible and not only its totals. The cost is a fixed amount per second, whatever the
    workload does.

    Nothing is written anywhere. Call :meth:`take` to get the samples, one list per
    field.

    Parameters
    ----------
    resolution_ms : float, optional
        Time one sample covers. The totals are counted rather than sampled, so they are
        exact whatever this is set to. What it sets is how precisely a completion is
        placed on the time axis, since an operation's bytes belong to the interval it
        completed in. A bandwidth derived from one interval is an average only where
        several operations completed in it, and ``in_flight_bytes`` is what the rest
        have instead. Memory follows from it rather than from what the workload does,
        so a run of seconds to minutes wants the default and nothing finer than a plot
        can show.
    capacity : int, optional
        Samples held before the oldest are dropped. The default holds a quarter of an
        hour at the default resolution.

    Notes
    -----
    A sample holds two kinds of quantity. The totals are counted, so they are exact
    whatever an operation's duration. ``in_flight`` and ``in_flight_bytes`` are sampled,
    so they miss operations shorter than the interval and see longer ones in every
    sample they span. No operations in flight against many completed means the work is
    finer-grained than the interval.

    Examples
    --------
    >>> import kvikio
    >>> monitor = kvikio.SamplingMonitor()
    >>> ...
    >>> monitor.stop()
    >>> samples = monitor.take()
    >>> samples["in_flight"]
    [0, 12, 31, 8, 0]
    """

    __slots__ = "_handle"

    def __init__(self, resolution_ms: float = 10.0, capacity: int = 100_000) -> None:
        self._handle = _statistics.SamplingMonitor(int(resolution_ms * 1e6), capacity)

    def take(self) -> dict:
        """Take the samples, leaving the monitor empty

        The interval in progress is kept back until it is complete, so consecutive calls
        partition the run rather than sharing an interval. Taking repeatedly loses
        nothing, and a call after :meth:`stop` hands out the last of it::

            while running:
                write(monitor.take())
                time.sleep(a_while)
            monitor.stop()
            write(monitor.take())

        Take at least once every ``capacity`` intervals of wall clock, or intervals are
        dropped and :meth:`stats` says how many.

        Returns
        -------
        dict
            One list per field, oldest first: ``start_unix_ns``, ``end_unix_ns``,
            ``num_ops``, ``num_reads``, ``num_writes``, ``bytes_requested``,
            ``bytes_transferred``, ``bytes_read``, ``bytes_written``, ``num_errors``,
            ``busy_ns``, ``total_duration_ns``, ``in_flight`` and ``in_flight_bytes``.
        """
        return self._handle.take()

    def stop(self) -> None:
        """Stop sampling and take a final sample of the tail

        Idempotent.
        """
        self._handle.stop()

    def stats(self) -> tuple[int, int]:
        """What has been sampled so far

        Returns
        -------
        tuple
            ``(samples_taken, samples_dropped)``. Dropped samples are not zero when
            nobody collected them and the buffer filled.
        """
        return self._handle.stats()

    def __enter__(self) -> "SamplingMonitor":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.stop()
