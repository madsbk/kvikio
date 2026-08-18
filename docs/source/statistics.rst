Statistics
==========

.. currentmodule:: kvikio.statistics

KvikIO can report what a run did: how many operations, how many bytes, how long it was
busy, and which backend carried the work.

A monitor accumulates for as long as it exists, and :meth:`~SummaryMonitor.get`
reads the totals so far.

.. literalinclude:: ../../python/kvikio/examples/hello_world.py
    :language: python
    :start-at: import cupy

Printing a summary, or calling :meth:`~Summary.report`, gives a report meant to be read
by a person:

.. code-block:: text

    KvikIO I/O summary
      wall time            238.22 ms
      busy time            5.66 ms (2.38 % of the wall time)
      busy bandwidth       565.50 kB/s
      operations           5 (4 read, 1 write)
      mean duration        1.13 ms
      bytes                3.12 KiB of 3.12 KiB requested (2.34 KiB read, 800 B written)
      errors               0
      backend POSIX        3.12 KiB in 5 ops, 5.66 ms, 565.50 kB/s
      backend GDS          unused
      backend MMAP         unused
      backend REMOTE_HTTP  unused
      backend REMOTE_HDFS  unused
      bounce buffer        5 acquisitions, 2 allocated in 5.45 ms, 32 MiB held
      unaligned buffers    0 copies, 0 B, 0 B held
      file system          5 calls, 3.12 KiB, 19.29 us, 0 short
      device staging       5 copies, 3.12 KiB, 44.02 us
      cufile               0 files, 0 s

Busy time and bandwidth
-----------------------

**Busy time** is the union of the operations' spans, so overlapping work counts once and
the gaps between calls count as idle.

**Busy bandwidth** divides the bytes by that rather than by the wall time. A program that
reads for 10 ms and then computes for 90 ms is doing I/O at its storage's speed for a
tenth of its life, and dividing by the wall time would report it as ten times slower than
it is. Multiply by :attr:`~Summary.busy_fraction` to recover the whole-span rate.

What KvikIO spent on itself
---------------------------

The rows below the backends are :attr:`~Summary.internals`, the work KvikIO did on its own
behalf rather than on the run's. They are what tells a slow endpoint apart from a bounce
buffer that had to be allocated, or from a transfer waiting for one to come free. In the
report above they say that 5.45 ms of the 5.66 ms busy time went on allocating two bounce
buffers, and only 63 us on the file system and the copies to the device.

They come in three groups, buffers, local I/O and remote I/O, and a group appears only if
the run touched it. A zero inside a group that did appear is an answer: ``0 retries`` and
``0 transfers`` waiting for a slot say the run was not retrying and was never held back.
A group that never ran says nothing, so a local program is not told what TLS cost it.

To see every row whatever the run did, which is one way to see what is recorded at all::

    print(summary.report(all_rows=True))

The counters run for the life of the process and a summary holds the part of them that
falls inside its span, so :meth:`~Summary.since` differences them like everything else.

Getting a summary out of the process
------------------------------------

:meth:`~Summary.to_json` gives the same content for anything that would rather
parse it, with the timestamps against the wall clock so another program can line the
summary up with its own log.

:meth:`~Summary.serialize` and :meth:`~Summary.deserialize` move a summary
between processes. They are exact, so one that has been through a pipe is still a valid
``previous`` for :meth:`~Summary.since`, and they are what pickling uses.

.. code-block:: python

    raw = summary.serialize()
    assert kvikio.Summary.deserialize(raw) == summary

The bytes are not a wire format. They are a copy of the C++ struct, readable only by the
same architecture and the same build of KvikIO, and anything else is refused rather than
misread.

Summaries are not additive. Most fields could be added across processes, but busy time
cannot, since two processes are genuinely busy at the same moment.
