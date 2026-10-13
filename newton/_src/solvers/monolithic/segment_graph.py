# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Bounded device graphs; host contracts and execution policy stay with callers."""

import time
from collections import Counter, OrderedDict

import warp as wp


class _SegmentGraphCache:
    def __init__(self, device, *, enabled):
        self.device = device
        self.enabled = enabled
        self.entries = {}
        self.captures = Counter()
        self.replays = Counter()
        self.capture_seconds = Counter()
        self.evictions = Counter()

    def run(self, name, parameters, buffers, execute, *, warmup=True):
        if not self.enabled:
            execute()
            return
        stream = wp.get_stream(self.device)
        buffers = tuple(buffer for buffer in buffers if buffer is not None)
        key = (
            stream.cuda_stream,
            parameters,
            tuple((buffer.ptr, buffer.shape, buffer.strides, buffer.dtype, str(buffer.device)) for buffer in buffers),
        )
        entries = self.entries.setdefault(name, OrderedDict())
        entry = entries.get(key)
        if entry is None:
            start = time.perf_counter()
            # Additive chains must skip warmup: capture records their work,
            # and the single launch below performs it exactly once.
            if warmup:
                execute()
            with wp.ScopedCapture(stream=stream) as capture:
                execute()
            graph = capture.graph
            wp.capture_launch(graph, stream=stream)
            # Retain views and their backing allocations for the graph lifetime.
            entries[key] = (graph, buffers, stream)
            self.captures[name] += 1
            self.capture_seconds[name] += time.perf_counter() - start
            # Three candidate buffers with/without tangents must coexist.
            if len(entries) > 6:
                entries.popitem(last=False)
                self.evictions[name] += 1
        else:
            entries.move_to_end(key)
            wp.capture_launch(entry[0], stream=stream)
        self.replays[name] += 1

    def summary(self):
        return {
            "enabled": self.enabled,
            "max_bindings_per_segment": 6,
            "segments": {
                name: {
                    "captures": self.captures[name],
                    "replays": self.replays[name],
                    "resident_graphs": len(entries),
                    "evictions": self.evictions[name],
                    "capture_host_seconds": self.capture_seconds[name],
                }
                for name, entries in self.entries.items()
            },
        }
