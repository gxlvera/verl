# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Adapter from verl's ``(name, tensor)`` generator API to delta flushes.

The verl rollout API speaks "generator of (HF name, tensor)". This module
wraps that generator with the delta machinery and yields one
:class:`DeltaFlush` per bucket boundary -- the checkpoint engine broadcasts
those flushes over NCCL directly.

The first call has to seed the snapshot (no flushes emitted, no engine RPCs);
subsequent calls produce one or more sparse flushes carrying only the
positions and values that actually changed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Generator, Iterable

import torch

from .delta_state import DeltaState
from .encode import (
    DeltaBucket,
    DeltaEncodingName,
    DeltaParam,
    checksum as _checksum,
    encode_chunk,
)

logger = logging.getLogger(__name__)


@dataclass
class DeltaFlush:
    """One ready-to-dispatch flush.

    * ``positions_cpu`` is a uint8 positions blob. The base encode path
      produces it on the host; the sharded engines keep it on the GPU (the wire
      broadcasts from the GPU, so a host round-trip would be pure overhead).
    * ``values_gpu`` stays on the GPU until the checkpoint engine broadcasts it
      over NCCL.
    * ``params`` carries the per-parameter manifest the receiver needs to
      decode the blob (sent alongside the data over the zmq side-channel).
    """

    encoding: DeltaEncodingName
    params: list[DeltaParam]
    positions_cpu: torch.Tensor
    values_gpu: torch.Tensor
    checksum: int

    @property
    def nnz(self) -> int:
        return self.values_gpu.numel()

    @property
    def wire_bytes(self) -> int:
        return (
            self.positions_cpu.numel()
            + self.values_gpu.numel() * self.values_gpu.element_size()
        )


def _chunked(
    iterable: Iterable[tuple[str, torch.Tensor]], chunk_params: int
) -> Iterable[list[tuple[str, torch.Tensor]]]:
    chunk: list[tuple[str, torch.Tensor]] = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) >= chunk_params:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def _materialize_flush(
    bucket: DeltaBucket, encoding: DeltaEncodingName
) -> DeltaFlush:
    positions_u8 = bucket.merged_positions()
    values_gpu = bucket.merged_values()
    params = list(bucket.params)
    bucket.clear()
    # Positions already live on the values' device (encode keeps them there);
    # checksum and the NCCL broadcast both consume them in place.
    cks = _checksum(positions_u8, values_gpu)
    return DeltaFlush(
        encoding=encoding,
        params=params,
        positions_cpu=positions_u8,
        values_gpu=values_gpu,
        checksum=cks,
    )


def iter_delta_flushes(
    weights: Generator[tuple[str, torch.Tensor], None, None],
    state: DeltaState,
    *,
    encoding: DeltaEncodingName,
    bucket_bytes: int,
    chunk_params: int = 16,
) -> Iterable[DeltaFlush]:
    """Wrap ``weights`` into a stream of delta flushes.

    On the very first invocation the snapshot is seeded from ``weights`` and
    no flush is emitted -- callers must catch the empty iterator and skip
    engine dispatch (the receiver is assumed to share the init checkpoint).

    Args:
        weights: ``(name, tensor)`` generator from
            ``actor.engine.get_per_tensor_param()``.
        state: persistent snapshot/streams owned by the worker.
        encoding: positions encoding.
        bucket_bytes: flush threshold; matches
            ``checkpoint_engine.update_weights_bucket_megabytes`` semantics.
        chunk_params: how many parameters per encode chunk. Tuning knob for
            the 1-step H2D prefetch lookahead.
    """
    if not state.seeded:
        seed = list(weights)
        state.seed(seed)
        logger.info("DeltaState seeded with %d HF tensors", len(seed))
        return

    import os as _os
    import time as _time

    _prof = bool(_os.environ.get("VERL_PROFILE_DELTA_SEND"))
    _t = {"pull": 0.0, "prefetch": 0.0, "diff": 0.0, "encode": 0.0}

    def _now():
        if _prof and torch.cuda.is_available():
            torch.cuda.synchronize()
        return _time.perf_counter()

    bucket = DeltaBucket()
    pending_chunk: list[tuple[str, torch.Tensor]] | None = None
    pending_prefetch: tuple[list[torch.Tensor], torch.cuda.Event] | None = None

    _t0 = _now()
    for hf_chunk in _chunked(weights, chunk_params):
        _t1 = _now()
        _t["pull"] += _t1 - _t0
        next_prefetch = state.prefetch_snapshot(hf_chunk)
        _t2 = _now()
        _t["prefetch"] += _t2 - _t1
        if pending_chunk is not None and pending_prefetch is not None:
            diffs = state.compute_diffs(pending_chunk, prefetched=pending_prefetch)
            state.update_snapshot_async(pending_chunk)
            _t3 = _now()
            _t["diff"] += _t3 - _t2
            chunk = encode_chunk(diffs, encoding)
            _t["encode"] += _now() - _t3
            if chunk.params:
                if bucket.should_flush_before_add(chunk, bucket_bytes):
                    yield _materialize_flush(bucket, encoding)
                bucket.add(chunk)
        pending_chunk, pending_prefetch = hf_chunk, next_prefetch
        _t0 = _now()

    if pending_chunk is not None and pending_prefetch is not None:
        _t2 = _now()
        diffs = state.compute_diffs(pending_chunk, prefetched=pending_prefetch)
        state.update_snapshot_async(pending_chunk)
        _t3 = _now()
        _t["diff"] += _t3 - _t2
        chunk = encode_chunk(diffs, encoding)
        _t["encode"] += _now() - _t3
        if chunk.params:
            if bucket.should_flush_before_add(chunk, bucket_bytes):
                yield _materialize_flush(bucket, encoding)
            bucket.add(chunk)

    if bucket.has_updates:
        yield _materialize_flush(bucket, encoding)
    if _prof:
        print(
            f"[delta-base-produce-profile] pull(full_tensor gather)={_t['pull']:.3f}s "
            f"snap_prefetch={_t['prefetch']:.3f}s diff={_t['diff']:.3f}s encode={_t['encode']:.3f}s",
            flush=True,
        )

    state.flush_snapshot()
