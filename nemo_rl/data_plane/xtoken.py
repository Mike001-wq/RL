# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
"""Bounded, tiled, synchronous single-sample xToken teacher payloads."""

from __future__ import annotations

import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterator, Literal

import torch
from pydantic import BaseModel, PositiveFloat, PositiveInt
from tensordict import TensorDict

from nemo_rl.data_plane.factory import build_data_plane_client
from nemo_rl.data_plane.interfaces import DataPlaneClient, DataPlaneConfig

if TYPE_CHECKING:
    from nemo_rl.distributed.batched_data_dict import BatchedDataDict
    from nemo_rl.models.policy import PolicyConfig
    from nemo_rl.models.policy.lm_policy import Policy

XTOKEN_LOGITS_FIELD = "xtoken_dense_logits"

#: FP32 element size for payload-capacity arithmetic on dense logits tiles.
XTOKEN_FP32_ELEMENT_BYTES = 4

#: Bounded re-checks of explicit tile IDs after a cleanup clear. Ray's kill of
#: a producer actor does not cancel storage RPCs it already submitted, so a
#: tile PUT in flight at kill time can land *after* the clear. Re-listing the
#: partition catches those late rows; a producer that is confirmed dead can
#: only contribute a finite number of them, so a small bound converges.
XTOKEN_CLEANUP_VERIFY_ATTEMPTS = 3
XTOKEN_CLEANUP_RETRY_DELAY_S = 1.0


class XTokenTransportConfig(BaseModel, extra="allow"):
    """Transport selection and bounds; defaults preserve the IPC path.

    ``max_payload_bytes`` bounds the *logical* payload — the full padded
    ``[1, T, V]`` FP32 logits — and is checked against the teacher's full
    forward working set before inference, so raising it requires raising the
    whole GPU budget. ``max_tile_bytes`` bounds one *transport object*: a
    single tile that is staged to CPU for PUT and received for GET. Tiling
    constrains only the objects on the wire/queue and their staging; the
    teacher forward, the student's full receive buffer and the loss working
    set still allocate the complete logits. Defaults make the two bounds
    equal, which keeps the historical single-object transfer behavior.
    """

    backend: Literal["ipc", "tq"] = "ipc"
    max_payload_bytes: PositiveInt = 64 * 1024 * 1024
    max_tile_bytes: PositiveInt = 64 * 1024 * 1024
    timeout_s: PositiveFloat = 120.0


@dataclass(frozen=True)
class XTokenTileSpec:
    """Coordinates of one tile inside the logical ``[1, T, V]`` payload.

    ``seq``/``vocab`` intervals are half-open ``[start, end)``; ``sample_id``
    is the TransferQueue row key this tile is stored under.
    """

    tile_id: int
    sample_id: str
    seq_start: int
    seq_end: int
    vocab_start: int
    vocab_end: int

    @property
    def seq_span(self) -> int:
        """Number of sequence rows this tile carries."""
        return self.seq_end - self.seq_start

    @property
    def vocab_span(self) -> int:
        """Number of vocabulary columns this tile carries."""
        return self.vocab_end - self.vocab_start

    @property
    def shape(self) -> tuple[int, int, int]:
        """Dense FP32 tile shape ``[1, seq_span, vocab_span]``."""
        return (1, self.seq_span, self.vocab_span)

    @property
    def nbytes(self) -> int:
        """Logical application payload bytes, excluding transport overhead."""
        return self.seq_span * self.vocab_span * XTOKEN_FP32_ELEMENT_BYTES

    def where(self) -> str:
        """Human-readable coordinates for error messages and logs."""
        return (
            f"tile_id={self.tile_id} seq=[{self.seq_start},{self.seq_end}) "
            f"vocab=[{self.vocab_start},{self.vocab_end})"
        )


@dataclass(frozen=True)
class XTokenTQManifest:
    """Metadata-only description of one tiled ``[1, T, V]`` FP32 payload.

    The driver holds this *before* the first tile PUT, so a partial publish
    can always be cleaned up by explicit tile IDs. Contains no tensor data.
    """

    partition_id: str
    sample_id: str
    shape: tuple[int, int, int]
    producer_node_id: str
    tiles: tuple[XTokenTileSpec, ...]

    def __post_init__(self) -> None:
        if self.shape[0] != 1:
            raise ValueError("xToken TQ manifests must describe exactly one sample")
        validate_tile_coverage(
            self.tiles, seq_len=self.shape[1], vocab_size=self.shape[2]
        )

    @property
    def nbytes(self) -> int:
        """Logical application payload bytes, excluding transport overhead."""
        return self.shape[1] * self.shape[2] * XTOKEN_FP32_ELEMENT_BYTES

    @property
    def tile_sample_ids(self) -> tuple[str, ...]:
        """Every TransferQueue row key this payload is expected to occupy."""
        return tuple(tile.sample_id for tile in self.tiles)


@dataclass(frozen=True)
class XTokenTQPublishReceipt:
    """Teacher-side acknowledgement of a complete tile publish; metadata only."""

    sample_id: str
    tile_count: int
    put_bytes: int
    put_seconds: float


@dataclass
class XTokenTQReceiveResult:
    """Student-owned IPC descriptors and receipt; contains no tensor payload."""

    handles: list[dict[str, Any]]
    consumer_node_id: str
    nbytes: int
    tile_count: int
    get_seconds: float
    buffer_bytes: int


@dataclass
class _XTokenTQStepScope:
    """Per-step bookkeeping: logical identity plus expected tile row keys."""

    sample_id: str
    tile_ids: list[str] = field(default_factory=list)


def validate_tq_support(
    *,
    data_plane: DataPlaneConfig | None,
    policies: list[PolicyConfig],
    num_nodes: int,
    gpus_per_node: int,
    batch_size: int,
) -> None:
    """Reject unsupported layouts before allocating models or Ray bundles."""
    if data_plane is None or not data_plane["enabled"]:
        raise ValueError("xToken TQ requires data_plane.enabled=true")
    if data_plane["impl"] != "transfer_queue" or data_plane["backend"] != "simple":
        raise ValueError("xToken TQ supports only the TransferQueue simple backend")
    if len(policies) != 2 or num_nodes != 2 or gpus_per_node != 1 or batch_size != 1:
        raise ValueError(
            "xToken TQ requires one teacher, two nodes, one GPU per node and batch=1"
        )
    for policy in policies:
        dtensor = policy["dtensor_cfg"]
        if dtensor["enabled"] is not True:
            raise ValueError("xToken TQ requires DTensor v2")
        if (
            dtensor.get("_v2") is not True
            or dtensor["tensor_parallel_size"] != 1
            or dtensor["context_parallel_size"] != 1
            or policy["train_global_batch_size"] != 1
            or policy["train_micro_batch_size"] != 1
            or policy["dynamic_batching"]["enabled"]
            or policy.get("sequence_packing", {}).get("enabled", False)
        ):
            raise ValueError(
                "xToken TQ requires DTensor v2, TP=CP=DP=GBS=MBS=1, no packing/dynamic batching"
            )


def check_payload_size(*, seq_len: int, vocab_size: int, max_bytes: int) -> None:
    """Check the padded FP32 size before teacher inference or allocation."""
    if seq_len <= 0 or vocab_size <= 0:
        raise ValueError("xToken TQ sequence and vocabulary sizes must be positive")
    nbytes = seq_len * vocab_size * XTOKEN_FP32_ELEMENT_BYTES
    if nbytes > max_bytes:
        raise ValueError(
            f"xToken TQ payload {nbytes} bytes exceeds max_payload_bytes={max_bytes}; "
            "raise max_payload_bytes (the full teacher forward, student receive "
            "buffer and loss working set follow the logical payload) or reduce "
            "the sequence length"
        )


def check_tile_size(*, tile: XTokenTileSpec, max_tile_bytes: int) -> None:
    """Check one tile against the transport-object bound before PUT/GET."""
    if tile.nbytes > max_tile_bytes:
        raise ValueError(
            f"xToken TQ tile {tile.where()} is {tile.nbytes} bytes, exceeding "
            f"max_tile_bytes={max_tile_bytes}"
        )


def plan_logit_tiles(
    *,
    seq_len: int,
    vocab_size: int,
    max_tile_bytes: int,
    partition_id: str,
    sample_id: str,
) -> tuple[XTokenTileSpec, ...]:
    """Deterministically split ``[1, seq_len, vocab_size]`` FP32 into tiles.

    Prefers whole rows per tile (sequence-first): as many rows as fit under
    ``max_tile_bytes``. Only when a single row exceeds the bound is that row
    split along the vocabulary axis. Raises if the bound cannot carry one
    FP32 element. Coverage is exact by construction and re-validated by
    :class:`XTokenTQManifest`.
    """
    if seq_len <= 0 or vocab_size <= 0:
        raise ValueError("xToken TQ sequence and vocabulary sizes must be positive")
    if max_tile_bytes < XTOKEN_FP32_ELEMENT_BYTES:
        raise ValueError(
            f"max_tile_bytes={max_tile_bytes} cannot carry a single FP32 element "
            f"({XTOKEN_FP32_ELEMENT_BYTES} bytes)"
        )
    row_bytes = vocab_size * XTOKEN_FP32_ELEMENT_BYTES
    tiles: list[XTokenTileSpec] = []

    def add(seq_start: int, seq_end: int, vocab_start: int, vocab_end: int) -> None:
        tiles.append(
            XTokenTileSpec(
                tile_id=len(tiles),
                sample_id=f"{sample_id}#tile-{len(tiles):05d}",
                seq_start=seq_start,
                seq_end=seq_end,
                vocab_start=vocab_start,
                vocab_end=vocab_end,
            )
        )

    if row_bytes <= max_tile_bytes:
        rows_per_tile = max(1, max_tile_bytes // row_bytes)
        for seq_start in range(0, seq_len, rows_per_tile):
            add(seq_start, min(seq_start + rows_per_tile, seq_len), 0, vocab_size)
    else:
        cols_per_tile = max(1, max_tile_bytes // XTOKEN_FP32_ELEMENT_BYTES)
        for seq_start in range(seq_len):
            for vocab_start in range(0, vocab_size, cols_per_tile):
                add(
                    seq_start,
                    seq_start + 1,
                    vocab_start,
                    min(vocab_start + cols_per_tile, vocab_size),
                )
    assert tiles, "plan_logit_tiles must produce at least one tile"
    return tuple(tiles)


def validate_tile_coverage(
    tiles: tuple[XTokenTileSpec, ...], *, seq_len: int, vocab_size: int
) -> None:
    """Require tiles to exactly cover ``[0, seq_len) x [0, vocab_size)``.

    Accepts exactly the row-major covers :func:`plan_logit_tiles` emits —
    full-vocabulary row blocks plus single-row column blocks — with no gap
    and no overlap. Row keys and tile IDs must be unique.
    """
    tile_ids = [tile.tile_id for tile in tiles]
    sample_ids = [tile.sample_id for tile in tiles]
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("xToken TQ tiles carry duplicate sample IDs")
    if len(set(tile_ids)) != len(tile_ids):
        raise ValueError("xToken TQ tiles carry duplicate tile IDs")
    frontier_seq, frontier_vocab = 0, 0
    for tile in sorted(tiles, key=lambda t: (t.seq_start, t.vocab_start)):
        if not (0 <= tile.seq_start < tile.seq_end <= seq_len):
            raise ValueError(f"xToken TQ tile {tile.where()} has invalid rows")
        if not (0 <= tile.vocab_start < tile.vocab_end <= vocab_size):
            raise ValueError(f"xToken TQ tile {tile.where()} has invalid columns")
        if (tile.seq_start, tile.vocab_start) != (frontier_seq, frontier_vocab):
            raise ValueError(
                f"xToken TQ tiles leave a gap or overlap at {tile.where()}; "
                f"expected coverage to resume at "
                f"seq={frontier_seq} vocab={frontier_vocab}"
            )
        if tile.vocab_end == vocab_size:
            frontier_seq, frontier_vocab = tile.seq_end, 0
        else:
            frontier_seq, frontier_vocab = tile.seq_start, tile.vocab_end
    if (frontier_seq, frontier_vocab) != (seq_len, 0):
        raise ValueError(
            f"xToken TQ tiles cover only seq=[0,{frontier_seq}) vocab=[0,{vocab_size}) "
            f"of the declared [1, {seq_len}, {vocab_size}] payload"
        )


def publish_logit_tile(
    client: DataPlaneClient,
    tile_payload: torch.Tensor,
    *,
    tile: XTokenTileSpec,
    partition_id: str,
    max_tile_bytes: int,
) -> None:
    """PUT one contiguous CPU FP32 tile row; called directly from its worker."""
    if (
        tile_payload.layout != torch.strided
        or tile_payload.ndim != 3
        or tile_payload.shape[0] != 1
        or tile_payload.dtype != torch.float32
    ):
        raise ValueError(
            f"xToken TQ requires a dense FP32 {tuple(tile.shape)} tile at {tile.where()}"
        )
    if tuple(tile_payload.shape) != tile.shape:
        raise ValueError(
            f"xToken TQ tile payload shape {tuple(tile_payload.shape)} does not "
            f"match {tile.where()} expected {tile.shape}"
        )
    check_tile_size(tile=tile, max_tile_bytes=max_tile_bytes)
    client.put_samples(
        sample_ids=[tile.sample_id],
        partition_id=partition_id,
        fields=TensorDict({XTOKEN_LOGITS_FIELD: tile_payload}, batch_size=[1]),
    )


def fetch_logit_tile(
    client: DataPlaneClient,
    tile: XTokenTileSpec,
    *,
    partition_id: str,
    max_tile_bytes: int,
) -> torch.Tensor:
    """GET one tile by its explicit row key and validate it against its spec."""
    check_tile_size(tile=tile, max_tile_bytes=max_tile_bytes)
    fields = client.get_samples(
        sample_ids=[tile.sample_id],
        partition_id=partition_id,
        select_fields=[XTOKEN_LOGITS_FIELD],
    )
    payload = fields[XTOKEN_LOGITS_FIELD]
    if (
        not isinstance(payload, torch.Tensor)
        or tuple(payload.shape) != tile.shape
        or payload.dtype != torch.float32
        or payload.layout != torch.strided
    ):
        raise ValueError(
            f"xToken TQ tile at {tile.where()} has shape/dtype "
            f"{tuple(payload.shape) if isinstance(payload, torch.Tensor) else type(payload)} "
            f"{payload.dtype if isinstance(payload, torch.Tensor) else ''}, "
            f"expected a strided FP32 {tile.shape}"
        )
    return payload


def select_tq_nodes(
    nodes: list[dict[str, Any]],
) -> tuple[dict[str, float], dict[str, float]]:
    """Select two distinct live GPU nodes using their advertised node resources."""
    candidates = sorted(
        (n for n in nodes if n["Alive"] and n["Resources"].get("GPU", 0) >= 1),
        key=lambda n: n["NodeID"],
    )
    if len(candidates) < 2:
        raise ValueError(
            "xToken TQ requires two live Ray nodes with at least one GPU each"
        )
    constraints = []
    for node in candidates[:2]:
        resource = f"node:{node['NodeManagerAddress']}"
        if resource not in node["Resources"]:
            raise ValueError(f"Ray node {node['NodeID']} does not advertise {resource}")
        constraints.append({resource: 0.001})
    return constraints[0], constraints[1]


class XTokenTQTransport:
    """Driver-owned lifetime for synchronous TQ tile sets and their workers."""

    def __init__(
        self,
        *,
        config: XTokenTransportConfig,
        data_plane: DataPlaneConfig,
        teacher: Policy,
        student: Policy,
    ) -> None:
        self.config = config
        self.data_plane = data_plane
        self.teacher = teacher
        self.student = student
        self.partition_id = f"xtoken-{uuid.uuid4().hex}"
        self.client: DataPlaneClient | None = None
        self._steps: list[_XTokenTQStepScope] = []
        self.metrics: dict[str, float] = {}
        self._failed = False
        self._stopped = False

    def __enter__(self) -> XTokenTQTransport:
        # Ray is optional for the pure payload helpers and CPU unit tests.
        import ray

        try:
            self.client = build_data_plane_client(self.data_plane, bootstrap=True)
            self.client.register_partition(
                self.partition_id, [XTOKEN_LOGITS_FIELD], 1, []
            )
            for policy in (self.teacher, self.student):
                ray.get(
                    policy.worker_group.run_all_workers_single_data(
                        "setup_data_plane", cfg=self.data_plane
                    ),
                    timeout=self.config.timeout_s,
                )
        except BaseException:
            try:
                self._stop_workers()
            finally:
                if self.client is not None:
                    self.client.close()
            raise
        return self

    def __exit__(self, *exc: object) -> None:
        try:
            if exc[0] is not None:
                self._stop_workers()
            if not self._failed:
                self.student.release_ipc_buffer()
        finally:
            if self.client is not None:
                self.client.close()

    def _stop_workers(self) -> None:
        # Actor termination prevents timed-out calls from reusing the buffers.
        # Killing the producer actor does not cancel storage RPCs it already
        # submitted — a tile PUT in flight can still land afterwards. That
        # residual window is what the post-clear verification in
        # _clear_tile_ids exists to close.
        import ray

        if self._stopped:
            return
        self._failed = True
        workers = self.teacher.worker_group.workers + self.student.worker_group.workers
        for worker in workers:
            ray.kill(worker, no_restart=True)
        # These queued calls must fail with ActorDiedError before any row is
        # cleared. A stop timeout aborts cleanup rather than racing a live PUT.
        for worker in workers:
            try:
                ray.get(
                    worker.get_free_memory_bytes.remote(), timeout=self.config.timeout_s
                )
            except ray.exceptions.ActorDiedError:
                pass
        self._stopped = True

    def _clear_tile_ids(self, scope: _XTokenTQStepScope) -> None:
        """Clear exactly the expected tile IDs, then verify they are gone."""
        assert self.client is not None
        tile_ids = list(scope.tile_ids)
        if not tile_ids:
            return
        leftover: list[str] = tile_ids
        for attempt in range(XTOKEN_CLEANUP_VERIFY_ATTEMPTS):
            self.client.clear_samples(tile_ids, self.partition_id)
            remaining = set(self.client.list_sample_ids(self.partition_id))
            leftover = [tile_id for tile_id in tile_ids if tile_id in remaining]
            if not leftover:
                print(
                    f"XTOKEN_TQ_CLEARED sample_id={scope.sample_id} "
                    f"tiles={len(tile_ids)} remaining_rows={len(remaining)}",
                    flush=True,
                )
                return
            # A confirmed-dead producer can only be re-adding rows via storage
            # RPCs that were in flight when it was killed; retry briefly.
            if attempt + 1 < XTOKEN_CLEANUP_VERIFY_ATTEMPTS:
                time.sleep(XTOKEN_CLEANUP_RETRY_DELAY_S)
        raise RuntimeError(
            f"xToken TQ cleanup for sample_id={scope.sample_id} could not clear "
            f"tile rows {leftover} after {XTOKEN_CLEANUP_VERIFY_ATTEMPTS} attempts; "
            "a producer storage RPC may still be in flight"
        )

    @contextmanager
    def step(self) -> Iterator[None]:
        """Retain a unique payload until training/evaluation has completed."""
        if self.client is None:
            raise RuntimeError("xToken TQ transport has not been initialized")
        scope = _XTokenTQStepScope(sample_id=f"{self.partition_id}/{uuid.uuid4().hex}")
        self._steps.append(scope)
        try:
            yield
        except BaseException as step_error:
            try:
                self._stop_workers()
                self._clear_tile_ids(scope)
            except Exception as cleanup_error:
                raise BaseExceptionGroup(
                    "xToken TQ step and cleanup failed", [step_error, cleanup_error]
                ) from None
            raise
        else:
            self._clear_tile_ids(scope)
        finally:
            self._steps.pop()

    def transfer(self, data: BatchedDataDict[Any]) -> list[dict[str, Any]]:
        """Publish remotely, then expose student-local descriptors to the loss."""
        if not self._steps:
            raise RuntimeError("xToken TQ transfer requires an active step scope")
        scope = self._steps[-1]
        manifest = self.teacher.prepare_logits_tq(
            data,
            partition_id=self.partition_id,
            sample_id=scope.sample_id,
            max_payload_bytes=self.config.max_payload_bytes,
            max_tile_bytes=self.config.max_tile_bytes,
            timeout_s=self.config.timeout_s,
        )
        if manifest.partition_id != self.partition_id or (
            manifest.sample_id != scope.sample_id
        ):
            raise ValueError("xToken TQ producer returned a stale or foreign key")
        # The full expected row-key set is recorded before the first PUT so a
        # partial publish can still be cleaned up explicitly.
        scope.tile_ids.extend(manifest.tile_sample_ids)
        receipt = self.teacher.publish_logits_tq(
            manifest,
            max_tile_bytes=self.config.max_tile_bytes,
            timeout_s=self.config.timeout_s,
        )
        if receipt.sample_id != manifest.sample_id or receipt.tile_count != len(
            manifest.tiles
        ):
            raise ValueError("xToken TQ publish receipt does not match its manifest")
        result = self.student.materialize_full_logits_tq(
            manifest,
            max_tile_bytes=self.config.max_tile_bytes,
            timeout_s=self.config.timeout_s,
        )
        if result.consumer_node_id == manifest.producer_node_id:
            raise ValueError("xToken TQ did not cross a node boundary")
        if result.nbytes != manifest.nbytes:
            raise ValueError("xToken TQ receipt byte count differs from the producer")
        if result.tile_count != len(manifest.tiles):
            raise ValueError("xToken TQ receipt tile count differs from the manifest")
        self.metrics = {
            "tile_count": float(len(manifest.tiles)),
            "logical_payload_bytes": float(manifest.nbytes),
            "put_payload_bytes": float(receipt.put_bytes),
            "get_payload_bytes": float(result.nbytes),
            "put_seconds": receipt.put_seconds,
            "get_seconds": result.get_seconds,
            "student_buffer_bytes": float(result.buffer_bytes),
        }
        # Byte counters are application-level tensor bytes; wire/queue bytes
        # also depend on transport encoding and are not measured here.
        print(
            f"XTOKEN_TQ_TRANSFER producer={manifest.producer_node_id} "
            f"consumer={result.consumer_node_id} shape={manifest.shape} dtype=float32 "
            f"tiles={len(manifest.tiles)} logical_payload_bytes={manifest.nbytes} "
            f"put_payload_bytes={receipt.put_bytes} get_payload_bytes={result.nbytes} "
            f"put_seconds={receipt.put_seconds:.6f} get_seconds={result.get_seconds:.6f} "
            f"student_buffer_bytes={result.buffer_bytes}",
            flush=True,
        )
        return result.handles
