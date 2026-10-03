# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""CPU contracts for the bounded, tiled xToken transport (no simulated S1 claim)."""

from copy import deepcopy
from unittest.mock import MagicMock, patch

import pytest
import torch
from pydantic import ValidationError
from tensordict import TensorDict

from nemo_rl.data_plane.xtoken import (
    XTOKEN_FP32_ELEMENT_BYTES,
    XTOKEN_LOGITS_FIELD,
    XTokenTileSpec,
    XTokenTQManifest,
    XTokenTQPublishReceipt,
    XTokenTQReceiveResult,
    XTokenTQTransport,
    XTokenTransportConfig,
    check_payload_size,
    check_tile_size,
    fetch_logit_tile,
    plan_logit_tiles,
    publish_logit_tile,
    select_tq_nodes,
    validate_tile_coverage,
    validate_tq_support,
)


def test_default_and_positive_bounds():
    cfg = XTokenTransportConfig()
    assert cfg.backend == "ipc"
    assert cfg.max_payload_bytes == 64 * 1024 * 1024
    assert cfg.max_tile_bytes == 64 * 1024 * 1024
    assert cfg.timeout_s == 120
    for override in (
        {"backend": "other"},
        {"timeout_s": 0},
        {"max_payload_bytes": 0},
        {"max_tile_bytes": 0},
        {"max_tile_bytes": -1},
    ):
        with pytest.raises(ValidationError):
            XTokenTransportConfig(**override)


def support_args():
    policy = {
        "dtensor_cfg": {
            "enabled": True,
            "_v2": True,
            "tensor_parallel_size": 1,
            "context_parallel_size": 1,
        },
        "train_global_batch_size": 1,
        "train_micro_batch_size": 1,
        "sequence_packing": {"enabled": False},
        "dynamic_batching": {"enabled": False},
    }
    return {
        "data_plane": {"enabled": True, "impl": "transfer_queue", "backend": "simple"},
        "policies": [deepcopy(policy), deepcopy(policy)],
        "num_nodes": 2,
        "gpus_per_node": 1,
        "batch_size": 1,
    }


def test_supported_layout():
    validate_tq_support(**support_args())


@pytest.mark.parametrize(
    "key,value",
    [
        ("num_nodes", 1),
        ("num_nodes", 3),
        ("gpus_per_node", 2),
        ("batch_size", 2),
        ("data_plane", None),
        ("policies", []),
    ],
)
def test_unsupported_layout(key, value):
    args = support_args()
    args[key] = value
    with pytest.raises(ValueError):
        validate_tq_support(**args)


@pytest.mark.parametrize(
    "key,value", [("enabled", False), ("backend", "mooncake_cpu"), ("impl", "local")]
)
def test_unsupported_plane(key, value):
    args = support_args()
    args["data_plane"][key] = value
    with pytest.raises(ValueError):
        validate_tq_support(**args)


@pytest.mark.parametrize(
    "path,value",
    [
        (("dtensor_cfg", "enabled"), False),
        (("dtensor_cfg", "_v2"), False),
        (("dtensor_cfg", "tensor_parallel_size"), 2),
        (("dtensor_cfg", "context_parallel_size"), 2),
        (("sequence_packing", "enabled"), True),
        (("dynamic_batching", "enabled"), True),
        (("train_global_batch_size",), 2),
        (("train_micro_batch_size",), 2),
    ],
)
@pytest.mark.parametrize("index", [0, 1])
def test_unsupported_policy(index, path, value):
    args = support_args()
    target = args["policies"][index]
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError):
        validate_tq_support(**args)


def node(node_id, gpu=1, alive=True):
    return {
        "NodeID": node_id,
        "Alive": alive,
        "NodeManagerAddress": node_id,
        "Resources": {"GPU": gpu, f"node:{node_id}": 1},
    }


def test_deterministic_distinct_nodes():
    nodes = [node("z"), node("a"), node("dead", alive=False), node("cpu", gpu=0)]
    assert select_tq_nodes(nodes) == ({"node:a": 0.001}, {"node:z": 0.001})
    assert select_tq_nodes(list(reversed(nodes))) == select_tq_nodes(nodes)


def test_no_local_fallback():
    with pytest.raises(ValueError, match="two live"):
        select_tq_nodes([node("a"), node("dead", alive=False)])
    broken = node("b")
    broken["Resources"].pop("node:b")
    with pytest.raises(ValueError, match="advertise"):
        select_tq_nodes([node("a"), broken])


def test_padded_capacity_boundary_keeps_logical_semantics():
    # max_payload_bytes still bounds the whole logical payload.
    check_payload_size(seq_len=64, vocab_size=151936, max_bytes=64 * 1024 * 1024)
    check_payload_size(seq_len=4, vocab_size=8, max_bytes=128)
    with pytest.raises(ValueError, match="exceeds max_payload_bytes"):
        check_payload_size(seq_len=4, vocab_size=8, max_bytes=127)
    with pytest.raises(ValueError, match="positive"):
        check_payload_size(seq_len=0, vocab_size=8, max_bytes=128)


# ===============================================================================
# Tile planning: boundaries, remainders and single-row overflow
# ===============================================================================


def tiles_of(seq_len, vocab, tile_bytes):
    return plan_logit_tiles(
        seq_len=seq_len,
        vocab_size=vocab,
        max_tile_bytes=tile_bytes,
        partition_id="p",
        sample_id="s",
    )


def test_single_tile_when_bound_covers_payload():
    tiles = tiles_of(4, 8, 4 * 8 * XTOKEN_FP32_ELEMENT_BYTES)
    assert len(tiles) == 1
    assert tiles[0].tile_id == 0
    assert tiles[0].sample_id == "s#tile-00000"
    assert tiles[0].shape == (1, 4, 8)


def test_rows_split_with_remainder_and_exact_cover():
    # 5 rows, 2 rows per tile -> 3 tiles with a 1-row remainder.
    tiles = tiles_of(5, 4, 2 * 4 * XTOKEN_FP32_ELEMENT_BYTES)
    assert [(t.seq_start, t.seq_end) for t in tiles] == [(0, 2), (2, 4), (4, 5)]
    validate_tile_coverage(tiles, seq_len=5, vocab_size=4)


def test_row_overflow_splits_vocabulary():
    # One row of 10 floats; bound carries 4 floats -> column blocks 0-4, 4-8, 8-10.
    tiles = tiles_of(3, 10, 4 * XTOKEN_FP32_ELEMENT_BYTES)
    assert [(t.seq_start, t.seq_end, t.vocab_start, t.vocab_end) for t in tiles] == [
        (0, 1, 0, 4),
        (0, 1, 4, 8),
        (0, 1, 8, 10),
        (1, 2, 0, 4),
        (1, 2, 4, 8),
        (1, 2, 8, 10),
        (2, 3, 0, 4),
        (2, 3, 4, 8),
        (2, 3, 8, 10),
    ]
    validate_tile_coverage(tiles, seq_len=3, vocab_size=10)
    for tile in tiles:
        assert tile.nbytes <= 4 * XTOKEN_FP32_ELEMENT_BYTES


def test_one_element_bound_is_usable():
    tiles = tiles_of(2, 3, XTOKEN_FP32_ELEMENT_BYTES)
    assert len(tiles) == 6
    validate_tile_coverage(tiles, seq_len=2, vocab_size=3)
    with pytest.raises(ValueError, match="single FP32 element"):
        tiles_of(2, 3, XTOKEN_FP32_ELEMENT_BYTES - 1)
    with pytest.raises(ValueError, match="positive"):
        tiles_of(0, 3, 64)


def test_tile_ids_and_keys_are_unique_and_ordered():
    tiles = tiles_of(5, 4, 2 * 4 * XTOKEN_FP32_ELEMENT_BYTES)
    assert [t.tile_id for t in tiles] == list(range(len(tiles)))
    assert len({t.sample_id for t in tiles}) == len(tiles)


# ===============================================================================
# Manifest coverage validation
# ===============================================================================


def manifest(tiles, shape):
    return XTokenTQManifest("run", "step", shape, "teacher", tiles)


def test_manifest_accepts_exact_cover_and_exposes_ids():
    tiles = tiles_of(4, 8, 2 * 8 * XTOKEN_FP32_ELEMENT_BYTES)
    m = manifest(tiles, (1, 4, 8))
    assert m.tile_sample_ids == tuple(t.sample_id for t in tiles)
    assert m.nbytes == 4 * 8 * XTOKEN_FP32_ELEMENT_BYTES
    assert m.tiles is tiles or m.tiles == tiles


def test_manifest_rejects_gap_overlap_and_duplicate_keys():
    tiles = tiles_of(8, 8, 2 * 8 * XTOKEN_FP32_ELEMENT_BYTES)
    assert len(tiles) == 4
    # Drop a middle tile: a genuine gap between the surviving blocks.
    with pytest.raises(ValueError, match="gap or overlap"):
        manifest(tiles[:1] + tiles[2:], (1, 8, 8))
    # Drop the final tile: the cover ends early.
    with pytest.raises(ValueError, match="cover only"):
        manifest(tiles[:3], (1, 8, 8))
    # Duplicate the first tile: overlapping keys.
    with pytest.raises(ValueError, match="duplicate sample IDs"):
        manifest(tiles[:1] + tiles, (1, 8, 8))
    # Duplicate tile IDs under distinct keys.
    renumbered = (
        XTokenTileSpec(0, "a", 0, 2, 0, 8),
        XTokenTileSpec(0, "b", 2, 4, 0, 8),
    )
    with pytest.raises(ValueError, match="duplicate tile IDs"):
        manifest(renumbered, (1, 4, 8))
    with pytest.raises(ValueError, match="exactly one sample"):
        manifest(tiles, (2, 8, 8))


# ===============================================================================
# Per-tile PUT/GET validation
# ===============================================================================


def test_tile_put_rejects_misshaped_payload_before_writing():
    client = MagicMock()
    tile = tiles_of(4, 8, 2 * 8 * XTOKEN_FP32_ELEMENT_BYTES)[0]
    for payload in (
        torch.zeros(1, 2, 8, dtype=torch.float16),
        torch.zeros(1, 4, 8),
        torch.zeros(2, 8),
    ):
        with pytest.raises(ValueError, match="dense FP32|does not match"):
            publish_logit_tile(
                client, payload, tile=tile, partition_id="run", max_tile_bytes=128
            )
    client.put_samples.assert_not_called()
    with pytest.raises(ValueError, match="exceeding max_tile_bytes"):
        publish_logit_tile(
            client,
            torch.zeros(1, 2, 8),
            tile=tile,
            partition_id="run",
            max_tile_bytes=63,
        )
    client.put_samples.assert_not_called()


def test_tile_roundtrip_is_exact_and_uses_explicit_key():
    client = MagicMock()
    tile = XTokenTileSpec(0, "row-key", 2, 4, 0, 8)
    payload = torch.arange(16, dtype=torch.float32).reshape(1, 2, 8)
    publish_logit_tile(
        client, payload, tile=tile, partition_id="run", max_tile_bytes=64
    )
    assert client.put_samples.call_args.kwargs["sample_ids"] == ["row-key"]
    assert client.put_samples.call_args.kwargs["partition_id"] == "run"
    fields = client.put_samples.call_args.kwargs["fields"]
    torch.testing.assert_close(fields[XTOKEN_LOGITS_FIELD], payload, rtol=0, atol=0)
    client.get_samples.return_value = TensorDict(
        {XTOKEN_LOGITS_FIELD: payload}, batch_size=[1]
    )
    received = fetch_logit_tile(client, tile, partition_id="run", max_tile_bytes=64)
    torch.testing.assert_close(received, payload, rtol=0, atol=0)
    assert client.get_samples.call_args.kwargs == {
        "sample_ids": ["row-key"],
        "partition_id": "run",
        "select_fields": [XTOKEN_LOGITS_FIELD],
    }


@pytest.mark.parametrize(
    "payload",
    [
        torch.zeros(1, 1, 8),
        torch.zeros(1, 3, 8),
        torch.zeros(1, 2, 8, dtype=torch.float64),
    ],
)
def test_tile_get_rejects_shape_dtype_mismatch(payload):
    client = MagicMock()
    tile = XTokenTileSpec(0, "row-key", 2, 4, 0, 8)
    client.get_samples.return_value = TensorDict(
        {XTOKEN_LOGITS_FIELD: payload}, batch_size=[1]
    )
    with pytest.raises(ValueError, match="shape/dtype"):
        fetch_logit_tile(client, tile, partition_id="run", max_tile_bytes=64)


def test_tile_size_check_matches_spec_bytes():
    tile = XTokenTileSpec(0, "k", 0, 3, 0, 4)
    assert tile.nbytes == 3 * 4 * XTOKEN_FP32_ELEMENT_BYTES
    check_tile_size(tile=tile, max_tile_bytes=tile.nbytes)
    with pytest.raises(ValueError, match="exceeding"):
        check_tile_size(tile=tile, max_tile_bytes=tile.nbytes - 1)


# ===============================================================================
# Transport lifetime: IDs known before PUT, explicit cleanup, no retry
# ===============================================================================


def transport():
    result = XTokenTQTransport(
        config=XTokenTransportConfig(backend="tq"),
        data_plane=support_args()["data_plane"],
        teacher=MagicMock(),
        student=MagicMock(),
    )
    result.client = MagicMock()
    result.client.list_sample_ids.return_value = []
    result._stop_workers = MagicMock()
    return result


def make_manifest(t, seq_len=4, vocab=8, tile_bytes=2 * 8 * XTOKEN_FP32_ELEMENT_BYTES):
    return XTokenTQManifest(
        t.partition_id,
        t._steps[-1].sample_id,
        (1, seq_len, vocab),
        "teacher",
        plan_logit_tiles(
            seq_len=seq_len,
            vocab_size=vocab,
            max_tile_bytes=tile_bytes,
            partition_id=t.partition_id,
            sample_id=t._steps[-1].sample_id,
        ),
    )


def stub_transfer(t, m, train_error=None):
    t.teacher.prepare_logits_tq.return_value = m
    t.teacher.publish_logits_tq.return_value = XTokenTQPublishReceipt(
        sample_id=m.sample_id,
        tile_count=len(m.tiles),
        put_bytes=m.nbytes,
        put_seconds=0.1,
    )
    t.student.materialize_full_logits_tq.return_value = XTokenTQReceiveResult(
        [{"teacher_shards": [{"payload_ipc": "student-local"}]}],
        "student",
        m.nbytes,
        len(m.tiles),
        0.2,
        m.nbytes,
    )
    if train_error is not None:
        t.student.train = MagicMock(side_effect=train_error)


def test_manifest_ids_recorded_before_first_put():
    t = transport()
    with t.step():
        m = make_manifest(t)
        t.teacher.prepare_logits_tq.return_value = m
        assert t._steps[-1].tile_ids == []
        # Publish must run after the full expected ID set is recorded, so a
        # partial publish can always be cleaned up.
        t.teacher.publish_logits_tq = MagicMock(
            side_effect=lambda *_, **kw: (
                pytest.fail("publish ran before IDs were recorded")
                if not t._steps[-1].tile_ids
                else XTokenTQPublishReceipt(
                    sample_id=m.sample_id,
                    tile_count=len(m.tiles),
                    put_bytes=m.nbytes,
                    put_seconds=0.1,
                )
            )
        )
        t.student.materialize_full_logits_tq.return_value = XTokenTQReceiveResult(
            [{"teacher_shards": [{"payload_ipc": "student-local"}]}],
            "student",
            m.nbytes,
            len(m.tiles),
            0.2,
            m.nbytes,
        )
        result = t.transfer(MagicMock())
        assert result[0]["teacher_shards"][0]["payload_ipc"] == "student-local"
        assert t._steps[-1].tile_ids == list(m.tile_sample_ids)
    assert t._steps == []
    assert [c.args for c in t.client.clear_samples.call_args_list] == [
        (list(m.tile_sample_ids), t.partition_id)
    ]


def test_partial_publish_failure_clears_all_expected_ids():
    t = transport()
    events = []
    t._stop_workers.side_effect = lambda: events.append("stop")
    t.client.clear_samples.side_effect = lambda *_: events.append("clear")
    t.client.list_sample_ids.return_value = []
    with pytest.raises(RuntimeError, match="tile 1"):
        with t.step():
            m = make_manifest(t, seq_len=8)
            t.teacher.prepare_logits_tq.return_value = m

            def fail_second_put(*_, **kw):
                # The first tile was written before the failure.
                t.client.put_samples(
                    sample_ids=["written"], partition_id=t.partition_id
                )
                raise RuntimeError("tile 1 PUT failed")

            t.teacher.publish_logits_tq.side_effect = fail_second_put
            t.transfer(MagicMock())
    # Cleanup used the manifest's full expected ID set, not a partition wipe.
    assert events == ["stop", "clear"]
    cleared_ids = t.client.clear_samples.call_args.args[0]
    assert cleared_ids == list(m.tile_sample_ids)
    assert t._steps == []


def test_put_get_and_train_failures_stop_and_clear_without_retry():
    for stage in ("put", "get", "train"):
        t = transport()
        events = []
        t._stop_workers.side_effect = lambda: events.append("stop")
        t.client.clear_samples.side_effect = lambda *_: events.append("clear")
        train = MagicMock(side_effect=TimeoutError("train"))
        with pytest.raises(TimeoutError, match=stage):
            with t.step():
                m = make_manifest(t)
                t.teacher.prepare_logits_tq.return_value = m
                if stage == "put":
                    t.teacher.publish_logits_tq.side_effect = TimeoutError("put")
                else:
                    t.teacher.publish_logits_tq.return_value = XTokenTQPublishReceipt(
                        sample_id=m.sample_id,
                        tile_count=len(m.tiles),
                        put_bytes=m.nbytes,
                        put_seconds=0.1,
                    )
                t.student.materialize_full_logits_tq.side_effect = (
                    TimeoutError("get")
                    if stage == "get"
                    else MagicMock(
                        return_value=XTokenTQReceiveResult(
                            [], "student", m.nbytes, len(m.tiles), 0.2, m.nbytes
                        )
                    )
                )
                t.transfer(MagicMock())
                if stage == "train":
                    train()
        assert events == ["stop", "clear"], stage
        assert t._steps == []
        assert t.teacher.prepare_logits_tq.call_count == 1
        assert train.call_count == (1 if stage == "train" else 0)


def test_cleanup_preserves_both_errors():
    t = transport()
    t.client.clear_samples.side_effect = RuntimeError("cleanup")
    with pytest.raises(ExceptionGroup) as caught:
        with t.step():
            # Tile IDs are in scope, so cleanup actually clears rows.
            stub_transfer(t, make_manifest(t))
            t.transfer(MagicMock())
            raise ValueError("training")
    assert [str(e) for e in caught.value.exceptions] == ["training", "cleanup"]


@pytest.mark.parametrize("terminated", [True, False])
def test_cleanup_waits_for_actor_death(terminated):
    import ray

    t = transport()
    del t._stop_workers  # Exercise the production termination barrier.
    actors = [MagicMock(), MagicMock()]
    t.teacher.worker_group.workers = actors[:1]
    t.student.worker_group.workers = actors[1:]
    error = ray.exceptions.ActorDiedError() if terminated else TimeoutError("stop")
    with patch.object(ray, "kill") as kill, patch.object(ray, "get", side_effect=error):
        with pytest.raises(ValueError if terminated else ExceptionGroup):
            with t.step():
                # Tile IDs in scope so the explicit clear runs after the barrier.
                stub_transfer(t, make_manifest(t))
                t.transfer(MagicMock())
                raise ValueError("training")
    assert kill.call_count == 2
    if terminated:
        t.client.clear_samples.assert_called_once()
    else:
        t.client.clear_samples.assert_not_called()


def test_cleanup_verifies_no_residual_rows_and_exposes_late_puts():
    # A producer killed mid-PUT can land rows after the clear (storage RPCs in
    # flight are not cancelled by actor death). The driver re-lists the
    # partition and clears again, and fails loudly if rows keep coming back.
    import nemo_rl.data_plane.xtoken as xtoken_mod

    # Success path still verifies emptiness.
    t = transport()
    with t.step():
        stub_transfer(t, make_manifest(t))
        t.transfer(MagicMock())
    assert t.client.clear_samples.call_count == 1

    # Late PUT after the first clear: second clear removes it, no error.
    t2 = transport()
    with t2.step():
        m2 = make_manifest(t2)
        stub_transfer(t2, m2)
        listings = iter([[m2.tile_sample_ids[0]], []])
        t2.client.list_sample_ids.side_effect = lambda *_: next(listings)
        t2.transfer(MagicMock())
    assert t2.client.clear_samples.call_count == 2

    # A producer that keeps re-adding rows fails loudly instead of pretending:
    # every listing still shows the expected tile rows.
    t3 = transport()
    t3.client.list_sample_ids.side_effect = lambda *_: (
        list(t3._steps[-1].tile_ids) if t3._steps else []
    )
    with patch.object(xtoken_mod.time, "sleep") as slept:
        with pytest.raises(RuntimeError, match="could not clear"):
            with t3.step():
                stub_transfer(t3, make_manifest(t3))
                t3.transfer(MagicMock())
    assert slept.call_count == xtoken_mod.XTOKEN_CLEANUP_VERIFY_ATTEMPTS - 1


def test_transfer_is_metadata_only_and_checks_key_receipt_and_bytes():
    t = transport()
    with t.step():
        m = make_manifest(t)
        stub_transfer(t, m)
        assert (
            t.transfer(MagicMock())[0]["teacher_shards"][0]["payload_ipc"]
            == "student-local"
        )
        assert t.metrics["tile_count"] == len(m.tiles)
        assert t.metrics["logical_payload_bytes"] == m.nbytes
        assert (
            t.metrics["put_payload_bytes"] == t.metrics["get_payload_bytes"] == m.nbytes
        )
        assert t.metrics["student_buffer_bytes"] == m.nbytes
    # Foreign manifest rejected before any publish or receive.
    t = transport()
    with t.step():
        foreign = make_manifest(t)
        object.__setattr__(foreign, "sample_id", "other-step")
        t.teacher.prepare_logits_tq.return_value = foreign
        with pytest.raises(ValueError, match="stale or foreign"):
            t.transfer(MagicMock())
    t.teacher.publish_logits_tq.assert_not_called()
    t.student.materialize_full_logits_tq.assert_not_called()
    # Receipt mismatch rejected before the student receives.
    t = transport()
    with t.step():
        m = make_manifest(t)
        t.teacher.prepare_logits_tq.return_value = m
        t.teacher.publish_logits_tq.return_value = XTokenTQPublishReceipt(
            sample_id=m.sample_id,
            tile_count=len(m.tiles) + 1,
            put_bytes=m.nbytes,
            put_seconds=0.1,
        )
        with pytest.raises(ValueError, match="receipt"):
            t.transfer(MagicMock())
    t.student.materialize_full_logits_tq.assert_not_called()
    # Byte-count mismatch between producer and receiver.
    t = transport()
    with t.step():
        m = make_manifest(t)
        t.teacher.prepare_logits_tq.return_value = m
        stub_transfer(t, m)
        t.student.materialize_full_logits_tq.return_value = XTokenTQReceiveResult(
            [], "student", m.nbytes + 4, len(m.tiles), 0.2, m.nbytes
        )
        with pytest.raises(ValueError, match="byte count"):
            t.transfer(MagicMock())


def test_transfer_rejects_same_node_consumer():
    t = transport()
    with t.step():
        m = make_manifest(t)
        t.teacher.prepare_logits_tq.return_value = m
        t.teacher.publish_logits_tq.return_value = XTokenTQPublishReceipt(
            sample_id=m.sample_id,
            tile_count=len(m.tiles),
            put_bytes=m.nbytes,
            put_seconds=0.1,
        )
        t.student.materialize_full_logits_tq.return_value = XTokenTQReceiveResult(
            [], "teacher", m.nbytes, len(m.tiles), 0.2, m.nbytes
        )
        with pytest.raises(ValueError, match="node boundary"):
            t.transfer(MagicMock())
    t.student.release_ipc_buffer.assert_not_called()
