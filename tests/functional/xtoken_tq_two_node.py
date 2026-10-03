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
"""Real two-node S1/S2 with bounded tiles and failure injection; never creates a local cluster."""

from __future__ import annotations

import argparse
import re
import tempfile
import uuid
from datetime import timedelta
from pathlib import Path

# Payload 256 x 128 FP32 = 128 KiB logical; a 8 KiB tile bound forces 16+ tiles.
PAYLOAD_SEQ = 256
PAYLOAD_VOCAB = 128
MAX_PAYLOAD_BYTES = 64 * 1024 * 1024
MAX_TILE_BYTES = 8 * 1024
EXPECTED_TILES = None  # computed from the plan below


def check_log(path: Path, steps: int) -> None:
    """Require tiled transfers, bounded buffers and explicit empty-row evidence."""
    text = path.read_text(encoding="utf-8")
    transfers = re.findall(
        r"XTOKEN_TQ_TRANSFER producer=(\S+) consumer=(\S+) .*?tiles=(\d+) "
        r"logical_payload_bytes=(\d+) put_payload_bytes=(\d+) get_payload_bytes=(\d+) "
        r".*?student_buffer_bytes=(\d+)",
        text,
    )
    assert len(transfers) == steps, (len(transfers), steps)
    for (
        producer,
        consumer,
        tiles,
        logical,
        put_bytes,
        get_bytes,
        buffer_bytes,
    ) in transfers:
        assert producer != consumer
        assert int(tiles) > 0
        assert int(logical) == int(put_bytes) == int(get_bytes) > 0
        assert int(buffer_bytes) <= MAX_PAYLOAD_BYTES
    assert len({row[-1] for row in transfers}) == 1, "Receive buffer grew"
    cleared = re.findall(r"XTOKEN_TQ_CLEARED .*?tiles=(\d+) remaining_rows=(\d+)", text)
    assert len(cleared) == steps and set(cleared) == {cleared[0]}, cleared
    assert {row[1] for row in cleared} == {"0"}, cleared
    losses = re.findall(r"Loss:\s+([-+\d.eE]+)", text)
    assert len(losses) == steps
    assert all(float("-inf") < float(value) < float("inf") for value in losses)
    print(
        f"PASS: {steps} cross-node training steps, finite loss, bounded buffer, "
        f"{cleared[0][0]} tiles cleared per step, no rows"
    )


def run_two_node() -> None:
    """Run tiled PUT/GET, frozen-payload comparison and failure injection on two GPU nodes."""
    global EXPECTED_TILES
    import ray
    import torch
    from omegaconf import OmegaConf

    from nemo_rl.algorithms.loss.loss_functions import CrossTokenizerDistillationLossFn
    from nemo_rl.algorithms.x_token.loss_utils import (
        LocalizedAlignment,
        rebuild_teacher_full_logits_from_ipc,
    )
    from nemo_rl.data_plane.factory import build_data_plane_client
    from nemo_rl.data_plane.worker_mixin import TQWorkerMixin
    from nemo_rl.data_plane.xtoken import (
        XTOKEN_LOGITS_FIELD,
        XTokenTQManifest,
        plan_logit_tiles,
        publish_logit_tile,
        select_tq_nodes,
    )
    from nemo_rl.distributed.batched_data_dict import BatchedDataDict
    from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES
    from nemo_rl.models.policy.utils import get_runtime_env_for_policy_worker
    from nemo_rl.models.policy.workers.dtensor_policy_worker_v2 import (
        DTensorPolicyWorkerV2Impl,
    )
    from nemo_rl.utils.config import load_config, register_omegaconf_resolvers

    register_omegaconf_resolvers()
    # address="auto" fails without an existing cluster; no single-host fallback.
    ray.init(address="auto")
    teacher_resource, student_resource = select_tq_nodes(ray.nodes())
    cfg = {
        "enabled": True,
        "impl": "transfer_queue",
        "backend": "simple",
        "claim_meta_poll_interval_s": 0.5,
        "simple": {"storage_capacity": 256, "num_storage_units": 4},
    }
    client = build_data_plane_client(cfg, bootstrap=True)
    partition = f"xtoken-acceptance-{uuid.uuid4().hex}"
    client.register_partition(partition, [XTOKEN_LOGITS_FIELD], 1, [])
    loss_cfg = OmegaConf.to_container(
        load_config("examples/configs/xtoken_off_policy_distillation.yaml").loss_fn,
        resolve=True,
    )

    sample_id_template = f"{partition}/{{uid}}"

    def plan_tiles(sample_id, seq_len=PAYLOAD_SEQ, vocab_size=PAYLOAD_VOCAB):
        return plan_logit_tiles(
            seq_len=seq_len,
            vocab_size=vocab_size,
            max_tile_bytes=MAX_TILE_BYTES,
            partition_id=partition,
            sample_id=sample_id,
        )

    EXPECTED_TILES = len(plan_tiles(sample_id_template.format(uid="probe")))
    assert EXPECTED_TILES > 1, "tile bound must split the acceptance payload"

    @ray.remote(
        num_gpus=1,
        runtime_env={
            **get_runtime_env_for_policy_worker("dtensor_policy_worker_v2"),
            "py_executable": PY_EXECUTABLES.AUTOMODEL,
            "env_vars": {"PYTHONPATH": str(Path.cwd())},
        },
    )
    class Probe(TQWorkerMixin):
        def __init__(self, config, loss_config):
            self.setup_data_plane(config)
            torch.cuda.set_device(0)
            self._teacher_ipc_storage = None
            self._teacher_ipc_handle = None
            self.tmp = tempfile.TemporaryDirectory(prefix="xtoken-tq-")
            torch.distributed.init_process_group(
                "nccl",
                rank=0,
                world_size=1,
                init_method=f"file://{self.tmp.name}/rendezvous",
                timeout=timedelta(seconds=120),
            )
            projection = str(Path(self.tmp.name) / "projection.pt")
            torch.save(
                {
                    "indices": torch.stack(
                        (torch.arange(16), torch.arange(16) + 16), dim=1
                    ),
                    "likelihoods": torch.tensor([[0.7, 0.3]]).repeat(16, 1),
                },
                projection,
            )
            loss_config.update(
                student_vocab_size=16,
                teacher_vocab_sizes=[PAYLOAD_VOCAB],
                projection_matrix_paths=[projection],
                teacher_weights=[1.0],
                teacher_gold_loss=[False],
                teacher_xtoken_loss=[False],
                vocab_topk=PAYLOAD_VOCAB,
            )
            self.loss_fn = CrossTokenizerDistillationLossFn(loss_config)

        def payload(self):
            return (
                (
                    torch.arange(
                        PAYLOAD_SEQ * PAYLOAD_VOCAB, dtype=torch.float32
                    ).reshape(1, PAYLOAD_SEQ, PAYLOAD_VOCAB)
                    * 37
                )
                % 509
            ) / 32

        def build_manifest(self, sample_id):
            tiles = plan_logit_tiles(
                seq_len=PAYLOAD_SEQ,
                vocab_size=PAYLOAD_VOCAB,
                max_tile_bytes=MAX_TILE_BYTES,
                partition_id=partition,
                sample_id=sample_id,
            )
            return XTokenTQManifest(
                partition_id=partition,
                sample_id=sample_id,
                shape=(1, PAYLOAD_SEQ, PAYLOAD_VOCAB),
                producer_node_id=ray.get_runtime_context().get_node_id(),
                tiles=tiles,
            )

        def publish(
            self,
            manifest,
            *,
            tile_limit=None,
            fail_after=None,
        ):
            """PUT manifest tiles one at a time, straight from CPU staging.

            ``tile_limit`` publishes only the first N tiles (partial publish),
            ``fail_after`` raises after N tiles were written.
            """
            payload = self.payload()
            written = 0
            for index, tile in enumerate(manifest.tiles):
                if tile_limit is not None and index >= tile_limit:
                    break
                if fail_after is not None and index >= fail_after:
                    raise RuntimeError(f"tile {index} PUT failed (injected)")
                tile_payload = torch.empty(tile.shape, dtype=torch.float32)
                tile_payload.copy_(
                    payload[
                        0,
                        tile.seq_start : tile.seq_end,
                        tile.vocab_start : tile.vocab_end,
                    ]
                )
                publish_logit_tile(
                    self._require_dp_client(),
                    tile_payload,
                    tile=tile,
                    partition_id=partition,
                    max_tile_bytes=MAX_TILE_BYTES,
                )
                written += 1
            return written

        def receive_and_compare(self, manifest):
            """Production student materialization, then loss/grad/optimizer parity."""
            receipt = DTensorPolicyWorkerV2Impl.materialize_full_logits_tq(
                self, manifest, max_tile_bytes=MAX_TILE_BYTES
            )
            received = rebuild_teacher_full_logits_from_ipc(
                receipt.handles, cp_group=None, device=0
            )
            expected = self.payload().cuda()
            torch.testing.assert_close(received, expected, rtol=0, atol=0)
            torch.manual_seed(42)
            initial = torch.randn(1, PAYLOAD_SEQ, 16, device="cuda")
            ids = (torch.arange(PAYLOAD_SEQ, device="cuda") % 16).unsqueeze(0)
            mask = torch.ones(1, PAYLOAD_SEQ, device="cuda")
            sample_mask = torch.ones(1, device="cuda")
            data = BatchedDataDict(
                input_ids=ids, token_mask=mask, sample_mask=sample_mask
            )
            align = LocalizedAlignment(
                sample_mask=sample_mask,
                student_chunk_id=torch.arange(PAYLOAD_SEQ, device="cuda").unsqueeze(0),
                teacher_chunk_id=torch.arange(PAYLOAD_SEQ, device="cuda").unsqueeze(0),
                pair_valid=mask.bool(),
                pair_is_correct=mask.bool(),
                student_input_ids=ids,
                student_token_mask=mask,
            )

            def update(teacher_logits):
                student = torch.nn.Parameter(initial.clone())
                optimizer = torch.optim.AdamW([student], lr=0.01)
                loss, _ = self.loss_fn(
                    data,
                    torch.ones((), device="cuda"),
                    mask.sum(),
                    student,
                    student,
                    {0: teacher_logits},
                    {0: align},
                )
                loss.backward()
                gradient = student.grad.detach().clone()
                optimizer.step()
                return loss.detach(), gradient, student.detach().clone()

            # Calibrate each tolerance using two repeats on the reference path.
            baseline = update(expected)
            repeat = update(expected)
            transported = update(received)
            errors = []
            tolerances = []
            for direct, again, tq in zip(baseline, repeat, transported):
                tolerance = max(
                    float((direct - again).abs().max()) * 4,
                    torch.finfo(torch.float32).eps,
                )
                torch.testing.assert_close(tq, direct, rtol=0, atol=tolerance)
                errors.append(float((tq - direct).abs().max()))
                tolerances.append(tolerance)
            assert baseline[1].abs().sum() > 0
            assert not torch.equal(baseline[2], initial)
            return {
                "consumer": receipt.consumer_node_id,
                "bytes": receipt.nbytes,
                "tile_count": receipt.tile_count,
                "buffer_bytes": receipt.buffer_bytes,
                "buffer_ptr": self._teacher_ipc_storage.data_ptr(),
                "errors": errors,
                "tolerances": tolerances,
            }

        def receive_expect_failure(self, manifest):
            """Materialize a manifest whose tiles are absent; must raise."""
            try:
                DTensorPolicyWorkerV2Impl.materialize_full_logits_tq(
                    self, manifest, max_tile_bytes=MAX_TILE_BYTES
                )
            except Exception as error:
                return f"{type(error).__name__}: {error}"
            raise AssertionError("materialize accepted an unpublished payload")

        def receive_then_fail(self, manifest):
            """Receive fully, then fail like a student training exception."""
            self.receive_and_compare(manifest)
            raise RuntimeError("student training failed (injected)")

    actors = []
    tracked_ids = []

    def require_empty_partition(context):
        remaining = client.list_sample_ids(partition)
        assert remaining == [], (context, remaining)

    try:
        producer = Probe.options(resources=teacher_resource).remote(cfg, loss_cfg)
        consumer = Probe.options(resources=student_resource).remote(cfg, loss_cfg)
        actors.extend([producer, consumer])

        # ── S1/S2: multi-tile publish, exact rebuild, frozen-payload parity ──
        reports = []
        for _ in range(10):
            sample_id = sample_id_template.format(uid=uuid.uuid4().hex)
            tracked_ids.append(sample_id)
            manifest = ray.get(producer.build_manifest.remote(sample_id), timeout=120)
            # All tile IDs are known to the driver before the first PUT.
            assert len(manifest.tiles) == EXPECTED_TILES
            written = ray.get(producer.publish.remote(manifest), timeout=120)
            assert written == EXPECTED_TILES
            report = ray.get(consumer.receive_and_compare.remote(manifest), timeout=120)
            assert report["consumer"] != manifest.producer_node_id
            assert report["bytes"] == manifest.nbytes > 0
            assert report["tile_count"] == EXPECTED_TILES
            reports.append(report)
            # Driver-side cleanup uses the pre-registered explicit tile IDs.
            client.clear_samples(list(manifest.tile_sample_ids), partition)
            require_empty_partition(sample_id)
        assert len({r["buffer_ptr"] for r in reports}) == 1
        assert len({r["buffer_bytes"] for r in reports}) == 1
        print(
            f"PASS S1/S2: teacher={manifest.producer_node_id}, "
            f"student={report['consumer']}, tiles={EXPECTED_TILES}, "
            f"payload_bytes={manifest.nbytes}, numerical={reports[-1]}"
        )

        # ── S3a: missing tiles — GET fails, no partial receive escapes ──
        sample_id = sample_id_template.format(uid=uuid.uuid4().hex)
        manifest = ray.get(producer.build_manifest.remote(sample_id), timeout=120)
        tracked_ids.append(sample_id)
        written = ray.get(producer.publish.remote(manifest, tile_limit=2), timeout=120)
        assert written == 2
        failure = ray.get(consumer.receive_expect_failure.remote(manifest), timeout=120)
        print(f"PASS missing-tile: {failure}")
        client.clear_samples(list(manifest.tile_sample_ids), partition)
        require_empty_partition("missing-tile")

        # ── S3b: partial publish failure — explicit IDs clear a mid-PUT abort ──
        sample_id = sample_id_template.format(uid=uuid.uuid4().hex)
        manifest = ray.get(producer.build_manifest.remote(sample_id), timeout=120)
        tracked_ids.append(sample_id)
        try:
            ray.get(
                producer.publish.remote(manifest, fail_after=EXPECTED_TILES // 2),
                timeout=120,
            )
            raise AssertionError("injected publish failure did not surface")
        except ray.exceptions.RayTaskError:
            pass
        # The manifest existed before the first PUT, so cleanup targets every
        # expected ID even though only some were written.
        client.clear_samples(list(manifest.tile_sample_ids), partition)
        require_empty_partition("partial-publish")

        # ── S3c: GET timeout — surfaced, then cleaned up ──
        sample_id = sample_id_template.format(uid=uuid.uuid4().hex)
        manifest = ray.get(producer.build_manifest.remote(sample_id), timeout=120)
        tracked_ids.append(sample_id)
        try:
            ray.get(consumer.receive_and_compare.remote(manifest), timeout=0.01)
            raise AssertionError("GET timeout was not enforced")
        except ray.exceptions.GetTimeoutError:
            pass
        ray.get(producer.publish.remote(manifest), timeout=120)
        ray.get(consumer.receive_and_compare.remote(manifest), timeout=120)
        client.clear_samples(list(manifest.tile_sample_ids), partition)
        require_empty_partition("timeout")

        # ── S3d: student exception after receive — cleanup still completes ──
        sample_id = sample_id_template.format(uid=uuid.uuid4().hex)
        manifest = ray.get(producer.build_manifest.remote(sample_id), timeout=120)
        tracked_ids.append(sample_id)
        try:
            ray.get(consumer.receive_then_fail.remote(manifest), timeout=120)
            raise AssertionError("injected student failure did not surface")
        except ray.exceptions.RayTaskError:
            pass
        client.clear_samples(list(manifest.tile_sample_ids), partition)
        require_empty_partition("student-exception")

        # ── S3e: buffer identity and staging do not grow across steps ──
        assert len({r["buffer_ptr"] for r in reports}) == 1
        assert len({r["buffer_bytes"] for r in reports}) == 1
        print(
            f"PASS S3: failure injection and explicit cleanup; "
            f"buffer_ptr stable={len({r['buffer_ptr'] for r in reports}) == 1}"
        )
    finally:
        for actor in actors:
            ray.kill(actor, no_restart=True)
        for actor in actors:
            try:
                ray.get(actor.payload.remote(), timeout=120)
            except ray.exceptions.ActorDiedError:
                pass
        client.clear_samples(tracked_ids, partition)
        leftover = client.list_sample_ids(partition)
        assert leftover == [], leftover
        client.close()
        ray.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-log", type=Path)
    parser.add_argument("--expected-steps", type=int, default=3)
    args = parser.parse_args()
    if args.check_log is not None:
        check_log(args.check_log, args.expected_steps)
    else:
        run_two_node()
