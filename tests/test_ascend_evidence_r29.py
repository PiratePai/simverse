"""Offline numerical/accounting review; archived NPU collectors are never run."""
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/ascend-910b3-r29"
spec = importlib.util.spec_from_file_location("r29_review", ROOT / "tools/review_ascend_r29.py")
AUDIT = importlib.util.module_from_spec(spec)
spec.loader.exec_module(AUDIT)


def read(name):
    return json.loads((EVIDENCE / name).read_text(encoding="utf-8"))


def test_archive_bytes_and_declared_hash_mismatch_are_frozen():
    review = read("review.json")
    assert review == AUDIT.review()
    assert review["original_manifest_sha256"] == "90b0f50b77e9d1744602571ae5f70441041bce71a9ec7cad2534402352a6c3d0"
    inventory = review["archive_inventory"]
    assert len(inventory) == 23
    # Digest frozen from PR #23, not regenerated to accommodate edited captures.
    assert hashlib.sha256(json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode()).hexdigest() == "46716541f8c6e91d479b1dbb7b615a28eb707c0df3ba48bf637d0644cd0a86a4"
    assert [entry["path"] for entry in review["manifest_mismatches"]] == [
        "docs/evidence/ascend-910b3-r29/raw/forward_execute.log"]
    mismatch = review["manifest_mismatches"][0]
    assert mismatch["declared"]["bytes"] == 16614
    assert mismatch["actual"]["bytes"] == 16603
    files = {p.relative_to(ROOT).as_posix() for p in EVIDENCE.rglob("*") if p.is_file()}
    assert files == {key for key in inventory if key.startswith(EVIDENCE.relative_to(ROOT).as_posix())} | {
        "docs/evidence/ascend-910b3-r29/raw/manifest.json",
        "docs/evidence/ascend-910b3-r29/review.json"}


def test_collective_allgather_correction_and_payload_domains():
    points = read("review.json")["collectives"]["points"]
    assert len(points) == 100
    for rank in (0, 1):
        data = read(f"raw/collectives_rank{rank}.json")
        assert data["correctness_summary"] == {"checked": 50, "exact": 50, "mismatch": 0, "errors": 0}
        assert len(data["results"]) == 50
        for row in data["results"]:
            point = next(p for p in points if p["rank"] == rank and p["op"] == row["op"]
                         and p["buffer_scale_bytes"] == row["numel_per_rank"] * 2)
            n = row["numel_per_rank"] * 2
            # Independent two-rank byte/time anchor: Gather and Scatter use total buffer.
            expected = n / row["median_ms"] / 1e6
            if row["op"] in ("all_gather", "reduce_scatter"):
                expected *= 2
            assert point["derived_algbw_GBps"] == pytest.approx(expected)
            assert point["derived_busbw_GBps"] == pytest.approx(
                expected if row["op"] in ("all_reduce", "broadcast") else expected / 2)
            assert point["original_busbw_GBps"] == row["busbw_GBps"]
    large = next(p for p in points if p["rank"] == 0 and p["op"] == "all_gather"
                 and p["buffer_scale_bytes"] == 2**28)
    assert large["logical_input_bytes"] == 268435456
    assert large["logical_output_bytes"] == 536870912
    assert large["derived_busbw_GBps"] == pytest.approx(17.725, abs=.001)
    assert large["original_busbw_GBps"] == 8.863


def test_graph_deltas_and_acl_observations_do_not_invent_pool_or_granularity():
    review = read("review.json")
    assert review["graph"]["capture_minus_baseline_free_bytes"] == 56623104
    assert review["graph"]["capture_minus_baseline_allocated_bytes"] == 16779264
    assert review["graph"]["replay1_minus_capture_free_bytes"] == 4194304  # old report: 4210304
    assert review["graph"]["pure_graph_pool_bytes"] is None
    assert review["acl"]["incremental_observations"] == [
        {"logical_dev": 0, "observed_free_delta_minus_requested_bytes": [2097152, 2097152, 2207744]},
        {"logical_dev": 1, "observed_free_delta_minus_requested_bytes": [2097152] * 3}]
    assert review["acl"]["allocation_granularity_bytes"] is None
    assert review["acl"]["deployment_budget_bytes"] is None


def test_flat_mrope_shared_oracle_can_miss_three_heads():
    # Independent scalar counterexample: each head rotates lanes (0,32).
    # At pos=1, lane 0=1 and lane 32=0 => lane 0=cos(1) in ALL four heads.
    heads, head_dim = 4, 256
    flat = [0.] * (heads * head_dim)
    for head in range(heads):
        flat[head * head_dim] = 1.
    archived = flat.copy()
    archived[0] = math.cos(1.)  # archive's [...,:64] touches only head 0
    expected = flat.copy()
    for head in range(heads):
        expected[head * head_dim] = math.cos(1.)
    assert sum(archived[h * head_dim] != expected[h * head_dim] for h in range(heads)) == 3
    assert read("review.json")["mrope"]["numeric_admitted"] is False


def test_slice_evidence_preserves_host_expert_and_toy_scope():
    review = read("review.json")["slices"]
    assert review["entries_per_rank_sent"] == [32, 32]  # report confused per-destination 16 with total
    assert review["max_allocator_peak_bytes"] == 3624960
    for rank in (0, 1):
        data = read(f"raw/tp2_slice_rank{rank}.json")
        assert data["summary"] == {"cases": 7, "passed": 7, "failed": 0, "failed_cases": []}
        assert data["moe_ep2"]["config"] == {"tokens": 32, "experts": 8, "routed_experts": 6,
                                              "top_k": 2, "hidden": 64, "expert_hidden": 32}
        assert sum(data["moe_ep2"]["expert_counts"]) == 64
    assert review["full_model_npu_ep_verified"] is False


def test_successful_forward_is_tp1_with_small_weight_log_and_unpinned_runtime():
    review = read("review.json")
    forward = read("raw/forward_execute.json")
    log = (EVIDENCE / "raw/forward_execute.log").read_text(encoding="utf-8")
    assert "Model loading took 0.32 GiB memory" in log
    assert "0.32 GiB for weights" in log and "Available KV cache memory: 11.81 GiB" in log
    assert "visible_npus=[5]" in log and "tensor_parallel_size=1" in log
    assert "ascend_warmup_patch.py" in log  # local patch provenance still missing
    assert "out of memory" not in log.lower()
    assert forward["execution"]["hardware"]["stack_pair_validated"] is False
    assert review["forward"]["tensor_parallel_size"] == 1
    assert review["forward"]["output_token_ids"] == [20] * 4
    assert review["forward"]["failed_attempts_archived"] is False
    assert review["script_identity"] == {"missing_from_env_snapshot": ["r29_tp2.py"],
                                          "hash_mismatches": ["r29_manifest.py"]}


def test_r29_cannot_grant_production_admission():
    from simverse.cost.profile_store import PROFILE_SETS, load_profile_set
    review = read("review.json")
    assert review["calibration_admission"] == "not-admitted"
    assert review["deployment_ranking_eligible"] is False
    assert review["does_not_unblock"] == ["G-D7", "R02b", "R17", "R18"]
    assert "ascend-910b3-r29" not in PROFILE_SETS
    with pytest.raises(ValueError, match="未知 profile"):
        load_profile_set("ascend-910b3-r29")
