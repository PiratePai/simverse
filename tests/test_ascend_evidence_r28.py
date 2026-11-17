"""Offline audit of the R28 archive; never executes the archived NPU collectors."""
import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/ascend-910b3"


def read(name):
    return json.loads((EVIDENCE / name).read_text(encoding="utf-8"))


def test_archive_integrity_and_original_measurements_unchanged():
    manifest, review = read("manifest.json"), read("review.json")
    inventory = review["raw_files_unchanged"]
    # Independent anchor from the original PR, not refreshed from edited artifacts.
    assert hashlib.sha256(json.dumps(
        inventory, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest() == "44e7157dc47d56b9da1c6490bd8c337af5cd4c706c19c39763dd4372c3a12027"
    assert len(inventory) == 28
    assert set(manifest["files"]) == set(inventory) | {"README.md", "review.json"}
    assert {p.relative_to(EVIDENCE).as_posix() for p in EVIDENCE.rglob("*")
            if p.is_file()} == set(manifest["files"]) | {"manifest.json"}
    for name, entry in manifest["files"].items():
        content = (EVIDENCE / name).read_bytes()
        assert len(content) == entry["bytes"], name
        assert hashlib.sha256(content).hexdigest() == entry["sha256"], name
        if name in inventory:
            assert entry == inventory[name], name


def test_allreduce_corrections_use_payload_time_and_both_ranks():
    points = read("review.json")["allreduce"]["points"]
    assert len(points) == 20
    keys = {(r["rank"], r["bytes_per_rank"]) for r in points}
    assert len(keys) == 20
    for rank in (0, 1):
        data = read(f"p1b_hccl_batch_rank{rank}.json")
        assert data["world_size"] == 2
        for row in data["results"]:
            if row["op"] != "all_reduce":
                continue
            corrected = next(p for p in points if
                             (p["rank"], p["bytes_per_rank"]) == (rank, row["bytes_per_rank"]))
            # Two-rank allreduce normalized busbw equals algbw (not half).
            expected = row["bytes_per_rank"] / (row["median_ms"] / 1000) / 1e9
            assert corrected["derived_busbw_GBps"] == pytest.approx(expected)
            assert corrected["derived_algbw_GBps"] == pytest.approx(expected)
            assert corrected["original_busbw_GBps"] == row["busbw_GBps"]
    large = next(p for p in points if p["rank"] == 0 and p["bytes_per_rank"] == 2**28)
    assert large["derived_busbw_GBps"] == pytest.approx(18.44, abs=.01)


def test_memory_review_preserves_measurement_phases():
    review = read("review.json")
    graph = read("p1a_runtime_pools.json")["probes"]["npu_graph_pool"]
    baseline = graph["free_before_capture"] - graph["baseline_delta"]
    assert graph["graph_pool_bytes"] == graph["baseline_delta"] == 35_651_584
    assert baseline - graph["free_after_capture"] == 0
    assert review["graph"]["derived_baseline_minus_after_bytes"] == 0
    assert review["graph"]["pure_graph_pool_bytes"] is None
    for api, filename in (("torch", "p0b_budget_torch.json"), ("acl", "p0a_budget_acl.json")):
        devices = read(filename)["devices"]
        for phase in ("after_init", "after_release"):
            assert review["budget"][f"{api}_{phase}_bytes"] == [d[f"free_{phase}"] for d in devices]
    assert review["budget"]["deployment_budget_bytes"] is None


def test_conflicting_mrope_claim_does_not_override_failed_capture():
    captured = read("p2_mrope.json")["three_row_identity"]
    unsupported_claim = read("p2_mrope_identity.json")
    review = read("review.json")["mrope"]
    assert captured["bit_identical"] is False
    assert captured["max_abs_diff"] == pytest.approx(5.45075457596721)
    assert unsupported_claim["bit_identical"] is True
    assert review["archived_script_identity_passed"] is False
    assert review["identity_admitted"] is False


def test_archive_does_not_grant_calibration_or_launch_evidence():
    from simverse.cost.profile_store import PROFILE_SETS, load_profile_set

    manifest, review = read("manifest.json"), read("review.json")
    assert manifest["calibration_admission"] == review["calibration_admission"] == "not-admitted"
    assert review["launch_overhead_us"] is None
    assert review["deployment_ranking_eligible"] is False
    assert "ev-ascend-910b3-20261117" not in PROFILE_SETS
    with pytest.raises(ValueError, match="未知 profile"):
        load_profile_set("ev-ascend-910b3-20261117")
