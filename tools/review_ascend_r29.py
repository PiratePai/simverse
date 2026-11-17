"""Replay the R29 archive review offline; never imports a device runtime.

The capture manifest and collectors stay immutable, including their defects.
Derived fields are separate from measurements and grant no admission.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = Path("docs/evidence/ascend-910b3-r29")
REPORT = EVIDENCE / "review.json"
SOURCE_COMMIT = "7de467d604d83cbee5eef76c2d12679356f3cad1"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def review(root: Path = ROOT) -> dict:
    def read(name):
        return json.loads((root / EVIDENCE / name).read_text(encoding="utf-8"))

    manifest = read("raw/manifest.json")
    inventory = {}
    mismatches = []
    for group in ("raw", "scripts", "docs"):
        for item in manifest[group]:
            path = item["path"].replace("\\", "/")
            content = (root / path).read_bytes()
            actual = {"bytes": len(content), "sha256": digest(content)}
            declared = {key: item[key] for key in actual}
            inventory[path] = actual
            if actual != declared:
                mismatches.append({"path": path, "declared": declared, "actual": actual})

    env = read("raw/env_identity.json")
    scripts = {Path(path).name: info["sha256"] for path, info in inventory.items()
               if "/scripts/" in path}
    snapshot_hashes = env["script_hashes"]
    script_identity = {
        "missing_from_env_snapshot": sorted(set(scripts) - set(snapshot_hashes)),
        "hash_mismatches": sorted(name for name in snapshot_hashes
                                  if scripts.get(name) != snapshot_hashes[name]),
    }
    points = []
    correctness = []
    for rank in (0, 1):
        data = read(f"raw/collectives_rank{rank}.json")
        world = data["world_size"]
        correctness.append({"rank": rank, **data["correctness_summary"]})
        for row in data["results"]:
            # n is the script's buffer scale, NOT uniform per-rank recv bytes.
            n = row["numel_per_rank"] * 2  # captured BF16
            op = row["op"]
            logical_input = n * world if op == "reduce_scatter" else n
            logical_output = n * world if op == "all_gather" else n
            normalized = n * world if op in ("all_gather", "reduce_scatter") else n
            factor = 2 * (world - 1) / world if op == "all_reduce" else (
                1 if op == "broadcast" else (world - 1) / world)
            algbw = normalized / (row["median_ms"] / 1000) / 1e9
            points.append({
                "rank": rank, "op": op, "buffer_scale_bytes": n,
                "logical_input_bytes": logical_input, "logical_output_bytes": logical_output,
                "normalized_bytes": normalized, "median_ms": row["median_ms"],
                "original_algbw_GBps": row["algbw_GBps"],
                "original_busbw_GBps": row["busbw_GBps"],
                "derived_algbw_GBps": algbw, "derived_busbw_GBps": algbw * factor,
            })

    graph = read("raw/graph_pools.json")
    stages = {s["stage"]: s for s in graph["snapshots"]}
    baseline = stages["baseline_after_warmup_and_empty_cache"]
    capture = stages["after_capture"]
    replay = stages["after_replay_1"]
    acl = read("raw/budget_acl.json")
    acl_overheads = []
    for device in acl["stages"]:
        previous = device["after_init"]["free_bytes"]
        values = []
        for step in device["steps"]:
            values.append(previous - step["free_bytes"] - step["requested_bytes"])
            previous = step["free_bytes"]
        acl_overheads.append({"logical_dev": device["logical_dev"],
                              "observed_free_delta_minus_requested_bytes": values})

    forward = read("raw/forward_execute.json")
    slices = [read(f"raw/tp2_slice_rank{rank}.json") for rank in (0, 1)]
    return {
        "schema_version": 1, "source_commit": SOURCE_COMMIT,
        "review_method": "offline archive/source audit; no new NPU execution",
        "original_manifest_sha256": digest((root / EVIDENCE / "raw/manifest.json").read_bytes()),
        "archive_inventory": inventory, "manifest_mismatches": mismatches,
        "script_identity": script_identity,
        "environment_limits": ["CANN version files missing; 9.1.0 is a path/collector declaration",
                               "runtime vLLM commit and local patches not archived",
                               "PCIe BDFs do not prove switch plane or HCCL transport"],
        "collectives": {
            "points": points, "correctness": correctness,
            "boundary": "per-rank batched host+sync wallclock, not kernel or wire bandwidth",
            "cross_rank_joint_timing_available": False,
            "definition_source": "https://github.com/NVIDIA/nccl-tests/tree/afd59abab774a6a4b492eb344bd0c845a0b7bb4d",
        },
        "graph": {
            "capture_minus_baseline_free_bytes": baseline["free_bytes"] - capture["free_bytes"],
            "capture_minus_baseline_allocated_bytes": capture["allocated_bytes"] - baseline["allocated_bytes"],
            "replay1_minus_capture_free_bytes": capture["free_bytes"] - replay["free_bytes"],
            "pure_graph_pool_bytes": None,
            "reason": "static_in clone occurs after baseline; no clone-only control or output finite/equality check",
        },
        "acl": {"incremental_observations": acl_overheads,
                "allocation_granularity_bytes": None, "deployment_budget_bytes": None},
        "mrope": {
            "numeric_admitted": False, "fused_target_support": "unresolved",
            "reasons": ["flat [tokens, heads*256] rotates only the first 64 dims, not 64 per head",
                        "oracle shares that slicing defect; tail check includes heads that should rotate",
                        "identity compares assembled frequencies with their device copy, not an independent T row",
                        "NeoX bands use adjacent pairs instead of indices j and j+32; low/high labels reversed",
                        "161002 is a recorded call failure, not proof that Qwen integration is unsupported"],
        },
        "forward": {
            "hardware_executed": forward["claims"]["hardware_executed"],
            "tensor_parallel_size": forward["engine"]["args"]["tensor_parallel_size"],
            "physical_participating_card": 5,
            "output_token_ids": forward["outputs"][0]["token_ids"],
            "weights_GiB_log_observation": 0.32, "kv_available_GiB_log_observation": 11.81,
            "failed_attempts_archived": False, "numerical_correctness_verified": False,
            "full_model_forward": False, "performance_measured": False,
            "reason": "success-only log; OOM reports are not weight-size measurements or a utilization upper bound",
        },
        "slices": {
            "captured_summaries": [data["summary"] for data in slices],
            "entries_per_rank_sent": [data["moe_ep2"]["entries_per_rank_sent"] for data in slices],
            "max_allocator_peak_bytes": max(s["peak_allocated_bytes"] for d in slices for s in d["mem"]),
            "full_model_npu_ep_verified": False,
            "limits": ["MLP tests one up/down GEMM, not full SwiGLU",
                       "GQA is unmasked; BF16 case computes scores in FP32",
                       "MoE uses E8/top2/H64/GELU with CPU expert compute, NPU metadata/output exchange",
                       "permutation flag checks received metadata multiset, not inverse roundtrip identity",
                       "empty experts are both owned by rank1; rank0 never exercises an owned empty expert",
                       "7/7 excludes KV/permutation flags from failure aggregation; toy peaks exclude runtime pools"],
        },
        "manifest_status_correction": "A-N3 generic slices executed; full model/state/capacity not verified (original not_executed is stale)",
        "calibration_admission": "not-admitted", "deployment_ranking_eligible": False,
        "does_not_unblock": ["G-D7", "R02b", "R17", "R18"],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--write", action="store_true", help="write derived review only")
    mode.add_argument("--check", action="store_true", help="reject stale derived review")
    args = parser.parse_args(argv)
    result = review()
    target = ROOT / REPORT
    if args.check:
        if json.loads(target.read_text(encoding="utf-8")) != result:
            raise SystemExit("R29 derived review drift; review changes before regenerating")
    elif args.write:
        target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"artifacts": len(result["archive_inventory"]),
                      "manifest_mismatches": len(result["manifest_mismatches"]),
                      "calibration_admission": result["calibration_admission"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
