"""R07b/R09 offline request trace acceptance and rejection cases."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from simverse.execute.request_trace import summarize_request_trace

FIXTURE = Path(__file__).with_name("fixtures") / "r07b_r09_request_trace.json"


@pytest.fixture
def trace():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def write_trace(directory, trace):
    path = directory / "trace.json"
    path.write_text(json.dumps(trace), encoding="utf-8")
    return path


def test_offline_request_trace_cli_matches_plan():
    proc = subprocess.run(
        [sys.executable, "-m", "simverse.cli", "trace-summary", str(FIXTURE)],
        cwd=FIXTURE.parents[2], capture_output=True, text=True, encoding="utf-8",
        errors="replace", timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    result = json.loads(proc.stdout)
    assert result["manifest"]["source_kind"] == "synthetic"
    assert result["window"]["time_basis"] == "committed"
    assert result["window_completed_requests"] == 2
    assert result["window_token_events"] == 7
    assert result["completed_token_per_s"] == 100
    assert result["event_token_per_s"] == 175
    assert result["status_counts"] == {"completed": 3, "cancelled": 1, "failed": 1}
    assert result["ttft"] == {
        "sample_count": 2, "estimator": "nearest_rank", "p50_ms": 4, "p95_ms": 6,
        "reason": None,
    }
    assert result["requests"]["A"]["itl_ms"] == [12, 0, 30]
    assert result["requests"]["B"] == {
        "status": "completed", "output_tokens": 3, "ttft_ms": 6,
        "ttft_reason": None, "e2e_ms": 44, "itl_ms": [20, 18], "mean_itl_ms": 19,
    }
    assert result["requests"]["C"]["ttft_ms"] == 4
    assert result["requests"]["C"]["mean_itl_ms"] is None
    assert result["requests"]["E"]["ttft_ms"] is None
    assert result["requests"]["E"]["ttft_reason"]
    assert result["goodput"] == {"requests_per_s": 25, "token_per_s": 25}


@pytest.mark.parametrize(("path", "value", "message"), [
    (("requests", 1, "emissions", 1, "time_ns"), 9_000_000, "time_ns"),
    (("requests", 0, "emissions", 1, "token_indices"), [1, 1], "token_indices"),
    (("requests", 1, "emissions", 0, "time_ns"), 3_000_000, "time_ns"),
    (("requests", 1, "terminal_ns"), 47_000_000, "terminal_ns"),
    (("requests", 1, "emissions", 0, "clock_id"), "client", "clock_id"),
    (("requests", 1, "emissions", 0, "time_basis"), "server_emitted", "time_basis"),
    (("requests", 1, "output_tokens"), 4, "output_tokens"),
    (("requests", 1, "request_id"), "A", "request_id"),
    (("requests", 0, "sample_id"), "different-workload", "workload_sha256"),
    (("window", "end_ns"), 10_000_000, "start_ns"),
    (("window", "time_basis"), "client_observed", "time_basis"),
    (("manifest", "source_kind"), "measured", "source_kind"),
])
def test_trace_rejects_invalid_request_data(trace, tmp_path, path, value, message):
    node = trace
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    with pytest.raises(ValueError, match=message):
        summarize_request_trace(write_trace(tmp_path, trace))


def test_no_completed_samples_keeps_unknown_ttft(trace, tmp_path):
    trace["window"].update(start_ns=51_000_000, end_ns=52_000_000, drain_end_ns=52_000_000)
    del trace["slo"]
    result = summarize_request_trace(write_trace(tmp_path, trace))
    assert result["ttft"]["sample_count"] == 0
    assert result["ttft"]["p50_ms"] is None
    assert result["ttft"]["p95_ms"] is None
    assert result["ttft"]["reason"]
    assert result["goodput"] is None


@pytest.mark.parametrize(("method", "proposals"), [
    ("dflash_v1", [(3, 3, 0, 1), (3, 3, 3, 4), (2, 2, 1, 1)]),
    ("mtp", [(1, 1, 0, 1), (1, 1, 1, 2)]),
])
def test_speculative_rounds_and_span_refs(trace, tmp_path, method, proposals):
    trace["manifest"]["method"] = method
    request = trace["requests"][0]
    request["emissions"].append({"time_ns": 50_000_000, "token_indices": [4, 5]})
    request["output_tokens"] = 6
    trace["rounds"] = []
    first = 0
    for i, (proposed, verified, accepted, committed) in enumerate(proposals):
        trace["rounds"].append({
            "round_id": str(i), "request_id": "A",
            "configured_k": 3 if method == "dflash_v1" else 1,
            "proposed": proposed, "verified": verified, "accepted": accepted,
            "committed": committed, "token_start": first,
        })
        first += committed
    span = {
        "span_id": "verify-0", "request_id": "A", "round_id": "0", "stage": "verify",
        "role": "target", "resource": "target:0", "timing_boundary": "server_wall",
        "start_ns": 2_000_000, "end_ns": 8_000_000,
    }
    trace["batches"] = {"batch-0": ["A", "B"]}
    trace["spans"] = [span, {
        "span_id": "batched", "batch_id": "batch-0", "stage": "prefill", "role": "target",
        "resource": "target:0", "timing_boundary": "server_wall",
        "start_ns": None, "end_ns": None, "timing_reason": "not in trace",
    }]
    assert summarize_request_trace(write_trace(tmp_path, trace))["completed_token_per_s"] == 100

    span["round_id"] = "absent"
    with pytest.raises(ValueError, match="round_id"):
        summarize_request_trace(write_trace(tmp_path, trace))
    span["round_id"] = "0"
    trace["rounds"][0]["accepted"] = trace["rounds"][0]["verified"] + 1
    with pytest.raises(ValueError, match="accepted"):
        summarize_request_trace(write_trace(tmp_path, trace))


def test_captured_trace_validates_raw_file_identity(trace, tmp_path):
    raw = tmp_path / "raw.jsonl"
    raw.write_bytes(b'{"raw":true}\n')
    trace["manifest"].update(
        source_kind="captured", backend_revision="example-backend",
        model_revision="example-model", tokenizer_revision="example-tokenizer",
        device="example-device", topology="single-device", precision="bf16",
        runtime_identity="example-runtime", command="record fixture", tp=1, ep=1,
        method_config={}, raw_files={"raw.jsonl": "sha256:" + hashlib.sha256(raw.read_bytes()).hexdigest()},
    )
    trace["window"]["time_basis"] = "server_emitted"
    path = write_trace(tmp_path, trace)
    summary = summarize_request_trace(path)
    assert summary["manifest"]["source_kind"] == "captured"
    assert summary["window"]["time_basis"] == "server_emitted"
    raw.write_bytes(b'{"raw":false}\n')
    with pytest.raises(ValueError, match="SHA256"):
        summarize_request_trace(path)


def test_trace_cli_rejects_duplicate_json_key(tmp_path):
    path = tmp_path / "duplicate.json"
    path.write_text('{"schema_version":"bad","schema_version":"1.0"}', encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, "-m", "simverse.cli", "trace-summary", str(path)],
        cwd=FIXTURE.parents[2], capture_output=True, text=True, encoding="utf-8",
        errors="replace", timeout=30,
    )
    assert proc.returncode == 2
    assert not proc.stdout
    assert "schema_version" in proc.stderr
