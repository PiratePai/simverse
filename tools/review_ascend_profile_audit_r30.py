"""Replay PR #26 review (historical record) from committed evidence; no pack/raw/NPU.

本工具只复核**历史审查记录**（提交 89803d0 / 8c07518 时的审计包状态）是否与
committed review.json 一致；它不再对当前审计包做断言——当前包已由修复后的
导入器从原件重建，见 tools/review_ascend_profile_audit_r30_rebuild.py。
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/evidence/ascend-profile-audit-r30-review/review.json"
REFERENCE = ROOT / "docs/evidence/ascend-profile-audit-r30-review/reference/patch_qwen3_5.py"
HISTORICAL_MANIFEST_SHA = "165cfed78fc5400e5fed3ae401160357409f1ff2c2af1483754a443c98e33fe5"
HISTORICAL_COMMIT = "89803d0a09c13bcd237049c2c2ccefe7ec381f4d"
REFERENCE_SHA = "a9a71634ed138f590fbc3ad0de277a5f56a06451522fd091b2657d6daf6084b0"


def historical() -> dict:
    review = json.loads(OUT.read_text(encoding="utf-8"))
    assert review["source_commit"] == HISTORICAL_COMMIT
    assert review["source_manifest_sha256"] == HISTORICAL_MANIFEST_SHA
    assert review["decision"].startswith("hold-merge")
    assert len(review["missing_consumed_source_hashes"]) == 6
    assert all(len(files) == 4
               for files in review["missing_consumed_source_hashes"].values())
    assert hashlib.sha256(REFERENCE.read_bytes()).hexdigest() == REFERENCE_SHA
    return review


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    result = historical()
    if args.check:
        assert json.loads(OUT.read_text(encoding="utf-8")) == result, \
            "Historical review record drift"
    print(result["decision"])
    print("historical source commit:", result["source_commit"])
    print("superseded by: docs/validation/ascend-profile-audit-r30-rebuild-20261118.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
