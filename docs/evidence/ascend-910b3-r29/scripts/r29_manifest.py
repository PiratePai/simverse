"""R29 清单：为 run 目录内每个产物计算 SHA256，并记录身份与使用边界。

清单只证明"归档一致性"，不构成校准准入，也不把任何 fixture 结果升级为整模证据。
"""

from __future__ import annotations

import json
import sys

from r29_common import ROOT, RUN_DIR, RAW_DIR, SCRIPT_DIR, base_record, sha256_file, write_json

ROLES = {
    "env_identity.json": ("栈/设备/板卡/版本文件/脚本哈希", "板卡序列号可核对；HCCS 路由未验证"),
    "budget_acl.json": ("ACL 分阶段 malloc/free + 返回码 + context 句柄", "空闲快照，不是容量保证"),
    "budget_torch.json": ("torch allocator 分阶段统计", "与 ACL 不可相减"),
    "collectives_rank0.json": ("rank0 五类 collective × 10 尺寸 + 数值校验", "批内墙钟；transport 未验证"),
    "collectives_rank1.json": ("rank1 同规格记录", "同上"),
    "graph_pools.json": ("eager/预热/baseline/capture/replay/release 分阶段", "单 workload，不是整模峰值"),
    "mrope.json": ("Qwen3.6 参数域 MRoPE 数值 + 融合入口能力探测", "非融合路径 fixture 级"),
    "forward_plan.json": ("forward 预检计划（dry-run）", "计划不等于执行"),
    "forward_execute.json": ("vLLM 0.32 + vllm-ascend fixture 执行结果", "2 层随机权重，非整模"),
    "forward_execute.log": ("执行日志（含两次 OOM 负例）", "原始日志，未编辑"),
    "tp2_slice_rank0.json": ("TP2/EP2 切片：数值、路由、显存（rank0）", "切片级；均衡路由，不代表倾斜分布"),
    "tp2_slice_rank1.json": ("TP2/EP2 切片：数值、路由、显存（rank1）", "同上"),
}


def artifact(path) -> dict:
    name = path.name
    role, limits = ROLES.get(name, ("本轮脚本/说明", "见 README"))
    return {
        "path": str(path.relative_to(ROOT)),
        "role": role,
        "limits": limits,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def main() -> int:
    rec = base_record("r29_manifest")
    rec["run_dir"] = str(RUN_DIR.relative_to(ROOT))
    rec["device_restriction"] = "ASCEND_RT_VISIBLE_DEVICES=5,6 (physical cards 5 and 6 only)"
    rec["raw"] = [
        artifact(p) for p in sorted(RAW_DIR.glob("*")) if p.is_file() and p.name != "manifest.json"
    ]
    rec["scripts"] = [artifact(p) for p in sorted(SCRIPT_DIR.glob("*.py"))]
    docs = [RUN_DIR / "README.md", ROOT / "docs" / "validation" / "ascend-910b3-r29.md"]
    rec["docs"] = [artifact(p) for p in docs if p.exists()]

    env_path = RAW_DIR / "env_identity.json"
    if env_path.exists():
        env = json.loads(env_path.read_text(encoding="utf-8"))
        rec["stack"] = env.get("modules", {})
        rec["devices"] = env.get("npu", {})
        rec["version_files"] = {
            k: {"path": v.get("path"), "sha256": v.get("sha256"), "missing": v.get("missing", False)}
            for k, v in (env.get("version_files") or {}).items()
        }

    rec["calibration_admission"] = "not-admitted"
    rec["does_not_unblock"] = ["G-D7", "R02b", "R17", "R18"]
    rec["not_executed"] = [
        "A-N3 TP2 numerical/shard reassembly/state conservation",
        "A-N5 profiling import and local calibration",
        "A-N6 matched E2E",
        "HCCS transport verification (no plog captured)",
        "fused torch_npu.npu_mrope on Qwen3.6 geometry (error 161002)",
    ]
    rec["boundary"] = (
        "Archival consistency only: hashes prove files are unchanged since capture. "
        "They do not prove numerical correctness of the full model, performance, or calibration admission."
    )

    path = write_json("manifest.json", rec)
    print(json.dumps({"wrote": str(path), "artifacts": len(rec["raw"]) + len(rec["scripts"]) + len(rec["docs"])}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
