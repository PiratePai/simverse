"""R29-0：环境与身份快照。

对应 A-N1「补版本、拓扑、输入与脚本哈希」与 3.1 节采集身份要求。
不测量性能，只固化可复核的栈身份。
"""

from __future__ import annotations

import json
import sys

import torch
import torch_npu

from r29_common import (
    SCRIPT_DIR,
    base_record,
    device_snapshot,
    npu_board_info,
    run_cmd,
    sha256_file,
    write_json,
)


def main() -> int:
    rec = base_record("r29_env_identity")

    rec["npu"] = device_snapshot(torch, torch_npu)
    rec["board"] = npu_board_info()

    # CANN / driver 版本文件（存在则读，缺失保持 null，不猜）
    version_files = {
        "toolkit_version_cfg": "/usr/local/Ascend/ascend-toolkit/latest/version.cfg",
        "driver_version_info": "/usr/local/Ascend/driver/version.info",
        "cann_version_cfg": "/usr/local/Ascend/cann-9.1.0/version.cfg",
    }
    rec["version_files"] = {}
    for key, path in version_files.items():
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                rec["version_files"][key] = {
                    "path": path,
                    "sha256": sha256_file(__import__("pathlib").Path(path)),
                    "content": fh.read()[:2000],
                }
        except FileNotFoundError:
            rec["version_files"][key] = {"path": path, "missing": True}
        except Exception as exc:  # noqa: BLE001
            rec["version_files"][key] = {"path": path, "error": str(exc)}

    # 本 run 的脚本哈希（自身 + 公共模块）
    rec["script_hashes"] = {
        p.name: sha256_file(p)
        for p in sorted(SCRIPT_DIR.glob("*.py"))
    }

    # 关键库的 device 能力面（只记录可查询到的部分，未知保持 null）
    probe = {}
    try:
        x = torch.ones(8, dtype=torch.float32, device="npu:0")
        probe["tiny_tensor_sum"] = float(x.sum().item())
    except Exception as exc:  # noqa: BLE001
        probe["tiny_tensor_error"] = f"{type(exc).__name__}: {exc}"
    try:
        free, total = torch.npu.mem_get_info(0)
        probe["mem_info_dev0"] = {"free_bytes": int(free), "total_bytes": int(total)}
    except Exception as exc:  # noqa: BLE001
        probe["mem_info_error"] = f"{type(exc).__name__}: {exc}"
    rec["runtime_probe"] = probe

    rec["npu_smi_cards"] = {
        str(card): run_cmd(
            ["bash", "-lc", f"npu-smi info -t board -i {card}"], timeout=30
        )
        for card in (5, 6)
    }

    path = write_json("env_identity.json", rec)
    print(json.dumps({"wrote": str(path), "devices": rec["npu"], "modules": rec["modules"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
