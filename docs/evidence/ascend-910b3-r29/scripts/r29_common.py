"""R29 采集公共模块。

约定（对应 docs/ascend-validation-plan.md 第 3.1 节）：
- 所有脚本只能通过本模块取路径，禁止绝对路径；
- 每个 run 的原始产物写到 docs/evidence/ascend-910b3-r29/raw/，并带 SHA256；
- 每条记录都带上设备可见性与环境身份，便于事后核对；
- 失败必须显式记录 error/stage，不得静默跳过。
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path

RUN_ID = "ascend-910b3-r29"
ROOT = Path(__file__).resolve().parents[4]
RUN_DIR = ROOT / "docs" / "evidence" / RUN_ID
RAW_DIR = RUN_DIR / "raw"
SCRIPT_DIR = RUN_DIR / "scripts"


def ensure_dirs() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(name: str, payload: dict) -> Path:
    ensure_dirs()
    path = RAW_DIR / name
    text = json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=False)
    path.write_text(text + "\n", encoding="utf-8")
    return path


def run_cmd(argv: list[str], env: dict | None = None, timeout: int = 60) -> dict:
    try:
        proc = subprocess.run(
            argv, capture_output=True, text=True, timeout=timeout, env=env
        )
        return {
            "argv": argv,
            "returncode": proc.returncode,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
        }
    except Exception as exc:  # noqa: BLE001 - 采集侧需要把失败原样记下来
        return {"argv": argv, "error": f"{type(exc).__name__}: {exc}"}


def module_versions() -> dict:
    out: dict = {"python": sys.version.split()[0], "executable": sys.executable}
    for name in ("torch", "torch_npu", "vllm", "vllm_ascend", "numpy"):
        try:
            mod = __import__(name)
            version = getattr(mod, "__version__", None)
            if not version:
                try:
                    from importlib.metadata import version as dist_version

                    version = dist_version(name)
                except Exception:  # noqa: BLE001
                    version = "unknown"
            out[name] = version
        except Exception as exc:  # noqa: BLE001
            out[name] = f"MISSING: {type(exc).__name__}: {str(exc)[:120]}"
    return out


def git_identity() -> dict:
    return {
        "commit": run_cmd(["git", "-C", str(ROOT), "rev-parse", "HEAD"])["stdout"],
        "branch": run_cmd(["git", "-C", str(ROOT), "rev-parse", "--abbrev-ref", "HEAD"])[
            "stdout"
        ],
        "dirty": run_cmd(["git", "-C", str(ROOT), "status", "--porcelain"])["stdout"],
    }


def host_identity() -> dict:
    return {
        "hostname": socket.gethostname(),
        "kernel": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
    }


def npu_board_info() -> dict:
    env = dict(os.environ)
    env.setdefault("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = (
        "/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common"
        ":/usr/local/Ascend/driver/lib64/driver:" + env["LD_LIBRARY_PATH"]
    )
    sudo_ok = run_cmd(["npu-smi", "info"], env=env, timeout=30)
    return {"npu_smi_info": sudo_ok}


def base_record(kind: str) -> dict:
    return {
        "kind": kind,
        "run_id": RUN_ID,
        "captured_at": now_iso(),
        "ascend_rt_visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES", ""),
        "host": host_identity(),
        "git": git_identity(),
        "modules": module_versions(),
    }


def device_snapshot(torch_mod, torch_npu_mod) -> dict:
    count = torch_mod.npu.device_count()
    devices = []
    for i in range(count):
        devices.append(
            {
                "logical_index": i,
                "name": torch_mod.npu.get_device_name(i),
            }
        )
    return {"device_count": count, "devices": devices}
