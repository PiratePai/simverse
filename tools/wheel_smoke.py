"""wheel 安装冒烟（R10c）：完整 Git 快照构建 → 隔离 venv → 仓库外执行。

步骤（每步失败即退出非 0，不静默跳过）：
  1. ``git archive HEAD`` 导出完整快照到临时 stage（保证内容 == 提交树，
     不受未提交改动影响）；
  2. 在 stage 内构建 wheel（setup.py 把 assets 清单拷入 simverse/_data）；
  3. 独立 venv 安装该 wheel（连同 flask 依赖），不在本仓 editable 安装；
  4. 从仓库外（临时 cwd）执行真实 ``simverse run-v2``，校验退出码 0、
     产生有效 run_id、模块来自已安装包而非本仓源码；
  5. 工作台首页与 /api/v2/catalog 返回 200，目录恰两款主模型；
  6. wheel 内清单文件与仓库原始字节逐一致（受哈希约束资产不重排）。

用法：py -3.12 tools/wheel_smoke.py [--keep]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _run(cmd, **kw) -> subprocess.CompletedProcess:
    print("+", " ".join(str(c) for c in cmd), flush=True)
    out = subprocess.run(cmd, capture_output=True, text=True,
                         encoding="utf-8", errors="replace", **kw)
    if out.returncode != 0:
        print(out.stdout)
        print(out.stderr)
        raise SystemExit(f"步骤失败（exit {out.returncode}）：{cmd[0]}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="store_true", help="保留 stage/venv 便于排查")
    args = ap.parse_args()

    work = Path(tempfile.mkdtemp(prefix="simverse-wheel-smoke-"))
    stage = work / "stage"
    stage.mkdir()
    print(f"工作目录：{work}")

    # 1. 完整 Git 快照（git archive = 提交树逐字节；未提交改动不进入）。
    #    解包用 Python tarfile：仓库含中文文件名 .bat，Windows bsdtar 会
    #    因代码页把条目名截成空路径而失败。
    import tarfile
    tar = stage / "_snap.tar"
    subprocess.run(["git", "-C", str(REPO), "archive", "--format=tar",
                    "-o", str(tar), "HEAD"], check=True)
    with tarfile.open(tar) as tf:
        tf.extractall(stage, filter="data")
    tar.unlink()

    # 2. 构建 wheel（默认隔离构建；WHEEL_SMOKE_NO_ISOLATION=1 时复用
    # 当前环境——离线 CI/本机 setuptools 满足要求时用）
    dist = work / "dist"
    dist.mkdir()
    no_iso = (["--no-build-isolation"]
              if os.environ.get("WHEEL_SMOKE_NO_ISOLATION") else [])
    _run([sys.executable, "-m", "pip", "wheel", "--no-deps", *no_iso,
          "-w", str(dist), str(stage)])
    wheels = sorted(dist.glob("simverse-*.whl"))
    if len(wheels) != 1:
        raise SystemExit(f"预期恰 1 个 wheel，得到 {wheels}")
    wheel = wheels[0]
    print(f"wheel：{wheel.name}")

    # 6a. wheel 清单文件与仓库原始字节一致（安装前直接对 zip 校）
    sys.path.insert(0, str(REPO))
    from simverse.assets import iter_manifest_files
    manifest = iter_manifest_files(stage)
    with zipfile.ZipFile(wheel) as zf:
        names = set(zf.namelist())
        for src, dst in manifest:
            entry = f"simverse/_data/{dst.as_posix()}"
            if entry not in names:
                raise SystemExit(f"wheel 缺清单文件：{entry}")
            if zf.read(entry) != src.read_bytes():
                raise SystemExit(f"wheel 文件字节与仓库不一致：{entry}")
    print("清单字节核对：通过（%d 个文件）" % len(manifest))

    # 3. 隔离 venv 安装
    venv = work / "venv"
    _run([sys.executable, "-m", "venv", str(venv)])
    py = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    _run([str(py), "-m", "pip", "install", "--quiet", "--upgrade", "pip"])
    _run([str(py), "-m", "pip", "install", "--quiet", str(wheel)])

    # 4. 仓库外执行真实 CLI；确认模块来自已安装包
    outside = work / "outside"
    outside.mkdir()
    probe = (
        "import simverse, simverse.cli, json, sys;"
        "from simverse.assets import repo_root;"
        "assert repo_root() is None, 'repo_root 应为 None（wheel 模式）';"
        "print(json.dumps({'module': simverse.__file__,"
        " 'site': simverse.cli.__file__}))"
    )
    out = _run([str(py), "-c", probe], cwd=str(outside))
    info = json.loads(out.stdout.strip().splitlines()[-1])
    for key in ("module", "site"):
        if str(work) not in info[key].replace("/", os.sep) \
                and str(venv) not in info[key]:
            raise SystemExit(f"模块未来自安装包：{info}")
    print(f"模块来源：{info['module']}")

    command = py.with_name("simverse.exe" if os.name == "nt" else "simverse")
    out = _run([str(command), "run-v2", "--out", "run.json"], cwd=str(outside))
    run = json.loads((outside / "run.json").read_text(encoding="utf-8"))["manifest"]
    assert run["status"] == "completed" and re.fullmatch(r"sha256:[0-9a-f]{64}", run["run_id"]), run
    run_ok = "run_id=" in out.stdout and "status=completed" in out.stdout
    if not run_ok:
        raise SystemExit("run-v2 输出缺少 status=completed/run_id：\n" + out.stdout)
    print("run-v2：completed，run_id 已产生")

    # 5. 工作台首页 + catalog 200，恰两款主模型
    check = (
        "import json;"
        "from pathlib import Path;"
        "from simverse.webapp import app;"
        "c=app.test_client();"
        "r1=c.get('/'); r2=c.get('/api/v2/catalog');"
        "assert r1.status_code==200, r1.status_code;"
        "assert r2.status_code==200, r2.status_code;"
        "catalog=r2.get_json();"
        "models=set(catalog['catalog']['models']);"
        "assert models=={'qwen36_35b_a3b','deepseek_v41_flash'}, models;"
        "preset=next(x for x in catalog['scenarios'] if x['id']=='qwen36_dflash_default');"
        "response=c.post('/api/v2/run',json=preset['scenario']);"
        "assert response.status_code==200, response.status_code;"
        "api=response.get_json();"
        "cli=json.loads(Path('run.json').read_text(encoding='utf-8'));"
        "assert api['manifest']['input_digest']==cli['manifest']['input_digest']==preset['scenario_digest'];"
        "assert api['metrics']['M01_tpot_selected']['value']==cli['metrics']['M01_tpot_selected']['value'];"
        "status=c.get('/docs/FOCUS_STATUS.md');"
        "assert status.status_code==200 and b'../ROADMAP.md' in status.data;"
        "roadmap=c.get('/ROADMAP.md');"
        "assert roadmap.status_code==200 and b'docs/MERGE_CRITERIA.md' in roadmap.data;"
        "assert c.get('/docs/MERGE_CRITERIA.md').status_code==200;"
        "assert c.get('/missing.md').status_code==404;"
        "print('workbench/catalog/docs OK')"
    )
    _run([str(py), "-c", check], cwd=str(outside))

    if not args.keep:
        import shutil
        shutil.rmtree(work, ignore_errors=True)
        print("已清理临时目录")
    print("wheel 冒烟：全部通过")


if __name__ == "__main__":
    main()
