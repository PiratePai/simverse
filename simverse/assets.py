"""资产定位：源码检出与 wheel 安装共用一套路径（R10c）。

背景：wheel 由完整 Git 快照构建（CI 冒烟同法），安装到仓库外后
``Path(__file__).parent.parent`` 不再指向仓库根，``assets/config/*.json``
等受哈希约束的配置/证据文件会缺失，``simverse run-v2`` 直接失败。

本模块是唯一的资产根解析入口：
  * 源码检出（含 editable 安装）：仓库根的 ``assets/``、``docs/`` 优先，
    开发期修改即时生效，哈希核对读到的就是仓库文件原始字节；
  * wheel 安装：回退到包内 ``simverse/_data/`` 下的构建期拷贝
    （setup.py 按 WHEEL_DATA_MANIFEST 从仓库逐字节复制，不重排内容）。

WHEEL_DATA_MANIFEST 同时是打包清单（setup.py 消费）与核对基准
（测试比对 wheel 内容 == 仓库对应文件字节）。清单按实际消费路径确定
（2026-11-19 文件读取追踪：run-v2 默认场景只读 3 份 config；工作台
/api/v2/catalog 追加 identity/references/profile manifest/scenarios）：
  * assets/config/**、assets/scenarios/*.json——活跃目录与场景预设；
  * assets/identity/**——受哈希约束的身份/证据文件，保留原始字节；
  * assets/profiles/*/manifest.json——证据登记哈希所引用的清单文件
    （不含归档模型的校准数据文件：历史回放按用途留在仓库）；
  * docs/rebuild-20261106/references/**——pinned 证据源码/pin 清单。
明确不进 wheel：assets/cost_records/**（昇腾归档观测，独立环境条件
对照，显式选择时才加载，缺失按 ProfileIntegrityError 如实报错）、
assets/profiles/*/ 数据文件、完整 docs/（工作台文档入口在 wheel 下
如实 404，不伪造内容）。
"""
from __future__ import annotations

import os
from pathlib import Path

_PACKAGE_DIR = Path(__file__).resolve().parent

# 打包根目录名（构建期由 setup.py 生成；源码检出中不存在）
PACKAGED_DATA_DIRNAME = "_data"

# wheel 数据清单：(仓库相对源, 目标相对包内 _data 的目标)。
# 目标保持与仓库相同的相对布局，定位器只切换根目录，不逐文件映射。
WHEEL_DATA_MANIFEST: tuple[tuple[str, str], ...] = (
    ("assets/config", "assets/config"),
    ("assets/identity", "assets/identity"),
    ("assets/scenarios", "assets/scenarios"),
    ("assets/profiles/h200_qwen3_30b_a3b_tiny/manifest.json",
     "assets/profiles/h200_qwen3_30b_a3b_tiny/manifest.json"),
    ("assets/profiles/h200_qwen3_235b_a22b/manifest.json",
     "assets/profiles/h200_qwen3_235b_a22b/manifest.json"),
    ("docs/rebuild-20261106/references", "docs/rebuild-20261106/references"),
)


def repo_root() -> Path | None:
    """源码检出根；wheel 安装（无仓库布局）返回 None。"""
    root = _PACKAGE_DIR.parent
    if (root / "assets").is_dir() and (root / "simverse").is_dir():
        return root
    return None


def _packaged_root() -> Path:
    return _PACKAGE_DIR / PACKAGED_DATA_DIRNAME


def assets_dir() -> Path:
    """assets/ 根：源码检出的仓库 assets，否则包内拷贝。"""
    root = repo_root()
    return (root / "assets") if root else _packaged_root() / "assets"


def docs_dir() -> Path:
    """docs/ 根：wheel 下仅含清单打包的 references 子树。"""
    root = repo_root()
    return (root / "docs") if root else _packaged_root() / "docs"


def scenario_dir() -> Path:
    return assets_dir() / "scenarios"


def packaged_data_dir() -> Path:
    """包内 _data 目录（wheel 资产根；源码检出可能不存在）。"""
    return _packaged_root()


def default_output_dir() -> Path:
    """CLI 未显式给 -o 时的输出目录：源码检出写仓库 outputs/（既有
    行为），wheel 安装写当前工作目录（site-packages 不可写也不该写）。"""
    root = repo_root()
    return (root / "outputs") if root else Path.cwd() / "outputs"


def iter_manifest_files(repo: Path) -> list[tuple[Path, Path]]:
    """展开清单为 (仓库源文件, 包内目标文件) 列表；源缺失时报错。"""
    out: list[tuple[Path, Path]] = []
    for src_rel, dst_rel in WHEEL_DATA_MANIFEST:
        src = repo / src_rel
        if src.is_dir():
            files = sorted(p for p in src.rglob("*") if p.is_file())
            if not files:
                raise FileNotFoundError(f"清单目录为空：{src_rel}")
            for f in files:
                out.append((f, Path(dst_rel) / f.relative_to(src)))
        elif src.is_file():
            out.append((src, Path(dst_rel)))
        else:
            raise FileNotFoundError(f"清单条目缺失：{src_rel}")
    return out


def manifest_bytes_match(repo: Path, packaged: Path) -> list[str]:
    """核对包内拷贝与仓库原始字节一致；返回不一致条目描述。"""
    bad = []
    for src, dst in iter_manifest_files(repo):
        target = packaged / dst
        if not target.is_file():
            bad.append(f"缺失 {dst}")
        elif target.read_bytes() != src.read_bytes():
            bad.append(f"字节不符 {dst}")
    return bad


def code_revision() -> str:
    """当前代码身份（git HEAD；wheel/无 git 环境如实返回 unknown）。"""
    import subprocess

    root = repo_root()
    if root is None:
        return "unknown"
    try:
        out = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return "unknown"


# 供 setup.py 以纯标准库方式取得清单（避免构建期导入包内其他模块）
if os.environ.get("SIMVERSE_PRINT_MANIFEST"):  # pragma: no cover
    for _src, _dst in WHEEL_DATA_MANIFEST:
        print(f"{_src}\t{_dst}")
