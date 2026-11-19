"""Check archived evidence against the last revision of its numerical contract."""
import argparse
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVISION = "a457c76665234fb09c9026b9e3ff5aad4ed89b2a"
SCRIPTS = {
    "r23": ("tools/replay_qwen36_mtp.py", "--check"),
    "r26": ("tools/replay_qwen36_mrope.py", "--check"),
    "r27": ("tools/replay_cost_integrity_r27.py", "--check"),
}


def replay_frozen(name: str) -> None:
    script = SCRIPTS[name]
    with tempfile.TemporaryDirectory(prefix="simverse-frozen-") as tmp:
        work = Path(tmp)
        archive = work / "revision.tar"
        tree = work / "tree"
        tree.mkdir()
        subprocess.run(["git", "-C", str(ROOT), "archive", "--format=tar",
                        "-o", str(archive), REVISION], check=True)
        with tarfile.open(archive) as source:
            source.extractall(tree, filter="data")
        subprocess.run([sys.executable, "-X", "utf8", *script], cwd=tree, check=True)


def legacy_entry(name: str) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="核对冻结结果")
    parser.parse_args()
    replay_frozen(name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", choices=tuple(SCRIPTS))
    replay_frozen(parser.parse_args().evidence)
