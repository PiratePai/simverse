"""setuptools 构建：把 assets.py 登记的 wheel 数据清单拷入包内 _data/。

清单来源 simverse/assets.py（单一事实源；该模块纯标准库，构建隔离环境
可安全导入）。构建期逐字节复制，不重排/改写内容——受哈希约束的配置与
证据文件保持仓库原始字节（MERGE_CRITERIA §1.1/§3.3）。
"""
import os
import shutil
import sys

from setuptools import setup
from setuptools.command.build_py import build_py

REPO = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, REPO)

from simverse.assets import iter_manifest_files  # noqa: E402


class BuildPyWithAssets(build_py):
    """build_py 钩子：清单文件在 build_lib 内还原仓库相对布局。"""

    def run(self):
        super().run()
        if not self.packages or "simverse" not in self.packages:
            return
        target_root = os.path.join(self.build_lib, "simverse", "_data")
        for src, dst in iter_manifest_files(__import__("pathlib").Path(REPO)):
            target = os.path.join(target_root, str(dst).replace("/", os.sep))
            os.makedirs(os.path.dirname(target), exist_ok=True)
            shutil.copyfile(str(src), target)


setup(cmdclass={"build_py": BuildPyWithAssets})
