"""纯标准库 PNG 折线图渲染（无 matplotlib 依赖；CLI 可复算导出）。

只做渲染，不做任何物理计算——数据全部来自 sweep 结果（前端/导出禁物理计算的
同一原则）。支持多系列、对数 x 轴、水平参考线（speedup=1）。
"""
from __future__ import annotations

import math
import struct
import zlib
from typing import Dict, List, Optional, Sequence, Tuple


def _png(width: int, height: int, rgba: bytearray) -> bytes:
    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    raw = b"".join(b"\x00" + bytes(rgba[y * width * 4:(y + 1) * width * 4])
                   for y in range(height))
    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 6))
            + chunk(b"IEND", b""))


class Canvas:
    def __init__(self, width: int, height: int, bg=(255, 255, 255)):
        self.w, self.h = width, height
        self.px = bytearray(bytes(bg) * (width * height))  # RGB×3，输出时补 A

    def _set(self, x: int, y: int, rgb: Tuple[int, int, int]):
        if 0 <= x < self.w and 0 <= y < self.h:
            i = (y * self.w + x) * 3
            self.px[i:i + 3] = bytes(rgb)

    def line(self, x0: float, y0: float, x1: float, y1: float,
             rgb: Tuple[int, int, int], width: int = 2):
        steps = max(int(max(abs(x1 - x0), abs(y1 - y0)) * 2) + 1, 1)
        for s in range(steps + 1):
            t = s / steps
            cx, cy = x0 + (x1 - x0) * t, y0 + (y1 - y0) * t
            for dx in range(-(width // 2), width // 2 + 1):
                for dy in range(-(width // 2), width // 2 + 1):
                    self._set(int(round(cx)) + dx, int(round(cy)) + dy, rgb)

    def text(self, x: int, y: int, s: str, rgb=(40, 40, 40), scale: int = 1):
        """5×7 点阵字体（可打印 ASCII 子集）；中文不渲染为字形，调用方应用英文标签。"""
        glyphs = _FONT
        cx = x
        for ch in s:
            g = glyphs.get(ch, glyphs.get("?", [(0, 1), (1, 0), (2, 1)]))
            for (gx, gy) in g:
                for sx in range(scale):
                    for sy in range(scale):
                        self._set(cx + gx * scale + sx, y + gy * scale + sy, rgb)
            cx += 6 * scale

    def to_png(self) -> bytes:
        rgba = bytearray()
        for i in range(0, len(self.px), 3):
            rgba += self.px[i:i + 3] + b"\xff"
        return _png(self.w, self.h, rgba)


def _fmt(v: float) -> str:
    if v == 0:
        return "0"
    if abs(v) >= 1000:
        return f"{v / 1000:.0f}k" if abs(v) >= 10000 else f"{v:.0f}"
    if abs(v) >= 10:
        return f"{v:.0f}"
    if abs(v) >= 1:
        return f"{v:.1f}"
    return f"{v:.2f}"


def line_chart(series: Dict[str, List[Tuple[float, float]]], title: str,
               x_label: str, y_label: str, log_x: bool = False,
               hline: Optional[float] = None, width: int = 900, height: int = 540,
               legend_at: Tuple[int, int] = (620, 40)) -> bytes:
    """多系列折线图。series: {label: [(x, y), ...]}；颜色自动分配。"""
    colors = [(31, 119, 180), (214, 39, 40), (44, 160, 44), (148, 103, 189),
              (255, 127, 14), (23, 190, 207)]
    ml, mr, mt, mb = 72, 16, 52, 48
    c = Canvas(width, height, (252, 252, 252))
    all_pts = [p for pts in series.values() for p in pts]
    if not all_pts:
        c.text(20, 20, "no data")
        return c.to_png()
    tx = [math.log10(p[0]) if log_x and p[0] > 0 else p[0] for p in all_pts]
    xs0, xs1 = min(tx), max(tx)
    ys0, ys1 = min(p[1] for p in all_pts), max(p[1] for p in all_pts)
    if xs0 == xs1:
        xs0, xs1 = xs0 - 1, xs1 + 1
    pad = (ys1 - ys0) * 0.08 or 0.1
    ys0, ys1 = ys0 - pad, ys1 + pad
    if hline is not None:
        ys0 = min(ys0, hline - pad)
        ys1 = max(ys1, hline + pad)

    def px(x: float) -> int:
        return int(ml + (x - xs0) / (xs1 - xs0) * (width - ml - mr))

    def py(y: float) -> int:
        return int(height - mb - (y - ys0) / (ys1 - ys0) * (height - mt - mb))

    # 网格与轴
    for gy in range(5):
        yv = ys0 + (ys1 - ys0) * gy / 4
        c.line(ml, py(yv), width - mr, py(yv), (225, 225, 225), 1)
        c.text(8, py(yv) - 3, _fmt(yv), scale=1)
    xt = sorted(set(p[0] for p in all_pts))
    if len(xt) > 8:
        xt = xt[:: max(1, len(xt) // 7)]
    for xv in xt:
        t = math.log10(xv) if log_x and xv > 0 else xv
        c.line(px(t), mt, px(t), height - mb, (238, 238, 238), 1)
        c.text(px(t) - 12, height - mb + 8, _fmt(xv), scale=1)
    c.line(ml, mt, ml, height - mb, (120, 120, 120), 1)
    c.line(ml, height - mb, width - mr, height - mb, (120, 120, 120), 1)
    if hline is not None:
        c.line(ml, py(hline), width - mr, py(hline), (150, 150, 150), 1)
    # 系列
    for i, (label, pts) in enumerate(series.items()):
        col = colors[i % len(colors)]
        scaled = [((math.log10(x) if log_x and x > 0 else x), y) for x, y in pts]
        for (x0, y0), (x1, y1) in zip(scaled, scaled[1:]):
            c.line(px(x0), py(y0), px(x1), py(y1), col, 3)
        if scaled:
            c._set(px(scaled[0][0]), py(scaled[0][1]), col)
    # 标题/轴标签/图例
    c.text((width - len(title) * 6) // 2, 12, title, (20, 20, 20))
    c.text(8, height - 16, x_label, (60, 60, 60))
    lx, ly = legend_at
    for i, label in enumerate(series):
        col = colors[i % len(colors)]
        c.line(lx - 26, ly + i * 18 + 4, lx - 6, ly + i * 18 + 4, col, 3)
        c.text(lx, ly + i * 18, label[:28], (40, 40, 40))
    c.text(6, 24, y_label, (60, 60, 60))
    return c.to_png()


# 5×7 点阵字体（数字/字母/常用符号）
_G = """
space:
!:0,0!0,1!0,2!0,3!0,4!2,4
":0,0!1,0!0,1!1,1
#:0,2!1,2!2,2!3,2!0,4!1,4!2,4!3,4!1,0!1,1!1,5!1,6!2,0!2,1!2,5!2,6
$:2,0!0,1!1,1!2,1!3,1!2,2!0,3!1,3!2,3!3,3!2,4!2,5!2,6
%:0,0!1,0!0,1!4,3!3,4!4,4!1,1!3,3
&:3,0!4,0!3,1!2,2!1,1!0,2!0,3!1,3!2,3!3,3!4,4!3,5!2,6!1,5!0,4
':0,0!0,1!0,2
(:3,0!2,1!2,2!2,3!2,4!2,5!3,6
):0,0!1,1!1,2!1,3!1,4!1,5!0,6
+:1,2!2,2!3,2!2,0!2,1!2,3!2,4!2,5
,:1,5!1,6!0,6
-:1,3!2,3!3,3
.:1,5!2,5
/:3,0!3,1!2,2!2,3!1,4!1,5!0,6!2,1!1,5
0:1,0!2,0!3,0!0,1!4,1!0,2!2,2!4,2!0,3!4,3!0,4!4,4!1,5!2,5!3,5!3,2!1,4
1:1,0!2,0!2,1!2,2!2,3!2,4!2,5!0,2!1,2!3,2
2:0,0!1,0!2,0!3,0!4,1!4,2!3,3!2,4!1,5!0,5!1,5!0,6!1,6!2,6!3,6!4,6
3:0,0!1,0!2,0!3,0!4,1!3,2!2,2!4,3!4,4!0,4!1,5!2,5!3,5
4:3,0!3,1!3,2!3,3!3,4!3,5!2,2!1,2!0,2!0,3
5:0,0!1,0!2,0!3,0!4,0!0,1!0,2!1,2!2,2!3,2!4,3!4,4!0,4!1,5!2,5!3,5
6:3,0!2,1!1,2!0,3!0,4!0,5!1,5!2,5!3,5!4,4!4,3!3,2!2,2!1,2!0,2
7:0,0!1,0!2,0!3,0!4,0!4,1!3,2!2,3!2,4!1,5!1,6
8:1,0!2,0!3,0!0,1!4,1!0,2!2,2!4,2!0,3!4,3!0,4!4,4!1,5!2,5!3,5!3,2!1,4
9:1,0!2,0!3,0!4,1!4,2!4,3!3,4!2,5!1,5!0,4!0,3!0,2!1,2!2,2!3,2!4,5!3,6!2,6!1,6!0,5
::1,2!2,2!1,4!2,4
;:1,2!2,2!1,4!1,5!0,5
<:3,1!2,2!1,3!0,4!3,5!2,4!1,3!3,2!4,1
=:1,2!2,2!3,2!1,4!2,4!3,4
>:0,1!1,2!2,3!3,4!0,5!1,4!2,3!0,2
?:1,0!2,0!3,1!3,2!2,3!2,4!2,5
@:2,0!3,0!4,1!4,2!4,3!4,4!3,5!2,5!1,4!1,3!2,3!3,3!1,2!0,3!0,1!1,0
A:1,0!2,0!3,0!0,1!4,1!0,2!4,2!0,3!1,3!2,3!3,3!4,3!0,4!4,4!0,5!4,5
B:0,0!1,0!2,0!3,0!0,1!4,1!0,2!3,2!0,3!4,3!0,4!4,4!0,5!1,5!2,5!3,5
C:1,0!2,0!3,0!4,0!0,1!0,2!0,3!0,4!1,5!2,5!3,5!4,5
D:0,0!1,0!2,0!3,0!0,1!4,1!4,2!0,2!0,3!4,3!0,4!4,4!0,5!1,5!2,5!3,5
E:0,0!1,0!2,0!3,0!4,0!0,1!0,2!1,2!2,2!3,2!0,3!0,4!0,5!1,5!2,5!3,5!4,5
F:0,0!1,0!2,0!3,0!4,0!0,1!0,2!1,2!2,2!3,2!0,3!0,4!0,5
G:1,0!2,0!3,0!4,0!0,1!0,2!0,3!0,4!1,5!2,5!3,5!4,5!4,4!4,3!3,3!2,3
H:0,0!4,0!0,1!4,1!0,2!4,2!0,3!1,3!2,3!3,3!4,3!0,4!4,4!0,5!4,5
I:1,0!2,0!3,0!2,1!2,2!2,3!2,4!1,5!2,5!3,5
J:2,0!2,1!2,2!2,3!2,4!1,5!0,4!0,3
K:0,0!0,1!0,2!0,3!0,4!0,5!1,2!2,1!3,0!2,3!3,4!4,5
L:0,0!0,1!0,2!0,3!0,4!0,5!1,5!2,5!3,5!4,5
M:0,0!1,0!3,0!4,0!0,1!2,1!4,1!0,2!2,2!4,2!0,3!4,3!0,4!4,4!0,5!4,5
N:0,0!1,0!4,0!0,1!2,1!4,1!0,2!3,2!4,2!0,3!4,3!0,4!4,4!0,5!4,5
O:1,0!2,0!3,0!0,1!4,1!0,2!4,2!0,3!4,3!0,4!4,4!1,5!2,5!3,5
P:0,0!1,0!2,0!3,0!0,1!4,1!0,2!4,2!0,3!1,3!2,3!3,3!0,4!0,5
Q:1,0!2,0!3,0!0,1!4,1!0,2!4,2!0,3!4,3!1,4!2,4!3,4!4,5!4,6
R:0,0!1,0!2,0!3,0!0,1!4,1!0,2!4,2!0,3!1,3!2,3!3,3!0,4!2,4!3,5!4,5!4,6
S:1,0!2,0!3,0!4,0!0,1!0,2!1,2!2,2!3,2!4,3!4,4!0,4!1,5!2,5!3,5
T:0,0!1,0!2,0!3,0!4,0!2,1!2,2!2,3!2,4!2,5
U:0,0!0,1!0,2!0,3!0,4!4,0!4,1!4,2!4,3!4,4!1,5!2,5!3,5
V:0,0!0,1!0,2!0,3!1,4!2,5!3,4!4,3!4,2!4,1!4,0
W:0,0!0,1!0,2!0,3!0,4!0,5!2,2!2,3!4,0!4,1!4,2!4,3!4,4!4,5
X:0,0!1,1!2,2!3,3!4,4!4,0!3,1!1,3!0,4!2,2!3,5!4,6!1,5!0,6
Y:0,0!1,1!2,2!3,1!4,0!2,3!2,4!2,5
Z:0,0!1,0!2,0!3,0!4,0!3,1!2,2!1,3!0,4!0,5!1,5!2,5!3,5!4,5
[:2,0!3,0!2,1!2,2!2,3!2,4!2,5!2,6!3,6
\\:0,0!0,1!1,2!1,3!2,4!2,5!3,6
]:2,0!3,0!2,1!2,2!2,3!2,4!2,5!2,6!1,6
^:1,3!2,2!3,3
_:0,6!1,6!2,6!3,6!4,6
`:1,1!2,0
a:1,2!2,2!3,2!4,3!3,4!2,4!1,4!0,3!0,5!1,5!2,5!3,5!4,4!4,5
b:0,0!0,1!0,2!0,3!0,4!0,5!1,2!2,2!3,2!4,3!4,4!3,5!2,5!1,5
c:1,2!2,2!3,2!4,2!0,3!0,4!1,5!2,5!3,5!4,5
d:4,0!4,1!4,2!4,3!4,4!4,5!1,2!2,2!3,2!0,3!0,4!1,5!2,5!3,5
e:1,2!2,2!3,2!0,3!1,3!2,3!3,3!4,3!0,4!1,5!2,5!3,5
f:3,0!2,1!1,2!2,2!2,3!2,4!1,5!2,5!3,5!2,0
g:1,2!2,2!3,2!4,3!4,4!3,5!2,5!1,5!0,4!3,6!2,7!1,7!0,6
h:0,0!0,1!0,2!0,3!0,4!0,5!1,2!2,2!3,2!4,3!4,4!4,5
i:2,0!2,2!2,3!2,4!2,5!1,5!3,5
j:3,0!3,2!3,3!3,4!3,5!2,5!1,6!2,7!3,6
k:0,0!0,1!0,2!0,3!0,4!0,5!1,3!2,2!3,1!2,4!3,5!3,6
l:2,0!2,1!2,2!2,3!2,4!2,5!1,5!3,5
m:0,2!0,3!0,4!0,5!1,2!2,3!3,2!4,3!4,4!4,5
n:1,2!2,2!3,2!0,3!4,3!0,4!4,4!0,5!4,5
o:1,2!2,2!3,2!0,3!4,3!0,4!4,4!1,5!2,5!3,5
p:1,2!2,2!3,2!4,3!4,4!3,5!2,5!1,5!0,4!0,6!1,7!2,7
q:1,2!2,2!3,2!0,3!4,3!0,4!1,5!2,5!3,5!4,4!4,6!3,7!2,7
r:1,2!2,2!0,3!1,3!2,3!2,4!2,5
s:1,2!2,2!3,2!4,2!0,3!1,3!2,3!3,4!4,5!3,5!2,5!1,5!0,5
t:2,0!2,1!2,2!2,3!2,4!0,2!1,4!3,4!2,5
u:0,2!0,3!4,2!4,3!0,4!4,4!1,5!2,5!3,5
v:0,2!0,3!4,2!4,3!1,4!2,5!3,4
w:0,2!0,3!0,4!4,2!4,3!4,4!2,3!2,4!2,5!1,3!3,3
x:0,2!1,3!2,4!3,3!4,2!3,5!4,6!1,5!0,6
y:0,2!0,3!4,2!4,3!0,4!1,5!2,6!3,5!4,4!3,6!2,7!1,7
z:1,2!2,2!3,2!4,2!3,3!2,4!1,5!0,5!1,5!2,5!3,5!4,5
"""
_FONT: Dict[str, list] = {}
for _entry in _G.strip().splitlines():
    # 每个字形条目 = 单字符名 + ':' + 坐标串（坐标以 '!' 分隔、'x,y' 为一点）
    if len(_entry) < 2 or _entry[1] != ":":
        continue
    _name, _pts = _entry[0], _entry[2:]
    _FONT[_name if _name != "space" else " "] = [
        tuple(int(v) for v in p.split(",")) for p in _pts.split("!") if p]
