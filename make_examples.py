"""
make_examples.py
================

生成一组示例通道图片（examples/R.png、G.png、B.png、A.png），
用于第一次打开网页工具时直接测试。

运行：
    python make_examples.py
"""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image

SIZE = 512
EXAMPLES_DIR = Path(__file__).resolve().parent / "examples"


def build_channels(size: int = SIZE):
    r = Image.new("L", (size, size))
    g = Image.new("L", (size, size))
    b = Image.new("L", (size, size))
    a = Image.new("L", (size, size))

    r_px, g_px, b_px, a_px = r.load(), g.load(), b.load(), a.load()
    center = (size - 1) / 2

    for y in range(size):
        for x in range(size):
            u = x / (size - 1)
            v = y / (size - 1)
            distance = math.hypot(x - center, y - center) / center

            r_px[x, y] = int(255 * u)
            g_px[x, y] = int(255 * (1 - v))
            b_px[x, y] = int(255 * (0.5 + 0.5 * math.sin(u * math.pi * 4)) * v)
            # Alpha：圆形柔边遮罩，中心不透明、边缘透明
            a_px[x, y] = max(0, min(255, int(255 * (1.05 - distance))))

    return r, g, b, a


def main():
    EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    channels = build_channels()
    for name, image in zip(("R", "G", "B", "A"), channels):
        path = EXAMPLES_DIR / f"{name}.png"
        image.save(path, format="PNG")
        print(f"written: {path}")
    print(f"\n示例生成完成，共 4 张 {SIZE}×{SIZE} 灰度图，可直接在网页 UI 中上传。")


if __name__ == "__main__":
    main()
