"""
cli.py
======

命令行版：不打开网页，直接合成 RGBA PNG，适合批量 / 脚本调用。

示例：
    # 用四张通道图合成，输出到 output/（自动命名）
    python cli.py --r examples/R.png --g examples/G.png --b examples/B.png --a examples/A.png

    # 只给 R/G，B 按 0（黑）填充、A 按 0（完全透明）填充
    python cli.py --r R.png --g G.png --size 1024x1024

    # 指定尺寸与文件名，Alpha 使用常量 255（完全不透明）
    python cli.py --r R.png --g G.png --b B.png --size 1024x1024 --alpha-const 255 -o out/tex.png

    # 保持比例裁剪，并反转 Alpha
    python cli.py --r R.png --g G.png --b B.png --a A.png --size 2048x2048 --mode crop --invert-alpha
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from image_processor import (
    HIGH_BITDEPTH_MODES,
    RESIZE_MODES,
    ChannelError,
    combine_to_png,
)


def parse_size(text: str):
    normalized = text.lower().replace(" ", "").replace("*", "x")
    if "x" not in normalized:
        raise argparse.ArgumentTypeError("尺寸格式应为 宽x高，例如 1024x1024")
    width_text, height_text = normalized.split("x", 1)
    try:
        width, height = int(width_text), int(height_text)
    except ValueError:
        raise argparse.ArgumentTypeError("尺寸必须是整数，例如 1024x1024")
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("尺寸必须大于 0")
    return width, height


def build_parser():
    parser = argparse.ArgumentParser(
        description="RGBA Image Composer CLI · 四通道合成 PNG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--r", default=None, help="R 通道图片路径；省略则该通道按 0（黑）填充")
    parser.add_argument("--g", default=None, help="G 通道图片路径；省略则该通道按 0（黑）填充")
    parser.add_argument("--b", default=None, help="B 通道图片路径；省略则该通道按 0（黑）填充")
    parser.add_argument(
        "--a", default=None,
        help="A 通道图片路径；省略则 Alpha 按 0（完全透明）填充",
    )
    parser.add_argument(
        "--alpha-const", type=int, default=None, metavar="0-255",
        help="使用常量 Alpha（设置后忽略 --a），例如 255 = 完全不透明",
    )
    parser.add_argument(
        "--size", type=parse_size, default=(1024, 1024), metavar="WxH",
        help="输出尺寸，默认 1024x1024",
    )
    parser.add_argument(
        "--mode", choices=RESIZE_MODES, default="stretch",
        help="缩放模式：stretch / fit / crop",
    )
    parser.add_argument("--invert-alpha", action="store_true", help="反转 Alpha 通道")
    parser.add_argument("--invert-rgb", action="store_true", help="反转 R/G/B 通道")
    parser.add_argument(
        "--depth", choices=HIGH_BITDEPTH_MODES, default="scale",
        help="16/32-bit 位深换算：scale=满量程（默认，16-bit ÷257） / normalize=按实际范围拉伸",
    )
    parser.add_argument(
        "--compress", type=int, default=6, metavar="0-9",
        help="PNG 压缩等级，默认 6（无损）",
    )
    parser.add_argument("-o", "--output", default=None, help="输出文件路径；省略则写入 output/ 自动命名")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    width, height = args.size

    try:
        saved = combine_to_png(
            output_path=args.output,
            output_dir="output",
            compress_level=args.compress,
            r_path=args.r,
            g_path=args.g,
            b_path=args.b,
            a_path=args.a,
            width=width,
            height=height,
            resize_mode=args.mode,
            invert_alpha=args.invert_alpha,
            invert_rgb=args.invert_rgb,
            alpha_value=args.alpha_const,
            high_bitdepth=args.depth,
        )
    except ChannelError as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 - CLI 顶层兜底
        print(f"[失败] {exc}", file=sys.stderr)
        return 1

    print(f"[完成] {width}x{height} RGBA PNG -> {saved}")
    print(f"[信息] 文件大小 {Path(saved).stat().st_size / 1024:.1f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
