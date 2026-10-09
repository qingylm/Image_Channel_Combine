"""
cli.py
======

命令行版：不打开网页，直接合成 RGBA PNG，适合批量 / 脚本调用。

示例：
    # 等大拼接（推荐）：四张 2048x2048 -> 一张 2048x2048 RGBA，尺寸自动取自输入
    python cli.py --r R.png --g G.png --b B.png --a A.png

    # 只给 R/G：B 按 0（黑）填充、A 按 0（完全透明）填充
    python cli.py --r R.png --g G.png

    # 手动指定尺寸与文件名，Alpha 使用常量 255（完全不透明）
    python cli.py --r R.png --g G.png --b B.png --size 1024x1024 --alpha-const 255 -o out/tex.png

    # 通道拆分：一张图 -> R/G/B/A 四个灰度 PNG + ZIP
    python cli.py --split tex.png --outdir channels/

    # 四宫格拼图：四张 2048x2048 -> 一张 2048x2048（每块 1024x1024，各自保留 RGBA）
    python cli.py --mosaic --tl a.png --tr b.png --bl c.png --br d.png

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
    mosaic_quadrants,
    output_file_name,
    pack_same_size,
    process_to_rgba,
    save_channel_png,
    save_png,
    split_channels,
    zip_files,
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
        "--size", type=parse_size, default=None, metavar="WxH",
        help="输出尺寸；省略则自动取第一张通道图的原始尺寸（等大拼接，1:1 零重采样）",
    )
    parser.add_argument(
        "--mode", choices=RESIZE_MODES, default="stretch",
        help="缩放模式：stretch / fit / crop（尺寸不一致时如何对齐）",
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

    # ---- 拆分模式 ----
    parser.add_argument(
        "--split", default=None, metavar="IMAGE",
        help="拆分模式：把该图片拆成 R/G/B/A 四个通道（此时忽略 --r/--g/--b/--a）",
    )
    parser.add_argument(
        "--outdir", default=None,
        help="拆分模式的输出目录；省略则写入 output/<图片名>_split/",
    )
    # ---- 四宫格拼图模式 ----
    parser.add_argument(
        "--mosaic", action="store_true",
        help="四宫格拼图模式：把 --tl/--tr/--bl/--br 四张图拼成一张 2×2 的新图",
    )
    parser.add_argument("--tl", default=None, help="拼图模式：左上图片")
    parser.add_argument("--tr", default=None, help="拼图模式：右上图片")
    parser.add_argument("--bl", default=None, help="拼图模式：左下图片")
    parser.add_argument("--br", default=None, help="拼图模式：右下图片")
    return parser


def run_mosaic(args) -> int:
    """四宫格拼图：四张图各自保留 RGBA -> 一张 2×2 新图。"""
    report: list[str] = []
    try:
        atlas, size = mosaic_quadrants(
            tl_path=args.tl, tr_path=args.tr, bl_path=args.bl, br_path=args.br,
            width=None if args.size is None else args.size[0],
            height=None if args.size is None else args.size[1],
            tile_mode=args.mode,
            high_bitdepth=args.depth,
            size_report=report,
        )
    except ChannelError as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        return 2

    output = args.output or Path("output") / output_file_name(size[0], size[1], prefix="mosaic")
    saved = save_png(atlas, output, compress_level=args.compress)

    print(f"[完成] 2x2 拼图 {size[0]}x{size[1]}（每块 {size[0] // 2}x{size[1] // 2}）-> {saved}")
    for line in report:
        print(f"       {line}")
    print(f"[信息] 文件大小 {Path(saved).stat().st_size / 1024:.1f} KB")
    return 0


def run_split(args) -> int:
    """拆分模式：一张图 -> 四个灰度通道 PNG + ZIP。"""
    try:
        r, g, b, a, info = split_channels(args.split, high_bitdepth=args.depth)
    except ChannelError as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        return 2

    stem = Path(info["name"]).stem
    out_dir = Path(args.outdir) if args.outdir else Path("output") / f"{stem}_split"
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = [
        save_channel_png(channel, out_dir / f"{stem}_{label}.png", compress_level=args.compress)
        for label, channel in zip("RGBA", (r, g, b, a), strict=True)
    ]
    zip_path = zip_files(paths, out_dir / f"{stem}_RGBA_channels.zip")

    print(f"[完成] 拆分 {info['name']}（{info['width']}x{info['height']} {info['mode']}）")
    print(f"[信息] A 通道来源：{info['alpha_source']}")
    for path in paths:
        print(f"       {Path(path).name}  {Path(path).stat().st_size / 1024:.1f} KB")
    print(f"[信息] 打包：{zip_path}")
    return 0


def main(argv=None):
    args = build_parser().parse_args(argv)

    if args.split:
        return run_split(args)

    if args.mosaic:
        return run_mosaic(args)

    auto_size = args.size is None
    try:
        if auto_size:
            # 等大拼接：输出尺寸取自第一张通道图
            *_, rgba, (width, height) = pack_same_size(
                r_path=args.r, g_path=args.g, b_path=args.b, a_path=args.a,
                width=None, height=None,
                mismatch_mode=args.mode,
                high_bitdepth=args.depth,
                size_report=None,
            )
            if args.alpha_const is not None:
                # 常量 Alpha 与自动尺寸不冲突：重新合成一次以保证语义一致
                rgba = process_to_rgba(
                    r_path=args.r, g_path=args.g, b_path=args.b, a_path=None,
                    width=width, height=height, resize_mode=args.mode,
                    invert_alpha=args.invert_alpha, invert_rgb=args.invert_rgb,
                    alpha_value=args.alpha_const, high_bitdepth=args.depth,
                )
            saved = save_png(rgba, args.output or Path("output") / output_file_name(width, height, prefix="packed"),
                             compress_level=args.compress)
        else:
            width, height = args.size
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

    mode_note = "（自动取输入尺寸）" if auto_size else ""
    print(f"[完成] {width}x{height} RGBA PNG{mode_note} -> {saved}")
    print(f"[信息] 文件大小 {Path(saved).stat().st_size / 1024:.1f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
