"""
image_processor.py
==================

RGBA 通道合成核心逻辑（不依赖任何 UI 框架，可单独在脚本 / 批处理中调用）。

职责：
    1. 读取 R / G / B / A 四张通道图片并统一转换为 8-bit 灰度图（L）
    2. 按指定模式（拉伸 / 保持比例填充 / 保持比例裁剪）缩放到目标尺寸
    3. 未上传（为空）的通道自动用常量 0 填充，并支持通道反转
    4. 合并为 RGBA 并保存为 PNG（PNG 保留 Alpha 通道）

所有公开函数都使用关键字参数，避免参数错位。
"""

from __future__ import annotations

import contextlib
import shutil
import warnings
import zipfile
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageChops, ImageOps

# Lanczos 重采样：缩放质量最好，适合纹理 / 贴图
RESAMPLE = Image.Resampling.LANCZOS

# 缩放模式
RESIZE_MODES = ("stretch", "fit", "crop")

# 未上传（空）通道的填充值：0
#   R/G/B 为 0 → 该颜色分量为黑
#   A 为 0     → 完全透明
MISSING_CHANNEL_VALUE = 0

# 默认归档目录（<项目根>/output），app.py 与 ui_tools.py 共用
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "output"

RESIZE_MODE_LABELS = {
    "stretch": "Stretch · 直接拉伸",
    "fit": "Fit · 保持比例 + 黑边填充",
    "crop": "Crop · 保持比例 + 居中裁剪",
}

# 单个通道允许的输入格式
SUPPORTED_SUFFIXES = (
    ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff",
    ".webp", ".gif", ".ppm", ".pgm",
)


class ChannelError(ValueError):
    """通道图片读取 / 参数错误。"""


# ---------------------------------------------------------------------------
# 位深处理（8-bit / 16-bit / 32-bit / float）
# ---------------------------------------------------------------------------

# Pillow 的 convert("L") 对 I;16 / I / F 是**裁剪**语义（>255 的值直接变 255），
# 会把 16-bit 灰度贴图整张压成纯白。这里给出正确的两种换算方式：
#   scale     - 按位深满量程换算：16-bit ÷257（65535 -> 255），保留绝对数值
#   normalize - 按图像实际范围 min–max 拉伸到 0–255，适合只用了部分量程的蒙版
HIGH_BITDEPTH_MODES = ("scale", "normalize")

HIGH_BITDEPTH_LABELS = {
    "scale": "满量程换算（16-bit ÷257，保留原值）",
    "normalize": "实际范围拉伸（min–max → 0–255）",
}

_INTEGER_MODES = ("I;16", "I;16B", "I;16L", "I;16N", "I")

_LUT16_SCALE: list[int] | None = None


def _scale_lut_16() -> list[int]:
    """16-bit -> 8-bit 的满量程换算表（惰性构建一次）。"""
    global _LUT16_SCALE
    if _LUT16_SCALE is None:
        _LUT16_SCALE = [min(255, value >> 8) for value in range(65536)]
    return _LUT16_SCALE


def _integer_to_l(image: Image.Image, high_bitdepth: str) -> Image.Image:
    """I;16 / I 模式（16-bit、32-bit 整数）-> 8-bit 灰度。"""
    int_image = image if image.mode == "I" else image.convert("I")
    low, high = int_image.getextrema()

    if high <= low:  # 常量图
        return Image.new("L", int_image.size, max(0, min(255, high >> 8)))

    if high <= 65535:
        if high_bitdepth == "normalize":
            scale = 255.0 / (high - low)
            lut = [
                max(0, min(255, round((value - low) * scale)))
                for value in range(65536)
            ]
        else:
            lut = _scale_lut_16()
        return int_image.point(lut, "L")

    # 32-bit 整数（值域超出 16-bit，如部分 TIFF）：交给 numpy 缩放
    return _numpy_to_l(int_image, low, high, high_bitdepth)


def _float_to_l(image: Image.Image, high_bitdepth: str) -> Image.Image:
    """F 模式（32-bit 浮点）-> 8-bit 灰度。

    0.0–1.0 的浮点图按 ×255 换算；超出该范围或选择 normalize 时按实际范围拉伸。
    """
    low, high = image.getextrema()

    if high <= low:  # 常量图
        return Image.new("L", image.size, max(0, min(255, round(high * 255))))

    if high_bitdepth == "normalize" or high > 1.0 or low < 0.0:
        scale = 255.0 / (high - low)
        offset = -low * scale
    else:
        scale, offset = 255.0, 0.0

    # F 模式的 point() 只接受线性表达式（v * a + b），随后 convert("L") 会裁剪到 0–255
    return image.point(lambda v: v * scale + offset).convert("L")


def _numpy_to_l(image, low, high, high_bitdepth) -> Image.Image:
    """32-bit 整数图的兜底换算（需要 numpy；gradio 环境一定有）。"""
    try:
        import numpy as np
    except ImportError:  # pragma: no cover - 仅在极简环境触发
        warnings.warn(
            "未安装 numpy，32-bit 图像将退回 Pillow 的裁剪式转换（可能丢失细节）",
            stacklevel=2,
        )
        return image.convert("L")

    array = np.asarray(image).astype("float64")
    if high_bitdepth == "normalize":
        array = (array - low) * (255.0 / (high - low))
    else:
        array = array * (255.0 / high)
    return Image.fromarray(np.clip(array, 0, 255).astype("uint8"))


def to_grayscale_8bit(image: Image.Image, high_bitdepth: str = "scale") -> Image.Image:
    """把任意位深的图片转成 8-bit 灰度，**不会把 16-bit 数据截断成纯白**。

    Args:
        image: 任意模式的 PIL 图像。
        high_bitdepth: "scale"（满量程换算，默认）或 "normalize"（实际范围拉伸）。

    Returns:
        PIL.Image.Image：模式为 "L" 的灰度图。
    """
    if high_bitdepth not in HIGH_BITDEPTH_MODES:
        raise ChannelError(
            f"未知高位深换算方式：{high_bitdepth}（可选：{', '.join(HIGH_BITDEPTH_MODES)}）"
        )

    mode = image.mode

    if mode in _INTEGER_MODES:
        return _integer_to_l(image, high_bitdepth)

    if mode == "F":
        return _float_to_l(image, high_bitdepth)

    if mode == "L":
        # 返回副本：调用方随后可能关闭源图，返回别名会变成 "Operation on closed image"
        return image.copy()

    # 8-bit 及以下：RGB / RGBA / P / LA / CMYK / 1 … 直接用亮度转灰度
    return image.convert("L")


# ---------------------------------------------------------------------------
# 基础操作
# ---------------------------------------------------------------------------

def _validate_path(file_path) -> None:
    """路径/格式基础校验（文件不存在、是目录、后缀不支持）。"""
    if not file_path:
        raise ChannelError("没有提供通道图片")

    if isinstance(file_path, (str, Path)):
        path = Path(file_path)
        if not path.exists():
            raise ChannelError(f"文件不存在：{path}")
        if path.is_dir():
            raise ChannelError(f"这是一个目录，不是图片：{path}")
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            raise ChannelError(f"不支持的图片格式：{path.suffix}（{path.name}）")


def _open_oriented(file_path) -> Image.Image:
    """打开图片并应用 EXIF 方向，返回已 load 的图像（调用方负责 close 语义）。"""
    _validate_path(file_path)
    try:
        image = Image.open(file_path)
        # JPEG 等带 EXIF 方向的图片：与浏览器/看图软件看到的方向保持一致
        if image.getexif().get(274, 1) != 1:
            transposed = None
            with contextlib.suppress(Exception):
                transposed = ImageOps.exif_transpose(image)
            if transposed is not None:
                image = transposed
        image.load()
    except ChannelError:
        raise
    except Exception as exc:  # Pillow 的解码异常种类很多，统一成可读错误
        raise ChannelError(f"无法读取图片 {file_path}：{exc}") from exc
    return image


def _read_channel(file_path, *, invert: bool = False, high_bitdepth: str = "scale"):
    """读取图片 -> 8-bit 灰度，返回 (channel, 源模式说明)。"""
    image = _open_oriented(file_path)
    try:
        source_mode = image.mode
        channel = to_grayscale_8bit(image, high_bitdepth)
        channel.load()
    finally:
        image.close()

    if invert:
        channel = invert_channel(channel)

    note = f"{source_mode}->{channel.mode}"
    if source_mode in _INTEGER_MODES or source_mode == "F":
        note += f"({high_bitdepth})"
    return channel, note


def load_channel(
    file_path,
    invert: bool = False,
    high_bitdepth: str = "scale",
) -> Image.Image:
    """读取一张图片并转换为 8-bit 灰度通道（L 模式）。

    16-bit / 32-bit / 浮点图会先按 ``high_bitdepth`` 正确换算到 8-bit，
    而不是被 Pillow 的 ``convert("L")`` 截断成纯白。

    Args:
        file_path: 图片路径（str / Path / 类文件对象均可被 Pillow 接受）。
        invert: 是否对灰度值取反（x -> 255 - x）。
        high_bitdepth: "scale"（满量程换算，默认）或 "normalize"（实际范围拉伸）。

    Returns:
        PIL.Image.Image：模式为 "L" 的灰度图。
    """
    return _read_channel(file_path, invert=invert, high_bitdepth=high_bitdepth)[0]


def invert_channel(channel: Image.Image) -> Image.Image:
    """灰度取反：0 -> 255，255 -> 0。"""
    return ImageChops.invert(channel.convert("L"))


def make_constant_channel(width: int, height: int, value: int = 255) -> Image.Image:
    """生成一张纯色灰度通道，常用于「默认 Alpha = 255（完全不透明）」。"""
    value = max(0, min(255, int(value)))
    return Image.new("L", (int(width), int(height)), value)


def resize_channel(
    channel: Image.Image,
    width: int,
    height: int,
    mode: str = "stretch",
    background: int = 0,
) -> Image.Image:
    """把通道缩放到目标尺寸。

    Args:
        channel: 灰度通道。
        width / height: 目标宽高（像素，必须 > 0）。
        mode:
            stretch - 直接拉伸到目标尺寸（可能变形）
            fit     - 保持比例完整放入，空白用 background 填充（黑边）
            crop    - 保持比例铺满，居中裁剪多余部分
        background: fit 模式的填充值（0 = 纯黑 = 全透明）。

    Returns:
        PIL.Image.Image：尺寸严格等于 (width, height) 的 "L" 图。
    """
    width = int(width)
    height = int(height)

    if width <= 0 or height <= 0:
        raise ChannelError("输出尺寸必须大于 0")

    if mode not in RESIZE_MODES:
        raise ChannelError(f"未知缩放模式：{mode}（可选：{', '.join(RESIZE_MODES)}）")

    channel = channel.convert("L")

    if mode == "stretch":
        if channel.size == (width, height):
            return channel
        return channel.resize((width, height), RESAMPLE)

    src_w, src_h = channel.size
    if src_w <= 0 or src_h <= 0:
        raise ChannelError("输入通道尺寸无效")

    target_ratio = width / height
    source_ratio = src_w / src_h

    if mode == "fit":
        # 完整放入目标框，短边对齐
        if source_ratio > target_ratio:
            new_w, new_h = width, max(1, round(width / source_ratio))
        else:
            new_h, new_w = height, max(1, round(height * source_ratio))

        resized = channel.resize((new_w, new_h), RESAMPLE)
        canvas = Image.new("L", (width, height), int(background))
        canvas.paste(resized, ((width - new_w) // 2, (height - new_h) // 2))
        return canvas

    # mode == "crop"：铺满目标框，多余部分居中裁掉
    if source_ratio > target_ratio:
        new_h, new_w = height, max(1, round(height * source_ratio))
    else:
        new_w, new_h = width, max(1, round(width / source_ratio))

    resized = channel.resize((new_w, new_h), RESAMPLE)
    left = (new_w - width) // 2
    top = (new_h - height) // 2
    return resized.crop((left, top, left + width, top + height))


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def resolve_channel(
    file_path,
    *,
    width: int,
    height: int,
    resize_mode: str = "stretch",
    invert: bool = False,
    missing_value: int = MISSING_CHANNEL_VALUE,
    high_bitdepth: str = "scale",
    label: str = "",
    source_notes: list[str] | None = None,
) -> tuple[Image.Image, bool]:
    """取得一个通道：有图就读图+缩放，为空就用 missing_value 常量填充。

    Returns:
        (channel, filled)：channel 为 "L" 灰度图；filled=True 表示该通道原本为空，
        已按 missing_value 填充。
    """
    if not file_path:
        channel = make_constant_channel(width, height, missing_value)
        if invert:
            channel = invert_channel(channel)
        return channel, True

    raw, note = _read_channel(
        file_path, invert=invert, high_bitdepth=high_bitdepth
    )
    if source_notes is not None:
        source_notes.append(f"{label}: {note}" if label else note)
    return resize_channel(raw, width, height, resize_mode), False


def process_channels(
    *,
    r_path=None,
    g_path=None,
    b_path=None,
    a_path=None,
    width: int = 1024,
    height: int = 1024,
    resize_mode: str = "stretch",
    invert_alpha: bool = False,
    invert_rgb: bool = False,
    alpha_value: int | None = None,
    missing_value: int = MISSING_CHANNEL_VALUE,
    high_bitdepth: str = "scale",
    filled_channels: list[str] | None = None,
    source_notes: list[str] | None = None,
) -> tuple[Image.Image, Image.Image, Image.Image, Image.Image, Image.Image]:
    """读取四个通道 -> 缩放 -> 合成 RGBA。

    **空通道按 0 填充**：任何一个通道没有提供图片（路径为空 / None）时，不再报错，
    而是用常量 ``missing_value``（默认 0）铺满目标尺寸：

        * R/G/B 为空 -> 该颜色分量为 0（黑）
        * A 为空     -> 完全透明（alpha = 0）

    **位深**：16-bit / 32-bit / 浮点输入会按 ``high_bitdepth`` 正确换算成 8-bit，
    不会被 Pillow 的 ``convert("L")`` 截断成纯白。

    只有「提供了路径但读不出来（文件不存在 / 格式错误）」才抛 ChannelError。

    Args:
        r_path / g_path / b_path / a_path: 三个颜色通道与 Alpha 通道的图片路径，
            为空表示该通道不提供图片，按 missing_value 填充。
        width / height: 输出尺寸。
        resize_mode: "stretch" | "fit" | "crop"。
        invert_alpha: 是否反转 Alpha（0 <-> 255）。
        invert_rgb: 是否反转 R/G/B 三个通道。
        alpha_value: 若不为 None，则忽略 a_path，直接用该常量作为 Alpha
                     （例如 255 = 完全不透明）。取值范围 0~255。
        missing_value: 空通道的填充值，默认 0。
        high_bitdepth: 高位深图的换算方式，"scale"（满量程，默认）或 "normalize"。
        filled_channels: 可选。传入一个 list 时，会把「因空而填充」的通道名
                     （"R" / "G" / "B" / "A"）追加进去，便于 UI 提示。
        source_notes: 可选。传入一个 list 时，会追加每个通道的源模式说明
                     （例如 "I;16->L(scale)"），便于确认位深是否被正确处理。

    Returns:
        (r, g, b, a, rgba)：五个 PIL 图像，前四个为 "L"，最后为 "RGBA"。
    """
    width = int(width)
    height = int(height)

    if width <= 0 or height <= 0:
        raise ChannelError("输出尺寸必须大于 0")

    if width > 16384 or height > 16384:
        raise ChannelError("输出尺寸过大（单边上限 16384）")

    if high_bitdepth not in HIGH_BITDEPTH_MODES:
        raise ChannelError(
            f"未知高位深换算方式：{high_bitdepth}（可选：{', '.join(HIGH_BITDEPTH_MODES)}）"
        )

    missing_value = max(0, min(255, int(missing_value)))

    def take(path, label, invert):
        channel, filled = resolve_channel(
            path,
            width=width,
            height=height,
            resize_mode=resize_mode,
            invert=invert,
            missing_value=missing_value,
            high_bitdepth=high_bitdepth,
            label=label,
            source_notes=source_notes,
        )
        if filled and filled_channels is not None:
            filled_channels.append(label)
        return channel

    r = take(r_path, "R", invert_rgb)
    g = take(g_path, "G", invert_rgb)
    b = take(b_path, "B", invert_rgb)

    if alpha_value is None:
        a = take(a_path, "A", invert_alpha)
    else:
        # 常量 Alpha 与 invert_alpha 组合时，同样遵守反转语义
        value = max(0, min(255, int(alpha_value)))
        if invert_alpha:
            value = 255 - value
        a = make_constant_channel(width, height, value)

    rgba = Image.merge("RGBA", (r, g, b, a))
    return r, g, b, a, rgba


def process_to_rgba(**kwargs) -> Image.Image:
    """process_channels 的简化入口，只返回最终的 RGBA 图。"""
    return process_channels(**kwargs)[-1]


# ---------------------------------------------------------------------------
# 工具 A：通道拆分（一张图片 -> R / G / B / A 四个通道）
# ---------------------------------------------------------------------------

def probe_image_info(file_path) -> dict:
    """只读文件头，拿到源图的尺寸 / 模式 / 是否自带 Alpha，不修改任何像素。"""
    _validate_path(file_path)
    image = Image.open(file_path)
    try:
        mode = image.mode
        size = image.size
        has_alpha = "A" in image.getbands() or (
            mode == "P" and "transparency" in image.info
        )
        return {
            "path": str(file_path),
            "name": Path(str(file_path)).name,
            "mode": mode,
            "size": size,
            "width": size[0],
            "height": size[1],
            "has_alpha": has_alpha,
            "is_square": size[0] == size[1],
            "bit_depth": 16 if mode in _INTEGER_MODES else (32 if mode == "F" else 8),
        }
    finally:
        image.close()


def split_channels(
    file_path,
    *,
    high_bitdepth: str = "scale",
) -> tuple[Image.Image, Image.Image, Image.Image, Image.Image, dict]:
    """把一张图片拆成 R / G / B / A 四个 8-bit 灰度通道。

    规则：
        * 彩色 / 带 Alpha 的图：直接取各自的 R、G、B、A 分量
        * 没有 Alpha 的图（RGB / L / P）：A 记为 255（完全不透明）
        * 16-bit / 32-bit / 浮点灰度图：先按 ``high_bitdepth`` 换算成 8-bit，
          再作为 R = G = B 使用，A 记 255

    Returns:
        (r, g, b, a, info)：四个 "L" 通道 + 源图信息字典
        （含 name / mode / size / has_alpha / bit_depth / alpha_source）。
    """
    info = probe_image_info(file_path)
    image = _open_oriented(file_path)
    try:
        size = image.size
        if image.mode in _INTEGER_MODES or image.mode == "F":
            grey = to_grayscale_8bit(image, high_bitdepth)
            r = grey.copy()
            g = grey.copy()
            b = grey.copy()
            a = Image.new("L", size, 255)
            alpha_source = "无（高位深灰度图，按 255 不透明合成）"
        else:
            rgba = image.convert("RGBA")
            r, g, b, a = rgba.split()
            alpha_source = "自带" if info["has_alpha"] else "无（按 255 不透明合成）"
    finally:
        image.close()

    info["alpha_source"] = alpha_source
    info["high_bitdepth"] = high_bitdepth
    return r, g, b, a, info


# ---------------------------------------------------------------------------
# 工具 B：等大拼接（四张同尺寸图 -> 一张同尺寸 RGBA，通道打包）
# ---------------------------------------------------------------------------

def pack_same_size(
    *,
    r_path=None,
    g_path=None,
    b_path=None,
    a_path=None,
    width: int | None = None,
    height: int | None = None,
    mismatch_mode: str = "stretch",
    high_bitdepth: str = "scale",
    missing_value: int = MISSING_CHANNEL_VALUE,
    filled_channels: list[str] | None = None,
    source_notes: list[str] | None = None,
    size_report: list[str] | None = None,
) -> tuple[Image.Image, Image.Image, Image.Image, Image.Image, Image.Image, tuple[int, int]]:
    """四张同尺寸正方形图 -> 一张**同尺寸** RGBA（1:1 通道打包）。

    与 ``process_channels`` 的区别：输出尺寸不手填，默认自动取「第一个上传通道的原始尺寸」，
    因此正常情况下四个通道都是**零重采样**直接搬进 R/G/B/A。

    Args:
        r_path / g_path / b_path / a_path: 四个通道图；可为空（按 missing_value 填充）。
        width / height: 显式指定输出尺寸；为 None 时自动取第一张图的原始尺寸。
        mismatch_mode: 尺寸不一致时如何对齐（"stretch" / "fit" / "crop"）。
        high_bitdepth: 16/32-bit 位深换算方式。
        missing_value: 空通道填充值，默认 0。
        filled_channels: 可选，追加「因空而填充」的通道名。
        source_notes: 可选，追加各通道源位深说明。
        size_report: 可选，追加每张图的尺寸与是否需要重采样。

    Returns:
        (r, g, b, a, rgba, (width, height))
    """
    provided = [
        (label, path)
        for label, path in (("R", r_path), ("G", g_path), ("B", b_path), ("A", a_path))
        if path
    ]
    if not provided:
        raise ChannelError("至少上传一张通道图")

    if width is None or height is None:
        base_label, base_path = provided[0]
        info = probe_image_info(base_path)
        width, height = info["size"]
        if size_report is not None:
            size_report.append(f"基准尺寸取自 {base_label}：{width} × {height} px")

    width, height = int(width), int(height)

    if size_report is not None:
        for label, path in provided:
            try:
                info = probe_image_info(path)
            except Exception as exc:  # noqa: BLE001 - 探测失败交给后续读取报错
                size_report.append(f"{label}: 尺寸探测失败（{exc}）")
                continue
            same = info["size"] == (width, height)
            mark = "原始尺寸一致，1:1 直接打包" if same else f"将被{mismatch_mode}到 {width}×{height}"
            size_report.append(f"{label}: {info['width']}×{info['height']} {info['mode']} · {mark}")

    channels = process_channels(
        r_path=r_path,
        g_path=g_path,
        b_path=b_path,
        a_path=a_path,
        width=width,
        height=height,
        resize_mode=mismatch_mode,
        alpha_value=None,
        missing_value=missing_value,
        high_bitdepth=high_bitdepth,
        filled_channels=filled_channels,
        source_notes=source_notes,
    )
    return (*channels, (width, height))


# ---------------------------------------------------------------------------
# 工具 C：四宫格拼图（四张图各自保留 RGBA -> 一张新图，四角摆放）
# ---------------------------------------------------------------------------

# 每块的适配方式（复用缩放模式语义）：stretch 拉伸 / fit 保持比例+透明填充 / crop 保持比例裁剪
TILE_POSITIONS = ("tl", "tr", "bl", "br")

TILE_POSITION_LABELS = {
    "tl": "左上",
    "tr": "右上",
    "bl": "左下",
    "br": "右下",
}


def _fit_rgba_into(
    image: Image.Image,
    width: int,
    height: int,
    mode: str = "stretch",
) -> Image.Image:
    """把一张 RGBA 图适配到 width×height 的格子（透明背景）。

    注：Pillow 的 ``resize()`` 对 RGBA 已按**预乘 Alpha** 处理（本机 11.3 实测：
    透明边缘不会混入黑边），因此这里直接缩放，不做手工预乘。
    """
    image = image.convert("RGBA")
    if image.size == (width, height):
        return image.copy()

    if mode == "stretch":
        return image.resize((width, height), RESAMPLE)

    src_w, src_h = image.size
    if src_w <= 0 or src_h <= 0:
        raise ChannelError("输入图片尺寸无效")

    target_ratio = width / height
    source_ratio = src_w / src_h

    if mode == "fit":
        if source_ratio > target_ratio:
            new_w, new_h = width, max(1, round(width / source_ratio))
        else:
            new_h, new_w = height, max(1, round(height * source_ratio))

        resized = image.resize((new_w, new_h), RESAMPLE)
        canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        canvas.paste(resized, ((width - new_w) // 2, (height - new_h) // 2))
        return canvas

    if mode == "crop":
        if source_ratio > target_ratio:
            new_h, new_w = height, max(1, round(height * source_ratio))
        else:
            new_w, new_h = width, max(1, round(width / source_ratio))

        resized = image.resize((new_w, new_h), RESAMPLE)
        left = (new_w - width) // 2
        top = (new_h - height) // 2
        return resized.crop((left, top, left + width, top + height))

    raise ChannelError(f"未知的每块适配方式：{mode}（可选：{', '.join(RESIZE_MODES)}）")


def _load_rgba(file_path, high_bitdepth: str = "scale") -> Image.Image:
    """读取任意位深的图片并转成 RGBA（保留原有颜色与透明度）。

    注意：16-bit / 32-bit / 浮点灰度图必须先走 ``to_grayscale_8bit`` 正确换算，
    否则 Pillow 的 ``convert("RGBA")`` 会把 >255 的值截断成 255（整张纯白）。
    """
    image = _open_oriented(file_path)
    try:
        if image.mode in _INTEGER_MODES or image.mode == "F":
            grey = to_grayscale_8bit(image, high_bitdepth)
            return Image.merge(
                "RGBA",
                (grey.copy(), grey.copy(), grey.copy(), Image.new("L", grey.size, 255)),
            )
        return image.convert("RGBA")
    finally:
        image.close()


def mosaic_quadrants(
    *,
    tl_path=None,
    tr_path=None,
    bl_path=None,
    br_path=None,
    width: int | None = None,
    height: int | None = None,
    tile_mode: str = "stretch",
    high_bitdepth: str = "scale",
    size_report: list[str] | None = None,
) -> tuple[Image.Image, tuple[int, int]]:
    """四张图 -> 一张新图，分别摆进左上 / 右上 / 左下 / 右下四宫格。

    **每张输入图各自保留自己的 RGBA 通道**（颜色与透明度都保留），不是拆成四个通道。
    每块格子尺寸 = 输出尺寸的一半：

        四张 2048×2048  --(每张缩放到 1024×1024)-->  一张 2048×2048
        ┌─────────┬─────────┐
        │  左上    │  右上    │      每块 1024×1024
        ├─────────┼─────────┤
        │  左下    │  右下    │
        └─────────┴─────────┘

    Args:
        tl_path / tr_path / bl_path / br_path: 四张图片；为空的格子保持透明。
        width / height: 输出尺寸；为 None 时自动取「第一张提供的图片」的原始尺寸
            （于是四张 2048×2048 得到 2048×2048 输出、每块 1024×1024）。
        tile_mode: 每块如何适配格子，"stretch" | "fit"（保持比例 + 透明填充）| "crop"。
        high_bitdepth: 16/32-bit 位深换算方式（备用，RGBA 化后通常用不到）。
        size_report: 可选，追加每张图的源尺寸与放置说明。

    Returns:
        (atlas, (width, height))：合成后的 RGBA 图与输出尺寸。
    """
    if tile_mode not in RESIZE_MODES:
        raise ChannelError(
            f"未知的每块适配方式：{tile_mode}（可选：{', '.join(RESIZE_MODES)}）"
        )

    sources = [
        ("tl", TILE_POSITION_LABELS["tl"], tl_path),
        ("tr", TILE_POSITION_LABELS["tr"], tr_path),
        ("bl", TILE_POSITION_LABELS["bl"], bl_path),
        ("br", TILE_POSITION_LABELS["br"], br_path),
    ]
    provided = [(key, label, path) for key, label, path in sources if path]
    if not provided:
        raise ChannelError("至少上传一张图片")

    if width is None or height is None:
        # 自动：输出尺寸 = 第一张图的原始尺寸（四张 2048 → 输出 2048、每块 1024）
        info = probe_image_info(provided[0][2])
        width, height = info["size"]

    width, height = int(width), int(height)
    if width <= 0 or height <= 0:
        raise ChannelError("输出尺寸必须大于 0")
    if width > 16384 or height > 16384:
        raise ChannelError("输出尺寸过大（单边上限 16384）")

    tile_w, tile_h = width // 2, height // 2
    if tile_w < 1 or tile_h < 1:
        raise ChannelError("输出尺寸太小：每块至少需要 1×1 像素")

    offsets = {
        "tl": (0, 0),
        "tr": (tile_w, 0),
        "bl": (0, tile_h),
        "br": (tile_w, tile_h),
    }

    atlas = Image.new("RGBA", (width, height), (0, 0, 0, 0))

    for key, label, path in sources:
        x, y = offsets[key]
        if not path:
            if size_report is not None:
                size_report.append(f"{label}: 空 · 保持透明")
            continue

        image = _open_oriented(path)
        try:
            source_size = image.size
            source_mode = image.mode
        finally:
            image.close()

        source_rgba = _load_rgba(path, high_bitdepth)
        tile = _fit_rgba_into(source_rgba, tile_w, tile_h, tile_mode)

        atlas.paste(tile, (x, y))
        if size_report is not None:
            note = "1:1 放置" if source_size == (tile_w, tile_h) else f"缩放到 {tile_w}×{tile_h}"
            size_report.append(
                f"{label}: {source_size[0]}×{source_size[1]} {source_mode} → "
                f"{tile_w}×{tile_h}（{tile_mode} · {note}）"
            )

    if size_report is not None:
        size_report.append(f"输出: {width}×{height} · 每块 {tile_w}×{tile_h}（各自保留 RGBA）")
        if width % 2 or height % 2:
            size_report.append("提示: 输出宽/高为奇数，最右/最下一列像素保持透明，建议用偶数尺寸")

    return atlas, (width, height)


def _split_dirs(base: Path) -> list[Path]:
    """归档目录下形如 <名字>_split/ 的拆分输出目录，按修改时间倒序（新的在前）。"""
    if not base.exists():
        return []
    dirs = [
        path for path in base.iterdir()
        if path.is_dir() and path.name.endswith("_split")
    ]
    dirs.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return dirs


def latest_split_dir(output_dir=None) -> Path | None:
    """返回最新的一次拆分输出目录（即③拆分工具当前那套），没有则 None。"""
    base = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
    dirs = _split_dirs(base)
    return dirs[0] if dirs else None


def list_output_images(
    output_dir=None,
    limit: int = 200,
    scope: str = "latest_split",
) -> list[dict]:
    """列出可复用的图片（「通道池」用）。

    Args:
        output_dir: 归档目录，默认 ``<项目根>/output``。
        limit: 最多返回多少条。
        scope:
            "latest_split"（默认）—— **只列最新一次拆分的那一套通道**
            （``output/<名字>_split/`` 里的 R/G/B/A），历史拆分条目不再出现在列表里；
            "all" —— 列出 ``output/`` 下全部图片（含历史拆分与打包 / 拼图结果），按时间倒序。

    Returns:
        每项为 dict：path / name / label / mtime / size / kind
        （kind 为 "channel" 表示 ``*_R.png`` 这类单通道图，否则 "image"）。
    """
    base = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
    if not base.exists():
        return []

    if scope == "latest_split":
        newest = latest_split_dir(base)
        if newest is None:
            return []
        candidates = [path for path in newest.iterdir() if path.is_file()]
        # 固定按 R / G / B / A 顺序，便于下拉里一一对应
        order = {"R": 0, "G": 1, "B": 2, "A": 3}

        def rank(path: Path) -> tuple[int, str]:
            stem = path.stem
            slot = stem[-1].upper() if len(stem) > 1 else ""
            return (order.get(slot, 9), path.name)

        candidates = [
            path for path in candidates
            if path.suffix.lower() in SUPPORTED_SUFFIXES
        ]
        candidates.sort(key=rank)
    else:
        candidates = list(base.rglob("*"))

    entries = []
    for path in candidates:
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        try:
            stat = path.stat()
        except OSError:  # pragma: no cover - 文件刚好被删掉
            continue
        stem = path.stem
        is_channel = (
            path.parent.name.endswith("_split")
            and len(stem) > 2
            and stem[-2] == "_"
            and stem[-1].upper() in ("R", "G", "B", "A")
        )
        entries.append({
            "path": str(path),
            "name": path.name,
            "label": str(path.relative_to(base)).replace("\\", "/"),
            "mtime": stat.st_mtime,
            "size": stat.st_size,
            "kind": "channel" if is_channel else "image",
        })

    if scope != "latest_split":
        entries.sort(key=lambda entry: entry["mtime"], reverse=True)
    return entries[:limit]


def prune_split_dirs(output_dir=None, keep: int = 1) -> dict:
    """删除历史的拆分输出目录，只保留最新 ``keep`` 个（默认 1 = 拆分工具当前那套）。

    Returns:
        dict：{"kept": 目录名或 None, "removed": [目录名…], "freed": 释放字节数}
    """
    base = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
    dirs = _split_dirs(base)
    removed: list[str] = []
    freed = 0

    for directory in dirs[max(0, int(keep)):]:
        try:
            freed += sum(
                file.stat().st_size
                for file in directory.rglob("*")
                if file.is_file()
            )
        except OSError:  # pragma: no cover
            pass
        shutil.rmtree(directory, ignore_errors=True)
        removed.append(directory.name)

    return {
        "kept": dirs[0].name if dirs else None,
        "removed": removed,
        "freed": freed,
    }


def zip_files(files, zip_path) -> str:
    """把若干文件打包成 ZIP，返回 ZIP 路径。"""
    zip_path = Path(zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file in files:
            file = Path(file)
            if file.exists():
                archive.write(file, arcname=file.name)
    return str(zip_path)


def save_png(
    image: Image.Image,
    output_path,
    *,
    compress_level: int = 6,
) -> str:
    """保存为 PNG（无损，保留 Alpha 通道）。

    Args:
        image: 任意模式图像，内部会转换为 RGBA 以保证输出带 Alpha。
        output_path: 输出路径，父目录会自动创建。
        compress_level: zlib 压缩等级 0~9（0 最快 / 9 最小）。

    Returns:
        str：最终写入的绝对路径。
    """
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    compress_level = max(0, min(9, int(compress_level)))
    image.convert("RGBA").save(
        output_path,
        format="PNG",
        compress_level=compress_level,
        optimize=False,
    )
    return str(output_path)


def save_channel_png(image: Image.Image, output_path, *, compress_level: int = 6) -> str:
    """按**灰度单通道**保存 PNG（不转 RGBA），用于通道拆分导出。

    灰度 PNG 体积更小，且作为遮罩 / 蒙版导入 Unity、Substance、Photoshop 时语义正确。
    """
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    compress_level = max(0, min(9, int(compress_level)))
    image.convert("L").save(
        output_path,
        format="PNG",
        compress_level=compress_level,
        optimize=False,
    )
    return str(output_path)


def output_file_name(width: int, height: int, prefix: str = "rgba") -> str:
    """生成形如 rgba_1024x1024_20250101-120000.png 的文件名。"""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")  # noqa: DTZ005 - 文件名用本地时间更直观
    return f"{prefix}_{int(width)}x{int(height)}_{stamp}.png"


def combine_to_png(
    *,
    output_path=None,
    output_dir: str = "output",
    compress_level: int = 6,
    **process_kwargs,
) -> str:
    """一步到位：合成并写盘（适合命令行 / 批处理）。

    未指定 output_path 时，自动按 output_dir + 时间戳命名。
    """
    rgba = process_to_rgba(**process_kwargs)

    if output_path is None:
        output_path = Path(output_dir) / output_file_name(
            process_kwargs.get("width", 0) or rgba.width,
            process_kwargs.get("height", 0) or rgba.height,
        )

    return save_png(rgba, output_path, compress_level=compress_level)


def describe_channels(channels: Iterable[Image.Image]) -> str:
    """生成通道信息文本，用于 UI 状态栏。"""
    names = ("R", "G", "B", "A")
    parts = []
    for name, channel in zip(names, channels):
        if channel is None:
            parts.append(f"{name}: -")
        else:
            parts.append(f"{name}: {channel.width}×{channel.height} {channel.mode}")
    return " · ".join(parts)
