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
import warnings
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
        return image

    # 8-bit 及以下：RGB / RGBA / P / LA / CMYK / 1 … 直接用亮度转灰度
    return image.convert("L")


# ---------------------------------------------------------------------------
# 基础操作
# ---------------------------------------------------------------------------

def _read_channel(file_path, *, invert: bool = False, high_bitdepth: str = "scale"):
    """读取图片 -> 8-bit 灰度，返回 (channel, 源模式说明)。"""
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

    try:
        with Image.open(file_path) as image:
            # JPEG 等带 EXIF 方向的图片：与浏览器/看图软件看到的方向保持一致
            if image.getexif().get(274, 1) != 1:
                transposed = None
                with contextlib.suppress(Exception):
                    transposed = ImageOps.exif_transpose(image)
                if transposed is not None:
                    image = transposed
            source_mode = image.mode
            channel = to_grayscale_8bit(image, high_bitdepth)
            channel.load()
    except ChannelError:
        raise
    except Exception as exc:  # Pillow 的解码异常种类很多，统一成可读错误
        raise ChannelError(f"无法读取图片 {file_path}：{exc}") from exc

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
