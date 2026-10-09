"""
ui_tools.py
===========

RGBA Image Composer 的两个附加工具面板（Gradio UI）：
    ② 四宫格拼图：四张图各自保留 RGBA -> 拼成一张 2×2 的新图（左上/右上/左下/右下）
    ③ 通道拆分：一张图片 -> R / G / B / A 四个通道，可分别或打包下载

两个面板都以 ``with gr.Tab(...)`` 的形式构建，供 app.py 放进 ``gr.Tabs()`` 里。
图像处理全部调用 image_processor，本文件只负责界面与文件落盘。
"""

from __future__ import annotations

import html
import tempfile
from pathlib import Path

import gradio as gr

from image_processor import (
    DEFAULT_OUTPUT_DIR,
    HIGH_BITDEPTH_LABELS,
    ChannelError,
    mosaic_quadrants,
    output_file_name,
    save_channel_png,
    save_png,
    split_channels,
    zip_files,
)

OUTPUT_DIR = DEFAULT_OUTPUT_DIR

SIZE_AUTO = "自动（= 第一张图的原始尺寸，每块减半）"
SIZE_MANUAL = "手动指定"

BITDEPTH_CHOICES = [(label, key) for key, label in HIGH_BITDEPTH_LABELS.items()]


def _ok(message: str) -> str:
    return f"<div class='status-ok'>{message}</div>"


def _wait(message: str) -> str:
    return f"<div class='status-wait'>{message}</div>"


def _err(message: str) -> str:
    return f"<div class='status-err'>{html.escape(message)}</div>"


def _report_html(lines) -> str:
    if not lines:
        return ""
    body = "<br>".join(html.escape(str(line)) for line in lines)
    return f"<div class='status-wait'>{body}</div>"


# ---------------------------------------------------------------------------
# ② 四宫格拼图（四张图各自保留 RGBA -> 一张新图）
# ---------------------------------------------------------------------------

TILE_MODE_CHOICES = [
    ("拉伸铺满 Stretch（默认）", "stretch"),
    ("保持比例 + 透明填充 Fit", "fit"),
    ("保持比例居中裁剪 Crop", "crop"),
]

QUADRANTS = [
    ("tl", "左上 TL"),
    ("tr", "右上 TR"),
    ("bl", "左下 BL"),
    ("br", "右下 BR"),
]

def _center_crop(image, size=256):
    crop = max(1, min(int(size), image.width, image.height))
    left, top = (image.width - crop) // 2, (image.height - crop) // 2
    return image.crop((left, top, left + crop, top + crop))

def mosaic_preview(
    tl_file,
    tr_file,
    bl_file,
    br_file,
    size_mode,
    width,
    height,
    tile_mode,
    high_bitdepth,
):
    """实时预览：四张图 -> 2×2 拼图。"""
    auto = size_mode == SIZE_AUTO
    try:
        atlas, size = mosaic_quadrants(
            tl_path=tl_file,
            tr_path=tr_file,
            bl_path=bl_file,
            br_path=br_file,
            width=None if auto else width,
            height=None if auto else height,
            tile_mode=tile_mode or "stretch",
            high_bitdepth=high_bitdepth or "scale",
        )
    except ChannelError as exc:
        return None, None, _err(str(exc)), ""
    except Exception as exc:  # noqa: BLE001 - UI 边界统一转成状态提示
        return None, None, _err(f"拼接失败：{exc}"), ""

    status = _ok(
        f"● 拼图预览就绪 · 输出 {size[0]} × {size[1]} px · "
        f"每块 {size[0] // 2} × {size[1] // 2} px"
        f"（{'自动尺寸' if auto else '手动尺寸'}）"
    )
    return atlas, _center_crop(atlas, 256), status, ""

def mosaic_generate(
    tl_file,
    tr_file,
    bl_file,
    br_file,
    size_mode,
    width,
    height,
    tile_mode,
    high_bitdepth,
    compress_level,
    archive,
):
    """生成 2×2 拼图 PNG 供下载。"""
    auto = size_mode == SIZE_AUTO
    report: list[str] = []
    try:
        atlas, size = mosaic_quadrants(
            tl_path=tl_file,
            tr_path=tr_file,
            bl_path=bl_file,
            br_path=br_file,
            width=None if auto else width,
            height=None if auto else height,
            tile_mode=tile_mode or "stretch",
            high_bitdepth=high_bitdepth or "scale",
            size_report=report,
        )
    except Exception as exc:  # noqa: BLE001 - 统一抛给 Gradio 展示
        raise gr.Error(f"拼接失败：{exc}")

    file_name = output_file_name(size[0], size[1], prefix="mosaic")
    temp_dir = Path(tempfile.mkdtemp(prefix="rgba-mosaic-"))
    saved = save_png(atlas, temp_dir / file_name, compress_level=int(compress_level))

    notes = [f"已生成 {file_name}（{size[0]} × {size[1]} RGBA PNG，2×2 四宫格）"]
    if archive:
        try:
            save_png(atlas, OUTPUT_DIR / file_name, compress_level=int(compress_level))
            notes.append(f"已归档：{OUTPUT_DIR / file_name}")
            saved = str((OUTPUT_DIR / file_name).resolve())
        except Exception as exc:  # noqa: BLE001 - 归档失败不影响下载
            notes.append(f"归档失败（不影响下载）：{exc}")

    return saved, _ok("<br>".join(html.escape(n) for n in notes)), _report_html(report)

def build_mosaic_tab():
    """构建「四宫格拼图」面板。"""
    with gr.Tab("② 四宫格拼图 · 四张 → 一张"):

        gr.HTML(
            "<div class='section-title'>2×2 Quadrant Mosaic</div>"
            "<div class='status-wait'>四张图<b>各自保留自己的 RGBA 通道</b>，"
            "分别放到新图的<b>左上 / 右上 / 左下 / 右下</b>四个位置："
            "四张 2048×2048 → 每张缩放到 1024×1024 → 一张新的 2048×2048。"
            "输出尺寸默认 = 第一张图的原始尺寸（每块减半），也可手动指定。</div>"
        )

        with gr.Row(equal_height=False):

            with gr.Column(scale=4, min_width=420, elem_classes="channel-card"):
                gr.HTML("<div class='section-title'>输入 · 按四宫格位置上传</div>")

                with gr.Row():
                    with gr.Column():
                        mosaic_tl = gr.Image(
                            label="① 左上 TL", type="filepath", image_mode=None,
                            sources=["upload", "clipboard"], height=150,
                        )
                    with gr.Column():
                        mosaic_tr = gr.Image(
                            label="② 右上 TR", type="filepath", image_mode=None,
                            sources=["upload", "clipboard"], height=150,
                        )
                with gr.Row():
                    with gr.Column():
                        mosaic_bl = gr.Image(
                            label="③ 左下 BL", type="filepath", image_mode=None,
                            sources=["upload", "clipboard"], height=150,
                        )
                    with gr.Column():
                        mosaic_br = gr.Image(
                            label="④ 右下 BR", type="filepath", image_mode=None,
                            sources=["upload", "clipboard"], height=150,
                        )

                gr.HTML("<div class='section-title'>拼图设置</div>")

                mosaic_size_mode = gr.Radio(
                    choices=[SIZE_AUTO, SIZE_MANUAL],
                    value=SIZE_AUTO,
                    label="输出尺寸",
                    info="自动 = 第一张图的原始尺寸（例：四张 2048 → 输出 2048，每块 1024）",
                )
                with gr.Row(visible=False) as mosaic_manual_row:
                    mosaic_width = gr.Number(value=2048, precision=0, minimum=2,
                                             maximum=16384, label="宽度")
                    mosaic_height = gr.Number(value=2048, precision=0, minimum=2,
                                              maximum=16384, label="高度")

                mosaic_tile_mode = gr.Dropdown(
                    choices=TILE_MODE_CHOICES, value="stretch",
                    label="每块适配方式",
                    info="源图与格子比例一致时，三种方式结果相同",
                )
                mosaic_depth = gr.Dropdown(
                    choices=BITDEPTH_CHOICES, value="scale",
                    label="16 / 32-bit 位深换算",
                    info="16-bit 灰度图会先正确换算成 8-bit，不会被截断成纯白",
                )
                mosaic_compress = gr.Slider(
                    minimum=0, maximum=9, step=1, value=6,
                    label="PNG 压缩等级（无损）",
                )
                mosaic_archive = gr.Checkbox(
                    value=True, label=f"同时归档到 {OUTPUT_DIR.name}/ 目录",
                )

                with gr.Row():
                    mosaic_button = gr.Button("一键拼接 PNG", variant="primary")
                    mosaic_clear = gr.Button("清空")

                mosaic_file = gr.File(label="PNG 下载", interactive=False)
                mosaic_status = gr.HTML(_wait("等待上传图片（至少要有一张）"), elem_classes="status")

            with gr.Column(scale=6):
                gr.HTML("<div class='section-title'>拼图结果预览</div>")
                with gr.Column(elem_classes=["preview-card", "checkerboard"]):
                    mosaic_preview_image = gr.Image(
                        label="2×2 拼图输出（棋盘格 = 透明区域）", height=380,
                        interactive=False, format="png",
                        show_download_button=False, show_fullscreen_button=True,
                    )
                with gr.Column(elem_classes=["preview-card", "checkerboard"]):
                    mosaic_detail_image = gr.Image(
                        label="1:1 细节预览（中心 256×256）", height=290,
                        interactive=False, format="png",
                        show_download_button=False, show_fullscreen_button=True,
                    )
                mosaic_report = gr.HTML("", elem_classes="status")

        # ---------------- 事件 ----------------
        mosaic_inputs = [
            mosaic_tl, mosaic_tr, mosaic_bl, mosaic_br,
            mosaic_size_mode, mosaic_width, mosaic_height,
            mosaic_tile_mode, mosaic_depth,
        ]
        mosaic_outputs = [
            mosaic_preview_image, mosaic_detail_image, mosaic_status, mosaic_report,
        ]

        def mosaic_bind(component, event="change"):
            listener = getattr(component, event)
            try:
                return listener(
                    fn=mosaic_preview, inputs=mosaic_inputs, outputs=mosaic_outputs,
                    show_progress="hidden", trigger_mode="always_last",
                )
            except TypeError:
                return listener(
                    fn=mosaic_preview, inputs=mosaic_inputs, outputs=mosaic_outputs,
                    show_progress="hidden",
                )

        for component in (mosaic_tl, mosaic_tr, mosaic_bl, mosaic_br):
            mosaic_bind(component, "change")
            mosaic_bind(component, "clear")
        for component in (
            mosaic_size_mode, mosaic_width, mosaic_height, mosaic_tile_mode, mosaic_depth,
        ):
            mosaic_bind(component, "change")

        mosaic_size_mode.change(
            fn=lambda mode: gr.update(visible=(mode == SIZE_MANUAL)),
            inputs=mosaic_size_mode,
            outputs=mosaic_manual_row,
            show_progress="hidden",
        )

        mosaic_button.click(
            fn=mosaic_generate,
            inputs=mosaic_inputs + [mosaic_compress, mosaic_archive],
            outputs=[mosaic_file, mosaic_status, mosaic_report],
        )

        def mosaic_reset():
            return (
                None, None, None, None,
                SIZE_AUTO, 2048, 2048, "stretch", "scale",
                6, True,
                None, None, _wait("等待上传图片（至少要有一张）"), "",
            )

        mosaic_clear_event = mosaic_clear.click(
            fn=mosaic_reset,
            inputs=None,
            outputs=mosaic_inputs + [mosaic_compress, mosaic_archive] + mosaic_outputs,
        )
        mosaic_clear_event.then(
            fn=mosaic_preview, inputs=mosaic_inputs, outputs=mosaic_outputs,
            show_progress="hidden",
        )

# ---------------------------------------------------------------------------
# ③ 通道拆分
# ---------------------------------------------------------------------------

def split_preview(file_path, high_bitdepth):
    """实时预览：一张图片 -> R / G / B / A 四个通道。"""
    if not file_path:
        return None, None, None, None, _wait("等待上传要拆分的图片"), ""
    try:
        r, g, b, a, info = split_channels(file_path, high_bitdepth=high_bitdepth or "scale")
    except Exception as exc:  # noqa: BLE001 - UI 边界统一转成状态提示
        return None, None, None, None, _err(f"拆分失败：{exc}"), ""

    status = _ok(
        f"● 已拆分 · 源图 {info['name']} · {info['width']} × {info['height']} px · "
        f"模式 {info['mode']}（{info['bit_depth']}-bit）"
    )
    info_html = _report_html([
        "R / G / B：取自图像的红 / 绿 / 蓝分量",
        f"A：{info['alpha_source']}",
        f"输出：四个 8-bit 灰度 PNG，尺寸与源图一致（{info['width']}×{info['height']}）",
    ])
    return r, g, b, a, status, info_html


def split_run(file_path, high_bitdepth, compress_level, make_zip):
    """把四个通道写成灰度 PNG（固定写入 output/<名字>_split/，便于①的通道池选用）。

    Returns:
        (四个通道路径, ZIP 路径或 None, 状态, 报告)
    """
    if not file_path:
        raise gr.Error("请先上传要拆分的图片")

    try:
        r, g, b, a, info = split_channels(file_path, high_bitdepth=high_bitdepth or "scale")
    except Exception as exc:  # noqa: BLE001 - 统一抛给 Gradio 展示
        raise gr.Error(f"拆分失败：{exc}")

    stem = Path(info["name"]).stem
    # 固定写到 output/<名字>_split/：① 面板的「通道池」要能直接读取这些文件
    out_dir = OUTPUT_DIR / f"{stem}_split"
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    for label, channel in (("R", r), ("G", g), ("B", b), ("A", a)):
        # 通道按灰度 PNG 导出（单通道、体积小、导入 Unity/PS 语义正确）
        paths.append(save_channel_png(channel, out_dir / f"{stem}_{label}.png",
                                      compress_level=int(compress_level)))

    zip_path = (
        zip_files(paths, out_dir / f"{stem}_RGBA_channels.zip") if make_zip else None
    )

    notes = [
        f"已拆分 {info['name']}（{info['width']} × {info['height']} px，{info['mode']}）",
        f"A 通道来源：{info['alpha_source']}",
        f"输出目录：{out_dir}",
        "这四个文件已加入 ① 面板的「通道池」，可直接选进 R/G/B/A 槽位",
    ]
    return (*paths, zip_path, _ok("<br>".join(html.escape(n) for n in notes)),
            _report_html([f"{Path(p).name}" for p in paths]))


def build_splitter_tab(composer_images=None, composer_pickers=None):
    """构建「通道拆分」面板。

    Args:
        composer_images: ① 面板的四个通道输入框（R/G/B/A 顺序），用于「送入通道合成」。
        composer_pickers: ① 面板的四个通道池下拉框，同一顺序。

    Returns:
        dict：{"send_button": 按钮组件或 None, "state": gr.State}
    """
    with gr.Tab("③ 通道拆分 · 一张 → R/G/B/A"):

        gr.HTML(
            "<div class='section-title'>Channel Split</div>"
            "<div class='status-wait'>上传一张图片（PNG / JPG / TIF，支持 16-bit），"
            "拆出它的 <b>R / G / B / A</b> 四个通道，分别下载或打包成 ZIP。"
            "原图没有 Alpha 时，A 按 255（完全不透明）输出。<br>"
            "拆分结果固定写入 <code>output/&lt;名字&gt;_split/</code>，"
            "并自动进入 ① 面板的<b>通道池</b>，可以直接接进 R/G/B/A 槽位。</div>"
        )

        with gr.Row(equal_height=False):

            with gr.Column(scale=4, min_width=420, elem_classes="channel-card"):
                gr.HTML("<div class='section-title'>输入 · 待拆分图片</div>")
                split_input = gr.Image(
                    label="源图（RGBA / RGB / 灰度 / 16-bit 均可）",
                    type="filepath", image_mode=None,
                    sources=["upload", "clipboard"], height=220,
                )
                split_depth = gr.Dropdown(
                    choices=BITDEPTH_CHOICES, value="scale",
                    label="16 / 32-bit 位深换算",
                    info="16-bit 灰度会先正确换算成 8-bit，再作为 R=G=B 使用",
                )
                split_compress = gr.Slider(
                    minimum=0, maximum=9, step=1, value=6, label="PNG 压缩等级（无损）",
                )
                split_archive = gr.Checkbox(
                    value=True, label="同时打包 ZIP（通道固定导出到 "
                    f"{OUTPUT_DIR.name}/<图片名>_split/，供①的通道池选用）",
                )

                with gr.Row():
                    split_button = gr.Button("拆分并导出通道", variant="primary")
                    split_clear = gr.Button("清空")

                split_files = [
                    gr.File(label="R 通道 PNG", interactive=False),
                    gr.File(label="G 通道 PNG", interactive=False),
                    gr.File(label="B 通道 PNG", interactive=False),
                    gr.File(label="A 通道 PNG", interactive=False),
                ]
                split_zip = gr.File(label="全部通道（ZIP）", interactive=False)
                split_status = gr.HTML(_wait("等待上传要拆分的图片"), elem_classes="status")

                split_send = gr.Button(
                    "→ 送入 ① 通道合成（R/G/B/A 直接接好）",
                    variant="secondary",
                ) if composer_images else None

            with gr.Column(scale=6):
                gr.HTML("<div class='section-title'>通道预览</div>")
                with gr.Row():
                    with gr.Column(elem_classes="preview-card"):
                        split_r = gr.Image(label="R", height=210, interactive=False,
                                           format="png", show_download_button=False)
                    with gr.Column(elem_classes="preview-card"):
                        split_g = gr.Image(label="G", height=210, interactive=False,
                                           format="png", show_download_button=False)
                with gr.Row():
                    with gr.Column(elem_classes="preview-card"):
                        split_b = gr.Image(label="B", height=210, interactive=False,
                                           format="png", show_download_button=False)
                    with gr.Column(elem_classes="preview-card"):
                        split_a = gr.Image(label="A（Alpha）", height=210, interactive=False,
                                           format="png", show_download_button=False)
                split_info = gr.HTML("", elem_classes="status")

        # ---------------- 事件 ----------------
        split_preview_inputs = [split_input, split_depth]
        split_preview_outputs = [split_r, split_g, split_b, split_a, split_status, split_info]

        def split_bind(component, event="change"):
            listener = getattr(component, event)
            try:
                return listener(
                    fn=split_preview, inputs=split_preview_inputs,
                    outputs=split_preview_outputs,
                    show_progress="hidden", trigger_mode="always_last",
                )
            except TypeError:
                return listener(
                    fn=split_preview, inputs=split_preview_inputs,
                    outputs=split_preview_outputs, show_progress="hidden",
                )

        split_bind(split_input, "change")
        split_bind(split_input, "clear")
        split_bind(split_depth, "change")

        split_event = split_button.click(
            fn=split_run,
            inputs=split_preview_inputs + [split_compress, split_archive],
            outputs=[*split_files, split_zip, split_status, split_info],
        )

        def split_reset():
            return (
                None, "scale", 6, True,
                None, None, None, None,
                _wait("等待上传要拆分的图片"), "",
                None,
            )

        split_clear.click(
            fn=split_reset,
            inputs=None,
            outputs=split_preview_inputs + [split_compress, split_archive]
            + split_preview_outputs + [split_zip],
        )

    return {
        "send_button": split_send,
        "files": split_files,
        "split_event": split_event,
    }
