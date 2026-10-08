"""
app.py
======

RGBA Image Composer · 基于 Python + Gradio 的网页图片通道合成工具。

功能：
    * 上传 / 拖拽 R、G、B、A 四个通道图片
    * Alpha 可选择「上传图片」或「使用常量」（如 255 = 完全不透明）
    * 设定输出宽高（含常用尺寸预设），三种缩放模式：Stretch / Fit / Crop
    * Alpha 反转、RGB 反转、PNG 压缩等级
    * 右侧实时预览四个通道 + 最终 RGBA（棋盘格透明背景）
    * 一键生成并下载 PNG，同时自动归档到 output/ 目录

启动：
    python app.py                 # 打开 http://127.0.0.1:7860
    python app.py --port 8000     # 自定义端口
    python app.py --open          # 启动后自动打开浏览器
"""

from __future__ import annotations

import argparse
import html
import tempfile
from pathlib import Path

import gradio as gr

from image_processor import (
    HIGH_BITDEPTH_LABELS,
    RESIZE_MODE_LABELS,
    ChannelError,
    describe_channels,
    output_file_name,
    process_channels,
    save_png,
)

APP_VERSION = "1.1.0"
DEFAULT_SIZE = 1024
OUTPUT_DIR = Path(__file__).resolve().parent / "output"

# 1:1 细节预览的裁剪边长（原始像素，不缩放）
DETAIL_SIZE = 256

ALPHA_UPLOAD = "上传 Alpha 图片"
ALPHA_CONSTANT = "使用常量 Alpha"

# ---------------------------------------------------------------------------
# 样式
# ---------------------------------------------------------------------------

CSS = """
:root {
    --bg: #0f1117;
    --panel: #171a21;
    --panel-2: #1d212b;
    --border: #2b303b;
    --text: #e7eaf0;
    --muted: #8b93a1;
    --accent: #6c8cff;
}

.gradio-container {
    max-width: 1560px !important;
    margin: 0 auto !important;
}

.app-header {
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    gap: 16px;
    padding: 22px 4px 6px 4px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 16px;
}

.app-title {
    font-size: 26px;
    font-weight: 700;
    letter-spacing: 0.2px;
    color: var(--text);
}

.app-subtitle {
    color: var(--muted);
    margin-top: 6px;
    font-size: 13px;
}

.app-badge {
    font-size: 12px;
    color: var(--muted);
    border: 1px solid var(--border);
    border-radius: 999px;
    padding: 4px 12px;
    white-space: nowrap;
}

.section-title {
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--muted);
    margin: 6px 0 10px 0;
}

.channel-card, .preview-card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 12px !important;
}

.preview-card {
    padding: 8px !important;
}

/* 透明棋盘格背景：只画在图片本身，不画容器，
   否则图片按比例缩放后左右/上下的留白也会变成棋盘格，看起来像"透明区域" */
.checkerboard img {
    background-color: #202430;
    background-image:
        linear-gradient(45deg, #2f3542 25%, transparent 25%, transparent 75%, #2f3542 75%),
        linear-gradient(45deg, #2f3542 25%, transparent 25%, transparent 75%, #2f3542 75%);
    background-size: 20px 20px;
    background-position: 0 0, 10px 10px;
}

.status-ok { color: #7ee2a8; font-size: 13px; }
.status-wait { color: var(--muted); font-size: 13px; }
.status-err { color: #ff8f8f; font-size: 13px; }

.footer-note {
    color: var(--muted);
    font-size: 12px;
    text-align: center;
    padding: 14px 0 6px 0;
    border-top: 1px solid var(--border);
    margin-top: 18px;
}
"""


# ---------------------------------------------------------------------------
# 核心处理（UI 适配层）
# ---------------------------------------------------------------------------

def _validate_size(width, height):
    try:
        width = int(width)
        height = int(height)
    except (TypeError, ValueError):
        raise ChannelError("宽高必须是整数")
    if width <= 0 or height <= 0:
        raise ChannelError("输出宽高必须大于 0")
    if width > 16384 or height > 16384:
        raise ChannelError("输出宽高单边上限为 16384")
    return width, height


def _build(
    r_file,
    g_file,
    b_file,
    a_file,
    alpha_mode,
    alpha_value,
    width,
    height,
    resize_mode,
    invert_alpha,
    invert_rgb,
    high_bitdepth,
):
    """统一的参数校验 + 合成入口。

    未上传（为空）的通道按 0 填充；16-bit / 32-bit / 浮点输入按 high_bitdepth 正确换算。

    Returns:
        (r, g, b, a, rgba, width, height, filled, source_notes)，
        filled 为因空而按 0 填充的通道名列表，source_notes 为各通道位深说明。
    """
    width, height = _validate_size(width, height)

    alpha_is_constant = alpha_mode == ALPHA_CONSTANT
    filled: list[str] = []
    source_notes: list[str] = []

    channels = process_channels(
        r_path=r_file,
        g_path=g_file,
        b_path=b_file,
        a_path=None if alpha_is_constant else a_file,
        width=width,
        height=height,
        resize_mode=resize_mode,
        invert_alpha=bool(invert_alpha),
        invert_rgb=bool(invert_rgb),
        alpha_value=int(alpha_value) if alpha_is_constant else None,
        high_bitdepth=high_bitdepth or "scale",
        filled_channels=filled,
        source_notes=source_notes,
    )
    return (*channels, width, height, filled, source_notes)


def _filled_note(filled):
    """把「空通道按 0 填充」整理成一句提示。"""
    if not filled:
        return ""
    parts = [
        "A=0（全透明）" if name == "A" else f"{name}=0（黑）"
        for name in filled
    ]
    return "空通道按 0 填充：" + " / ".join(parts)


def _source_note(source_notes):
    """整理各通道的源位深说明，例如「G: I;16->L(scale)」。"""
    if not source_notes:
        return ""
    return "源位深： " + " · ".join(source_notes)


def detail_preview(image, size=DETAIL_SIZE):
    """截取中心 1:1 区域：高频贴图整图缩放后看起来像纯色，用它判断细节。

    返回的裁剪块**不做缩放**，按原始像素显示。
    """
    crop = max(1, min(int(size), image.width, image.height))
    left = (image.width - crop) // 2
    top = (image.height - crop) // 2
    return image.crop((left, top, left + crop, top + crop))


def live_preview(
    r_file,
    g_file,
    b_file,
    a_file,
    alpha_mode,
    alpha_value,
    width,
    height,
    resize_mode,
    invert_alpha,
    invert_rgb,
    high_bitdepth,
):
    """实时预览：未上传的通道按 0 填充，非法参数只给出状态提示而不抛异常。"""
    try:
        r, g, b, a, rgba, width, height, filled, source_notes = _build(
            r_file, g_file, b_file, a_file, alpha_mode, alpha_value,
            width, height, resize_mode, invert_alpha, invert_rgb, high_bitdepth,
        )
    except Exception as exc:  # noqa: BLE001 - UI 边界统一转成状态提示
        return (None, None, None, None, None, None,
                f"<div class='status-err'>合成失败：{html.escape(str(exc))}</div>", "")

    note = _filled_note(filled)
    if len(filled) == 4:
        status = (
            "<div class='status-wait'>● 四个通道均为空 · 全部按 0 填充 "
            f"· 输出 {width} × {height} px</div>"
        )
    else:
        status = (
            f"<div class='status-ok'>● 预览就绪 · 输出 {width} × {height} px · "
            f"{RESIZE_MODE_LABELS.get(resize_mode, resize_mode)}"
            + (f"<br>{html.escape(note)}" if note else "")
            + "</div>"
        )
    detail = _source_note(source_notes)
    info = (
        "<div class='status-wait'>"
        + html.escape(describe_channels((r, g, b, a)) + f" · RGBA {rgba.width}×{rgba.height}")
        + (f"<br>{html.escape(detail)}" if detail else "")
        + "</div>"
    )
    return r, g, b, a, rgba, detail_preview(rgba), status, info


def generate_png(
    r_file,
    g_file,
    b_file,
    a_file,
    alpha_mode,
    alpha_value,
    width,
    height,
    resize_mode,
    invert_alpha,
    invert_rgb,
    high_bitdepth,
    compress_level,
    archive,
):
    """生成 PNG 供下载，并按需归档到 output/ 目录（空通道按 0 填充）。"""
    try:
        r, g, b, a, rgba, width, height, filled, source_notes = _build(
            r_file, g_file, b_file, a_file, alpha_mode, alpha_value,
            width, height, resize_mode, invert_alpha, invert_rgb, high_bitdepth,
        )
    except Exception as exc:  # noqa: BLE001 - 统一抛给 Gradio 展示
        raise gr.Error(f"合成失败：{exc}")

    file_name = output_file_name(width, height)

    temp_dir = Path(tempfile.mkdtemp(prefix="rgba-composer-"))
    temp_path = temp_dir / file_name
    saved = save_png(rgba, temp_path, compress_level=int(compress_level))

    notes = [f"已生成 {file_name}（{width} × {height} RGBA PNG）"]
    note = _filled_note(filled)
    if note:
        notes.append(note)
    if archive:
        try:
            archived = save_png(
                rgba, OUTPUT_DIR / file_name, compress_level=int(compress_level)
            )
            notes.append(f"已归档：{archived}")
        except Exception as exc:  # noqa: BLE001 - 归档失败不影响下载
            notes.append(f"归档失败（不影响下载）：{exc}")
        # 下载优先使用归档文件，文件名更直观
        if (OUTPUT_DIR / file_name).exists():
            saved = str((OUTPUT_DIR / file_name).resolve())

    notes.append("点击下方文件即可下载，或右键「链接另存为」")
    status = "<div class='status-ok'>" + "<br>".join(html.escape(n) for n in notes) + "</div>"
    detail = _source_note(source_notes)
    info = (
        "<div class='status-wait'>"
        + html.escape(describe_channels((r, g, b, a)))
        + (f"<br>{html.escape(detail)}" if detail else "")
        + "</div>"
    )
    return saved, status, info


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def build_app() -> gr.Blocks:
    with gr.Blocks(
        title="RGBA Image Composer",
        css=CSS,
        theme=gr.themes.Base(),
    ) as demo:

        gr.HTML(
            f"""
            <div class="app-header">
                <div>
                    <div class="app-title">RGBA Image Composer</div>
                    <div class="app-subtitle">
                        把 R / G / B / A 四张通道图合成为指定尺寸的 RGBA PNG · 实时预览
                    </div>
                </div>
                <div class="app-badge">v{APP_VERSION} · Python + Gradio + Pillow</div>
            </div>
            """
        )

        with gr.Row(equal_height=False):

            # ============================ 左：输入 & 参数 ============================
            with gr.Column(scale=4, min_width=420, elem_classes="channel-card"):

                gr.HTML("<div class='section-title'>Input Channels · 输入通道</div>")
                gr.HTML(
                    "<div class='status-wait'>未上传的通道按 <b>0</b> 填充："
                    "R/G/B 记为该颜色分量为 0（黑），A 记为完全透明；"
                    "四个通道可以只上传其中任意几个。</div>"
                )

                r_input = gr.Image(
                    label="R · Red 通道",
                    type="filepath",
                    # 必须为 None：Gradio 5 默认 "RGB"，会在服务端先把 16-bit 图转成
                    # 8-bit RGB（>255 直接截断成纯白），原始位深数据就丢失了。
                    image_mode=None,
                    sources=["upload", "clipboard"],
                    height=170,
                )
                g_input = gr.Image(
                    label="G · Green 通道",
                    type="filepath",
                    image_mode=None,
                    sources=["upload", "clipboard"],
                    height=170,
                )
                b_input = gr.Image(
                    label="B · Blue 通道",
                    type="filepath",
                    image_mode=None,
                    sources=["upload", "clipboard"],
                    height=170,
                )

                with gr.Row():
                    with gr.Column(scale=3):
                        a_input = gr.Image(
                            label="A · Alpha 通道",
                            type="filepath",
                            image_mode=None,
                            sources=["upload", "clipboard"],
                            height=150,
                        )
                    with gr.Column(scale=2):
                        alpha_mode = gr.Radio(
                            choices=[ALPHA_UPLOAD, ALPHA_CONSTANT],
                            value=ALPHA_UPLOAD,
                            label="Alpha 来源",
                        )
                        alpha_value = gr.Slider(
                            minimum=0,
                            maximum=255,
                            step=1,
                            value=255,
                            label="常量 Alpha（255 = 完全不透明）",
                            visible=False,
                        )

                gr.HTML("<div class='section-title'>Output · 输出设置</div>")

                with gr.Row():
                    width = gr.Number(
                        value=DEFAULT_SIZE, precision=0, label="宽度 Width", minimum=1,
                        maximum=16384,
                    )
                    height = gr.Number(
                        value=DEFAULT_SIZE, precision=0, label="高度 Height", minimum=1,
                        maximum=16384,
                    )

                size_preset = gr.Dropdown(
                    choices=[
                        "512 × 512", "1024 × 1024", "2048 × 2048", "4096 × 4096",
                        "1024 × 512", "512 × 1024", "2048 × 1024", "1920 × 1080",
                    ],
                    value=None,
                    label="尺寸预设（选中即填入宽高）",
                    allow_custom_value=False,
                )

                resize_mode = gr.Dropdown(
                    choices=[(label, key) for key, label in RESIZE_MODE_LABELS.items()],
                    value="stretch",
                    label="缩放模式 Resize Mode",
                )

                with gr.Row():
                    invert_alpha = gr.Checkbox(value=False, label="Alpha 反转")
                    invert_rgb = gr.Checkbox(value=False, label="RGB 反转")

                high_bitdepth = gr.Dropdown(
                    choices=[
                        (label, key) for key, label in HIGH_BITDEPTH_LABELS.items()
                    ],
                    value="scale",
                    label="16 / 32-bit 位深换算",
                    info="16-bit 灰度贴图不再被截断成纯白；蒙版只用了部分量程时选「实际范围拉伸」",
                )

                compress_level = gr.Slider(
                    minimum=0, maximum=9, step=1, value=6,
                    label="PNG 压缩等级（0 最快 / 9 最小，均无损）",
                )

                archive = gr.Checkbox(
                    value=True, label=f"同时归档到 {OUTPUT_DIR.name}/ 目录",
                )

                with gr.Row():
                    generate_button = gr.Button(
                        "生成 PNG", variant="primary", elem_classes="generate-btn",
                    )
                    clear_button = gr.Button("清空重置")

                download_file = gr.File(label="PNG 下载", interactive=False)
                status = gr.HTML(
                    "<div class='status-wait'>● 四个通道均为空 · 全部按 0 填充</div>",
                    elem_classes="status",
                )

            # ============================ 右：预览 ============================
            with gr.Column(scale=6):

                gr.HTML("<div class='section-title'>Channel Preview · 通道预览</div>")

                with gr.Row():
                    with gr.Column(elem_classes="preview-card"):
                        r_preview = gr.Image(
                            label="R", height=190, interactive=False, format="png",
                            show_download_button=False, show_fullscreen_button=True,
                        )
                    with gr.Column(elem_classes="preview-card"):
                        g_preview = gr.Image(
                            label="G", height=190, interactive=False, format="png",
                            show_download_button=False, show_fullscreen_button=True,
                        )
                with gr.Row():
                    with gr.Column(elem_classes="preview-card"):
                        b_preview = gr.Image(
                            label="B", height=190, interactive=False, format="png",
                            show_download_button=False, show_fullscreen_button=True,
                        )
                    with gr.Column(elem_classes="preview-card"):
                        a_preview = gr.Image(
                            label="Alpha", height=190, interactive=False, format="png",
                            show_download_button=False, show_fullscreen_button=True,
                        )

                gr.HTML("<div class='section-title'>RGBA Output · 合成结果</div>")

                with gr.Column(elem_classes=["preview-card", "checkerboard"]):
                    rgba_preview = gr.Image(
                        label="最终 RGBA（棋盘格 = 透明区域）",
                        height=430,
                        interactive=False,
                        format="png",
                        show_download_button=False,
                        show_fullscreen_button=True,
                    )

                with gr.Column(elem_classes=["preview-card", "checkerboard"]):
                    rgba_detail = gr.Image(
                        label=f"1:1 细节预览（中心 {DETAIL_SIZE}×{DETAIL_SIZE} 原始像素，不缩放）",
                        height=DETAIL_SIZE + 40,
                        interactive=False,
                        format="png",
                        show_download_button=False,
                        show_fullscreen_button=True,
                    )

                info = gr.HTML(
                    "<div class='status-wait'>通道信息将在合成后显示</div>",
                    elem_classes="status",
                )

        gr.HTML(
            "<div class='footer-note'>"
            "提示：四个通道都会先按源位深正确转成 8-bit 灰度（16-bit ÷257，不再截断成纯白），"
            "再统一缩放到输出尺寸后合并；PNG 无损，Alpha 不会被丢弃。"
            "</div>"
        )

        # ============================ 事件绑定 ============================

        preview_inputs = [
            r_input, g_input, b_input, a_input,
            alpha_mode, alpha_value,
            width, height, resize_mode,
            invert_alpha, invert_rgb, high_bitdepth,
        ]
        preview_outputs = [
            r_preview, g_preview, b_preview, a_preview,
            rgba_preview, rgba_detail, status, info,
        ]

        def bind_preview(component, event="change"):
            """为组件绑定实时预览事件（兼容不同 Gradio 5 小版本）。"""
            listener = getattr(component, event)
            try:
                return listener(
                    fn=live_preview,
                    inputs=preview_inputs,
                    outputs=preview_outputs,
                    show_progress="hidden",
                    trigger_mode="always_last",
                )
            except TypeError:
                return listener(
                    fn=live_preview,
                    inputs=preview_inputs,
                    outputs=preview_outputs,
                    show_progress="hidden",
                )

        for component in (r_input, g_input, b_input, a_input):
            bind_preview(component, "change")
            bind_preview(component, "clear")

        for component in (
            alpha_mode, alpha_value, width, height,
            resize_mode, invert_alpha, invert_rgb, high_bitdepth,
        ):
            bind_preview(component, "change")

        alpha_mode.change(
            fn=lambda mode: gr.update(visible=(mode == ALPHA_CONSTANT)),
            inputs=alpha_mode,
            outputs=alpha_value,
            show_progress="hidden",
        )

        def apply_preset(preset):
            if not preset:
                return gr.update(), gr.update()
            width_text, height_text = preset.replace(" ", "").split("×")
            return gr.update(value=int(width_text)), gr.update(value=int(height_text))

        size_preset.change(
            fn=apply_preset,
            inputs=size_preset,
            outputs=[width, height],
            show_progress="hidden",
        )

        generate_event_inputs = preview_inputs + [compress_level, archive]
        generate_button.click(
            fn=generate_png,
            inputs=generate_event_inputs,
            outputs=[download_file, status, info],
        )

        def reset_all():
            return (
                None, None, None, None,
                ALPHA_UPLOAD, 255,
                DEFAULT_SIZE, DEFAULT_SIZE, "stretch",
                False, False, "scale",
                6, True,
                None, None, None, None, None, None,
                "<div class='status-wait'>● 四个通道均为空 · 全部按 0 填充</div>",
                "",
            )

        clear_event = clear_button.click(
            fn=reset_all,
            inputs=None,
            outputs=preview_inputs + [compress_level, archive] + preview_outputs,
        )
        # 清空后立即按「空通道 = 0」重算一次预览
        clear_event.then(
            fn=live_preview,
            inputs=preview_inputs,
            outputs=preview_outputs,
            show_progress="hidden",
        )

        demo.load(
            fn=live_preview,
            inputs=preview_inputs,
            outputs=preview_outputs,
            show_progress="hidden",
        )

    return demo


# ---------------------------------------------------------------------------
# 启动
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="RGBA Image Composer")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址")
    parser.add_argument("--port", type=int, default=7860, help="监听端口")
    parser.add_argument("--share", action="store_true", help="生成 Gradio 公网临时链接")
    parser.add_argument("--open", action="store_true", help="启动后自动打开浏览器")
    return parser.parse_args()


def main():
    args = parse_args()
    app = build_app()
    app.queue(default_concurrency_limit=4).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        inbrowser=args.open,
        show_api=False,
        allowed_paths=[str(OUTPUT_DIR)],
    )


if __name__ == "__main__":
    main()
