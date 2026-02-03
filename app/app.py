"""Gradio application for building segmentation on aerial imagery."""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Literal

import gradio as gr
import numpy as np
import torch
from huggingface_hub import hf_hub_download

logger = logging.getLogger(__name__)

# HF Hub repository for example images
HF_EXAMPLES_REPO = "MindForgeTim/building-segmentation-examples"
EXAMPLE_FILENAMES = ["austin_suburban.jpg", "tyrol_alpine.jpg", "vienna_urban.jpg"]

# Local cache directory for examples
EXAMPLES_DIR = Path(__file__).parent / "examples"


def _download_examples() -> list[str]:
    """
    Download example images from HF Hub.

    Returns:
        List of absolute paths to downloaded example images.
    """
    example_paths = []
    EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    for filename in EXAMPLE_FILENAMES:
        local_path = EXAMPLES_DIR / filename
        try:
            hf_hub_download(
                repo_id=HF_EXAMPLES_REPO,
                filename=filename,
                repo_type="dataset",
                local_dir=EXAMPLES_DIR,
            )
            example_paths.append(str(local_path))
            logger.info("Downloaded example: %s", filename)
        except Exception:
            logger.warning("Failed to download example: %s", filename)
            # Если файл уже есть локально, используем его
            if local_path.exists():
                example_paths.append(str(local_path))

    return example_paths

try:
    from app.config import DEFAULT_GSD_M
    from app.gsd_model import estimate_gsd, get_gsd_transform, load_gsd_model
    from app.inference import calculate_area, count_buildings, get_device, predict_mask
    from app.postprocessing import calculate_coverage_percent, create_overlay
    from app.preprocessing import load_image, normalize_image, resize_image, validate_image
except ImportError:
    from config import DEFAULT_GSD_M
    from gsd_model import estimate_gsd, get_gsd_transform, load_gsd_model
    from inference import calculate_area, count_buildings, get_device, predict_mask
    from postprocessing import calculate_coverage_percent, create_overlay
    from preprocessing import load_image, normalize_image, resize_image, validate_image


# Деловой минималистичный CSS
CUSTOM_CSS = """
/* Глобальный шрифт для ВСЕХ элементов */
.gradio-container,
.gradio-container * {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* Основной контейнер */
.gradio-container {
    background: #f8fafc !important;
    max-width: 1400px !important;
}

/* Скрытие футера */
footer { display: none !important; }

/* Единый стиль для всех лейблов */
.section-label,
.stat-label,
.gradio-container label,
.gradio-container .label-wrap span,
.gradio-container .label-wrap,
span[data-testid="block-label"],
.gradio-container .gallery-label,
.gradio-container [class*="example"] label,
.gradio-container [class*="example"] span,
.gradio-container .prose {
    font-size: 0.6875rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    color: #64748b !important;
}

/* Компактные отступы для лейблов секций */
.section-label,
.stat-label {
    display: block !important;
    margin: 0 0 4px 0 !important;
    padding: 0 !important;
}

/* Уменьшаем gap между элементами в колонках */
.gradio-container .gap,
.gradio-container .form {
    gap: 4px !important;
}

/* Компактные Row */
.gradio-container [class*="row"] {
    gap: 12px !important;
}

/* Статистические значения */
.stat-value input {
    font-size: 1.75rem !important;
    font-weight: 700 !important;
    color: #0f172a !important;
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
    text-align: left !important;
}

/* Все кнопки */
.gradio-container button {
    font-size: 0.875rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.025em !important;
}

/* Primary кнопка */
.primary-btn {
    background: #0f172a !important;
    border: none !important;
    border-radius: 8px !important;
    transition: all 0.15s ease !important;
}

.primary-btn:hover {
    background: #1e293b !important;
    transform: translateY(-1px) !important;
}

/* Dropdown и Input */
.gradio-container input,
.gradio-container select,
.gradio-container textarea,
.gradio-container .wrap input,
.gradio-container [data-testid="textbox"] input {
    font-size: 0.875rem !important;
    font-weight: 400 !important;
    color: #0f172a !important;
}

/* Dropdown текст */
.gradio-container .wrap,
.gradio-container .secondary-wrap,
.gradio-container span.svelte-1gfkn6j {
    font-size: 0.875rem !important;
    font-weight: 400 !important;
}

/* Placeholder текст в image upload */
.gradio-container .upload-text,
.gradio-container .image-container span,
.gradio-container [data-testid="image"] span {
    font-size: 0.8125rem !important;
    font-weight: 400 !important;
    color: #64748b !important;
}

/* Блоки изображений */
.image-block {
    border: 1px solid #e2e8f0 !important;
    border-radius: 8px !important;
    overflow: hidden !important;
}

/* Секция настроек */
.settings-row {
    background: #f1f5f9 !important;
    border-radius: 8px !important;
    padding: 12px !important;
    margin-top: 8px !important;
}

/* Примеры изображений */
.examples-row {
    gap: 16px !important;
    justify-content: flex-start !important;
}
"""

# =============================================================================
# GSD model cache
# =============================================================================

_gsd_model_cache: torch.nn.Module | None = None
_gsd_model_lock = threading.Lock()

def _get_gsd_model() -> torch.nn.Module:
    """Get cached GSD model, loading if necessary."""
    global _gsd_model_cache
    if _gsd_model_cache is None:
        with _gsd_model_lock:
            # Double-check after acquiring lock
            if _gsd_model_cache is None:
                device = get_device()
                _gsd_model_cache = load_gsd_model(device)
    return _gsd_model_cache


# =============================================================================
# Processing functions
# =============================================================================


def auto_detect_gsd(image) -> float:
    """
    Auto-detect GSD from image using the GSD estimation model.

    Args:
        image: Input RGB image (PIL Image or numpy array).

    Returns:
        Estimated GSD in meters/pixel, or default if detection fails.
    """
    from PIL import Image as PILImage

    if image is None:
        return DEFAULT_GSD_M

    try:
        # Convert PIL to numpy if needed
        if isinstance(image, PILImage.Image):
            image = np.array(image.convert("RGB"))

        # Загрузка и валидация
        image = load_image(image)

        if not validate_image(image):
            logger.warning("Invalid image for GSD estimation, using default GSD")
            return DEFAULT_GSD_M

        # Подготовка изображения для GSD модели
        device = get_device()
        model = _get_gsd_model()

        # Конвертация в tensor с нормализацией (GSD модель ожидает 256x256)
        resized = resize_image(image, target_size=(256, 256))
        tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
        tensor = tensor.unsqueeze(0)

        # Нормализация ImageNet
        normalizer = get_gsd_transform()
        tensor = normalizer(tensor).to(device)

        # Оценка GSD
        gsd_m = estimate_gsd(model, tensor, device)

        # Ограничиваем разумным диапазоном (0.1 - 5.0 м/пиксель)
        gsd_m = max(0.1, min(5.0, gsd_m))

        return round(gsd_m, 3)

    except Exception:
        logger.exception("Failed to estimate GSD, using default value")
        return DEFAULT_GSD_M


def _convert_for_display(image):
    """
    Convert uploaded image to ensure browser-compatible display.

    Forces re-encoding of the image data, which converts non-web formats
    (like TIFF) to displayable format when Gradio serializes the output.
    Also downscales very large images for preview to avoid browser limits.

    Returns:
        Tuple of (converted_image, original_size) where original_size is (width, height).
    """
    from PIL import Image as PILImage

    if image is None:
        return None, None

    # Handle PIL Image
    if isinstance(image, PILImage.Image):
        original_size = image.size  # (width, height)
        w, h = original_size

        # Convert to RGB if needed (handles RGBA, L, P modes)
        if image.mode not in ("RGB",):
            image = image.convert("RGB")

        # Limit preview size to avoid browser/websocket payload limits
        max_preview_size = 2048
        if max(h, w) > max_preview_size:
            scale = max_preview_size / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            image = image.resize((new_w, new_h), PILImage.Resampling.LANCZOS)

        return image, original_size

    # Handle numpy array (fallback)
    if hasattr(image, "shape"):
        original_size = (image.shape[1], image.shape[0])  # (width, height)
        return image.copy(), original_size

    return image, None


def analyze_image(
    original_image,
    mode: str = "Tiling (Accurate)",
    gsd: float = DEFAULT_GSD_M,
    original_size: tuple[int, int] | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None, str, str, str]:
    """
    Main processing pipeline for building segmentation.

    Args:
        original_image: Original full-size image (PIL Image or numpy array).
        mode: Processing mode - 'Tiling (Accurate)' or 'Resize (Fast)'.
        gsd: Ground Sampling Distance in meters/pixel for the ORIGINAL image.
        original_size: Original image size (width, height) before any resizing.
            Used to correct GSD if image was resized for display.

    Returns:
        Tuple of (mask, overlay, area_stat, coverage_stat, count_stat).
        Returns empty results (None, None, "—", "—", "—") on invalid input or error.
    """
    from PIL import Image as PILImage

    image = original_image

    # Проверка входных данных
    if image is None:
        return None, None, "—", "—", "—"

    try:
        # Convert PIL to numpy if needed
        if isinstance(image, PILImage.Image):
            image = np.array(image.convert("RGB"))

        # Загрузка и валидация
        image = load_image(image)

        if not validate_image(image):
            return None, None, "—", "—", "—"

        # Нормализация
        image = normalize_image(image)

        # Определяем режим обработки
        inference_mode: Literal["tiling", "resize"] = (
            "tiling" if "Tiling" in mode else "resize"
        )

        # Инференс
        mask = predict_mask(image, mode=inference_mode)

        # Корректируем GSD если изображение было ресайзнуто
        # GSD задаётся для ОРИГИНАЛЬНОГО изображения, но мы работаем с ресайзнутым
        current_h, current_w = image.shape[:2]
        effective_gsd = gsd

        if original_size is not None:
            orig_w, orig_h = original_size
            # Если размеры изменились, пересчитываем GSD
            if orig_w != current_w or orig_h != current_h:
                # GSD масштабируется пропорционально изменению размера
                scale_w = orig_w / current_w
                scale_h = orig_h / current_h
                # Используем среднее для случая неравномерного масштабирования
                scale = (scale_w + scale_h) / 2
                effective_gsd = gsd * scale

        # Статистика с учётом скорректированного GSD
        area = calculate_area(mask, gsd_m=effective_gsd)

        # Coverage — это процент площади, он НЕ зависит от ресайза
        # (соотношение building_pixels / total_pixels сохраняется при пропорциональном ресайзе)
        coverage = calculate_coverage_percent(mask)

        building_count = count_buildings(mask)

        # Визуализация
        overlay = create_overlay(image, mask)

        # Форматированные статистики
        area_stat = f"{area:,.0f} m²"
        coverage_stat = f"{coverage:.1f}%"
        count_stat = str(building_count)

        return mask, overlay, area_stat, coverage_stat, count_stat

    except Exception:
        logger.exception("Failed to analyze image")
        return None, None, "—", "—", "—"


def create_app() -> gr.Blocks:
    """
    Create and configure the Gradio application.

    Returns:
        Configured Gradio Blocks application.
    """
    with gr.Blocks(title="Building Segmentation") as app:

        # Компактный заголовок
        gr.HTML("""
            <div style="
                display: flex;
                align-items: center;
                justify-content: space-between;
                padding: 16px 0;
                margin-bottom: 8px;
                border-bottom: 1px solid #e2e8f0;
            ">
                <div style="display: flex; align-items: center; gap: 12px;">
                    <div style="
                        width: 36px;
                        height: 36px;
                        background: #0f172a;
                        border-radius: 8px;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        color: white;
                        font-size: 1rem;
                    ">▣</div>
                    <div>
                        <h1 style="margin: 0; font-size: 1.125rem;
                            font-weight: 600; color: #0f172a;">
                            Building Segmentation
                        </h1>
                        <p style="margin: 0; font-size: 0.75rem; color: #64748b;">
                            Aerial imagery analysis • Building area estimation
                        </p>
                    </div>
                </div>
            </div>
        """)

        # Статистика в одну строку
        with gr.Row():
            with gr.Column(scale=1, min_width=120):
                gr.HTML("""
                    <div style="text-align: left; padding: 24px 0 8px 0;">
                        <div class="stat-label">Total Area</div>
                    </div>
                """)
                area_output = gr.Textbox(
                    value="—",
                    show_label=False,
                    interactive=False,
                    container=False,
                    elem_classes=["stat-value"],
                )

            with gr.Column(scale=1, min_width=120):
                gr.HTML("""
                    <div style="text-align: left; padding: 24px 0 8px 0;">
                        <div class="stat-label">Coverage</div>
                    </div>
                """)
                coverage_output = gr.Textbox(
                    value="—",
                    show_label=False,
                    interactive=False,
                    container=False,
                    elem_classes=["stat-value"],
                )

            with gr.Column(scale=1, min_width=120):
                gr.HTML("""
                    <div style="text-align: left; padding: 24px 0 8px 0;">
                        <div class="stat-label">Buildings</div>
                    </div>
                """)
                count_output = gr.Textbox(
                    value="—",
                    show_label=False,
                    interactive=False,
                    container=False,
                    elem_classes=["stat-value"],
                )

        # State для хранения оригинального размера изображения (до ресайза)
        original_image_size = gr.State(value=None)

        # Основной контент: три колонки изображений
        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                gr.HTML('<div class="section-label">Input</div>')
                input_image = gr.Image(
                    label="",
                    type="pil",
                    sources=["upload", "clipboard"],
                    show_label=False,
                    height=320,
                )

            with gr.Column(scale=1):
                gr.HTML('<div class="section-label">Segmentation</div>')
                output_mask = gr.Image(
                    label="",
                    type="numpy",
                    show_label=False,
                    height=320,
                    format="webp",
                )

            with gr.Column(scale=1):
                gr.HTML('<div class="section-label">Overlay</div>')
                output_overlay = gr.Image(
                    label="",
                    type="numpy",
                    show_label=False,
                    height=320,
                    format="webp",
                )

        # Нижняя строка: Examples слева, настройки справа
        with gr.Row(equal_height=False):
            # Examples слева (под INPUT)
            with gr.Column(scale=1):
                # Загружаем примеры с HF Hub (с кешированием)
                example_paths = _download_examples()
                if example_paths:
                    gr.Examples(
                        examples=example_paths,
                        inputs=input_image,
                        label="Examples (click to load)",
                    )

            # Настройки и кнопки справа (под SEGMENTATION и OVERLAY)
            with gr.Column(scale=2):
                with gr.Row():
                    mode_dropdown = gr.Dropdown(
                        choices=["Tiling (Accurate)", "Resize (Fast)"],
                        value="Tiling (Accurate)",
                        label="Mode",
                        scale=1,
                    )
                    gsd_input = gr.Number(
                        value=DEFAULT_GSD_M,
                        label="GSD (m/px)",
                        minimum=0.01,
                        maximum=10.0,
                        step=0.01,
                        scale=1,
                    )

                with gr.Row():
                    analyze_btn = gr.Button(
                        "Analyze",
                        variant="primary",
                        size="lg",
                        elem_classes=["primary-btn"],
                        scale=1,
                    )
                    auto_gsd_btn = gr.Button(
                        "Calculate Scale",
                        size="lg",
                        scale=1,
                    )

        # Event handlers
        # Force conversion of non-web formats (TIFF, etc.) to displayable format
        # Also saves original image size before any resizing
        input_image.upload(
            fn=_convert_for_display,
            inputs=[input_image],
            outputs=[input_image, original_image_size],
        )

        analyze_btn.click(
            fn=analyze_image,
            inputs=[input_image, mode_dropdown, gsd_input, original_image_size],
            outputs=[
                output_mask, output_overlay,
                area_output, coverage_output, count_output,
            ],
        )

        auto_gsd_btn.click(
            fn=auto_detect_gsd,
            inputs=[input_image],
            outputs=[gsd_input],
        )

    return app


def _create_theme() -> gr.themes.Base:
    """Create the application theme."""
    return gr.themes.Base(
        primary_hue="slate",
        secondary_hue="slate",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("Inter"),
    ).set(
        body_background_fill="#f8fafc",
        block_background_fill="white",
        block_border_width="1px",
        block_border_color="#e2e8f0",
        block_shadow="none",
        block_radius="8px",
        button_primary_background_fill="#0f172a",
        button_primary_background_fill_hover="#1e293b",
        button_primary_text_color="white",
        input_radius="6px",
        input_border_color="#e2e8f0",
    )


def launch() -> None:
    """Launch the Gradio application."""
    app = create_app()
    theme = _create_theme()
    # Allow Gradio to serve example images from the examples directory
    app.launch(allowed_paths=[str(EXAMPLES_DIR)], theme=theme, css=CUSTOM_CSS)


if __name__ == "__main__":
    launch()
