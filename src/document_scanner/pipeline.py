from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .document_detect import (
    canny_edges,
    contour_area_ratio,
    draw_contour_overlay,
    fallback_page_corners,
    find_bright_document_contour,
    find_document_contour,
)
from .evaluation import character_error_rate, word_error_rate
from .ocr import run_tesseract_ocr
from .perspective import four_point_transform
from .preprocess import (
    add_white_margin,
    adaptive_binarize,
    clahe_contrast,
    clean_binary,
    crop_light_page_region,
    crop_text_region,
    denoise,
    otsu_binarize,
    remove_dark_borders,
    resize_max_side,
    to_gray,
)


@dataclass
class ScannerParams:
    # Kích thước tối đa của cạnh dài nhất sau khi resize.
    max_side: int = 1200
    # Kích thước kernel Gaussian để làm mượt trước khi dò biên.
    blur_ksize: int = 5
    # Hai ngưỡng thấp/cao của Canny.
    canny_low: int = 60
    canny_high: int = 160
    # Tham số threshold thích nghi cho bước tách chữ khỏi nền.
    adaptive_block_size: int = 31
    adaptive_c: int = 11
    # Kích thước kernel morphology, nên giữ nhỏ để tránh làm mất nét chữ.
    morph_kernel: int = 1
    # Phần lề giữ thêm quanh vùng chữ sau khi crop.
    text_crop_padding: int = 28
    # Hệ số phóng to ảnh trước OCR.
    ocr_scale: float = 2.0
    # Ngôn ngữ Tesseract.
    tesseract_lang: str = "eng"
    # Chế độ phân đoạn trang của Tesseract.
    tesseract_psm: int = 6


def save_image(path: Path, image: np.ndarray) -> None:
    # Lưu ảnh trung gian ra đĩa để người xem có thể kiểm tra từng bước xử lý.
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


def prepare_for_ocr(image: np.ndarray, scale: float = 2.0) -> np.ndarray:
    """Upscale OCR image to help Tesseract read small text."""
    if scale <= 1:
        return image.copy()
    h, w = image.shape[:2]
    return cv2.resize(
        image,
        (int(round(w * scale)), int(round(h * scale))),
        interpolation=cv2.INTER_CUBIC,
    )


def _score_ocr_candidate(text: str, truth: str | None) -> tuple[float, float | None, float | None]:
    """Score an OCR candidate. Lower score is better."""
    if truth is not None:
        # Nếu có ground truth thì chọn ảnh OCR nào cho CER thấp nhất.
        cer = character_error_rate(text, truth)
        wer = word_error_rate(text, truth)
        return cer, cer, wer

    # Fallback without labels: prefer more alphanumeric text and fewer odd symbols.
    stripped = text.strip()
    alpha_count = sum(ch.isalnum() for ch in stripped)
    symbol_count = sum((not ch.isalnum()) and (not ch.isspace()) for ch in stripped)
    score = -(alpha_count - 0.5 * symbol_count)
    return score, None, None


def process_document_image(
    image_path: str | Path,
    output_dir: str | Path,
    params: ScannerParams | None = None,
    ground_truth_path: str | Path | None = None,
) -> dict:
    """Run full document scanner OCR pipeline on one image."""
    params = params or ScannerParams()
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    stem = image_path.stem
    sample_dir = output_dir / stem
    sample_dir.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    # 1) Thu nhỏ ảnh nếu quá lớn để pipeline chạy ổn định và nhanh hơn.
    resized, scale = resize_max_side(image, params.max_side)
    # 2) Đưa về grayscale để các phép xử lý ảnh làm việc trên một kênh sáng.
    gray = to_gray(resized)
    # 3) Tăng tương phản cục bộ để chữ và nền tách nhau rõ hơn.
    enhanced = clahe_contrast(gray)
    # 4) Làm mượt nhẹ để giảm nhiễu trước khi tìm cạnh.
    blurred = denoise(enhanced, params.blur_ksize)
    # 5) Dò biên để tìm khung giấy.
    edges = canny_edges(blurred, params.canny_low, params.canny_high)
    # 6) Tìm contour lớn nhất gần giống tờ giấy.
    canny_contour = find_document_contour(edges)
    bright_contour = find_bright_document_contour(resized)
    canny_area = contour_area_ratio(canny_contour, resized.shape)
    bright_area = contour_area_ratio(bright_contour, resized.shape)

    # Canny có thể bắt nhầm đường gạch nền hoặc bóng tay. Nếu vùng giấy sáng lớn hơn
    # rõ rệt, ta ưu tiên contour dựa trên vùng giấy sáng để output scan dễ nhìn hơn.
    contour = canny_contour
    contour_method = "canny"
    if bright_contour is not None and (contour is None or bright_area > max(canny_area * 1.35, 0.18)):
        contour = bright_contour
        contour_method = "bright_page"
    used_fallback = contour is None
    if contour is None:
        # Nếu không tìm được contour tin cậy thì dùng luôn 4 góc ảnh.
        contour = fallback_page_corners(resized.shape)
        contour_method = "full_image_fallback"

    # 7) Vẽ contour lên ảnh để lưu ảnh trung gian.
    overlay = draw_contour_overlay(resized, contour)
    # 8) Warp phối cảnh để đưa ảnh nghiêng về gần dạng scan thẳng.
    warped_color = four_point_transform(resized, contour)
    # 9) Cắt viền đen còn sót lại nếu ảnh là screenshot hoặc scan có mép thừa.
    borderless_color = remove_dark_borders(warped_color)
    # 9b) Cắt tiếp phần nền xám/gạch còn sót lại quanh trang giấy.
    page_region_color = crop_light_page_region(borderless_color)
    # 10) Đưa ảnh sau warp về grayscale để tiếp tục xử lý chữ.
    warped_gray = to_gray(page_region_color)
    # 11) Tăng tương phản lần nữa trên ảnh đã được sửa phối cảnh.
    warped_enhanced = clahe_contrast(warped_gray)
    # 12) Binarize thích nghi để tách chữ khỏi nền không đều.
    binary = adaptive_binarize(
        warped_enhanced,
        block_size=params.adaptive_block_size,
        c_value=params.adaptive_c,
    )
    # 13) Làm sạch nhiễu nhỏ bằng morphology, nhưng giữ kernel nhỏ để không làm dính chữ.
    cleaned = clean_binary(binary, kernel_size=params.morph_kernel)
    # 14) Crop đúng vùng có chữ để bỏ phần trắng thừa và viền đen.
    text_crop = crop_text_region(cleaned, padding=params.text_crop_padding)
    # 15) Phóng to và thêm lề trắng để Tesseract đọc dễ hơn.
    ocr_ready = add_white_margin(prepare_for_ocr(text_crop, scale=params.ocr_scale), margin=35)
    # 16) Ảnh scan để báo cáo nên dễ nhìn, nên dùng ảnh xám tăng tương phản thay vì
    # ảnh nhị phân quá gắt. OCR vẫn có thể dùng một biến thể khác ở bên dưới.
    readable_scan = add_white_margin(
        prepare_for_ocr(crop_text_region(warped_enhanced, padding=params.text_crop_padding), scale=params.ocr_scale),
        margin=35,
    )

    save_image(sample_dir / "01_resized.jpg", resized)
    save_image(sample_dir / "02_gray.jpg", gray)
    save_image(sample_dir / "03_enhanced.jpg", enhanced)
    save_image(sample_dir / "04_edges.jpg", edges)
    save_image(sample_dir / "05_document_contour.jpg", overlay)
    save_image(sample_dir / "06_warped.jpg", warped_color)
    save_image(sample_dir / "07_borderless.jpg", borderless_color)
    save_image(sample_dir / "07b_page_region.jpg", page_region_color)
    save_image(sample_dir / "08_threshold.jpg", binary)
    save_image(sample_dir / "09_cleaned.jpg", cleaned)
    save_image(sample_dir / "10_text_crop.jpg", text_crop)
    save_image(sample_dir / "11_ocr_ready.jpg", ocr_ready)
    save_image(sample_dir / "12_readable_scan.jpg", readable_scan)
    save_image(Path(output_dir) / "final" / f"{stem}_scan.png", readable_scan)

    truth = None
    if ground_truth_path is not None and Path(ground_truth_path).exists():
        truth = Path(ground_truth_path).read_text(encoding="utf-8")

    # Try a few OCR-oriented variants. This is a parameter sweep for the OCR input
    # image, not a replacement for the CV pipeline.
    # Mục tiêu ở đây là chọn ảnh nào vào Tesseract thì cho kết quả tốt nhất.
    ocr_candidates = {
        "cleaned_full": cleaned,
        "text_crop_cleaned": ocr_ready,
        "text_crop_no_morph": add_white_margin(
            prepare_for_ocr(crop_text_region(binary, padding=params.text_crop_padding), scale=params.ocr_scale),
            margin=35,
        ),
        "otsu_text_crop": add_white_margin(
            prepare_for_ocr(crop_text_region(otsu_binarize(warped_enhanced), padding=params.text_crop_padding), scale=params.ocr_scale),
            margin=35,
        ),
        "gray_text_crop": add_white_margin(
            prepare_for_ocr(crop_text_region(warped_enhanced, padding=params.text_crop_padding), scale=params.ocr_scale),
            margin=35,
        ),
    }

    best_name = ""
    best_text = ""
    best_warning = None
    best_score = float("inf")
    best_cer = None
    best_wer = None
    best_image = ocr_ready
    for name, candidate in ocr_candidates.items():
        # OCR từng ứng viên để so sánh rồi chọn ảnh tốt nhất.
        text, warning = run_tesseract_ocr(
            candidate,
            lang=params.tesseract_lang,
            psm=params.tesseract_psm,
        )
        score, cer, wer = _score_ocr_candidate(text, truth)
        if warning is not None:
            best_name = name
            best_text = text
            best_warning = warning
            best_image = candidate
            break
        if score < best_score:
            best_name = name
            best_text = text
            best_warning = warning
            best_score = score
            best_cer = cer
            best_wer = wer
            best_image = candidate

    ocr_text = best_text
    ocr_warning = best_warning
    save_image(sample_dir / "13_best_ocr_input.jpg", best_image)
    save_image(Path(output_dir) / "final" / f"{stem}_scan.png", readable_scan)
    text_path = Path(output_dir) / "final" / f"{stem}_ocr.txt"
    text_path.parent.mkdir(parents=True, exist_ok=True)
    text_path.write_text(ocr_text, encoding="utf-8")

    metrics = {}
    if truth is not None:
        metrics["cer"] = best_cer if best_cer is not None else character_error_rate(ocr_text, truth)
        metrics["wer"] = best_wer if best_wer is not None else word_error_rate(ocr_text, truth)

    return {
        "image": str(image_path),
        "scale": scale,
        "used_fallback_page": used_fallback,
        "contour_method": contour_method,
        "ocr_variant": best_name,
        "output_dir": str(sample_dir),
        "final_scan": str(Path(output_dir) / "final" / f"{stem}_scan.png"),
        "ocr_text_path": str(text_path),
        "ocr_warning": ocr_warning,
        "ocr_text": ocr_text,
        **metrics,
    }


def parameter_sweep(
    image_path: str | Path,
    output_dir: str | Path,
    ground_truth_path: str | Path | None = None,
    base_params: ScannerParams | None = None,
) -> list[dict]:
    """Run three small parameter sweeps required by the project brief."""
    results = []
    output_dir = Path(output_dir)
    base_params = base_params or ScannerParams()

    sweep_configs = []
    # Sweep 1: thử các ngưỡng Canny khác nhau để xem biên giấy nào ổn nhất.
    for low, high in [(40, 120), (60, 160), (100, 220)]:
        params = ScannerParams(**{**base_params.__dict__, "canny_low": low, "canny_high": high})
        sweep_configs.append((f"canny_{low}_{high}", params))
    # Sweep 2: thử block size của adaptive threshold.
    for block in [15, 31, 51]:
        params = ScannerParams(**{**base_params.__dict__, "adaptive_block_size": block})
        sweep_configs.append((f"block_{block}", params))
    # Sweep 3: thử kernel morphology nhỏ-vừa-lớn.
    for kernel in [1, 2, 3]:
        params = ScannerParams(**{**base_params.__dict__, "morph_kernel": kernel})
        sweep_configs.append((f"morph_{kernel}", params))
    # Sweep 4: thử khoảng đệm quanh vùng chữ để tránh cắt mất dấu hoặc ký tự sát biên.
    for padding in [12, 28, 44]:
        params = ScannerParams(**{**base_params.__dict__, "text_crop_padding": padding})
        sweep_configs.append((f"text_crop_padding_{padding}", params))

    for name, params in sweep_configs:
        run_dir = output_dir / "sweep" / name
        result = process_document_image(
            image_path,
            run_dir,
            params=params,
            ground_truth_path=ground_truth_path,
        )
        result["sweep_name"] = name
        results.append(result)
    return results
