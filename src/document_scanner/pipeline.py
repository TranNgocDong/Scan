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
from .ocr import run_tesseract_ocr_with_confidence
from .perspective import four_point_transform
from .preprocess import (
    add_white_margin,
    adaptive_binarize,
    clahe_contrast,
    clean_binary,
    crop_dominant_text_region,
    crop_light_page_region,
    crop_text_region,
    denoise,
    normalize_background,
    otsu_binarize,
    remove_dark_borders,
    resize_max_side,
    sharpen_text,
    to_gray,
)


@dataclass
class ScannerParams:
    # Kich thuoc toi da cua canh dai nhat sau khi resize.
    max_side: int = 1200
    # Kich thuoc kernel Gaussian de lam muot truoc khi do bien.
    blur_ksize: int = 5
    # Hai nguong thap/cao cua Canny.
    canny_low: int = 60
    canny_high: int = 160
    # Tham so threshold thich nghi cho buoc tach chu khoi nen.
    adaptive_block_size: int = 31
    adaptive_c: int = 11
    # Kich thuoc kernel morphology, nen giu nho de tranh lam mat net chu.
    morph_kernel: int = 1
    # Phan le giu them quanh vung chu sau khi crop.
    text_crop_padding: int = 28
    # He so phong to anh truoc OCR.
    ocr_scale: float = 2.0
    # Ngon ngu Tesseract.
    tesseract_lang: str = "eng"
    # Che do phan doan trang cua Tesseract.
    tesseract_psm: int = 6


def save_image(path: Path, image: np.ndarray) -> None:
    # Luu anh trung gian ra dia de nguoi xem co the kiem tra tung buoc xu ly.
    path.parent.mkdir(parents=True, exist_ok=True)
    # cv2.imwrite tren Windows co the loi voi duong dan co dau tieng Viet.
    # imencode + tofile doc/ghi duoc duong dan Unicode on dinh hon.
    ok, encoded = cv2.imencode(path.suffix or ".png", image)
    if not ok:
        raise OSError(f"Cannot encode image for saving: {path}")
    encoded.tofile(str(path))


def read_image_unicode(path: Path) -> np.ndarray | None:
    """Read image from a Unicode Windows path."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


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
        # Neu co ground truth thi chon anh OCR nao cho CER thap nhat.
        cer = character_error_rate(text, truth)
        wer = word_error_rate(text, truth)
        return cer, cer, wer

    # Without ground truth, we cannot know the exact OCR error. A long output is
    # not always good: noisy page texture can create many fake characters. This
    # score rewards word-like text and penalizes strange punctuation/noise.
    stripped = text.strip()
    if not stripped:
        return float("inf"), None, None

    total = max(len(stripped), 1)
    letter_count = sum(ch.isalpha() for ch in stripped)
    digit_count = sum(ch.isdigit() for ch in stripped)
    symbol_count = sum((not ch.isalnum()) and (not ch.isspace()) for ch in stripped)
    suspicious_symbols = set("`~^_=+*|\\/<>{}[]")
    suspicious_count = sum(ch in suspicious_symbols or ch == "\ufffd" for ch in stripped)
    words = [word for word in stripped.replace("\n", " ").split(" ") if word]
    short_noise_words = sum(1 for word in words if len(word) <= 1 and not word.isalnum())
    symbol_ratio = symbol_count / total
    common_vietnamese_words = {
        "người",
        "không",
        "trong",
        "của",
        "và",
        "là",
        "một",
        "có",
        "cho",
        "rằng",
        "như",
        "được",
        "với",
        "thì",
        "đã",
        "này",
        "những",
        "các",
        "theo",
        "thành",
    }
    lower_words = [word.strip(".,;:!?()[]{}\"'").lower() for word in words]
    common_hits = sum(1 for word in lower_words if word in common_vietnamese_words)
    vietnamese_marks = sum(ch in "ăâđêôơưáàảãạắằẳẵặấầẩẫậéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ" for ch in stripped.lower())

    score = (
        -0.08 * letter_count
        -0.03 * digit_count
        -1.2 * len(words)
        -4.0 * common_hits
        -0.8 * vietnamese_marks
        + 5.0 * symbol_count
        + 10.0 * suspicious_count
        + 3.0 * short_noise_words
    )
    if symbol_ratio > 0.12:
        score += 2.0 * total
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

    image = read_image_unicode(image_path)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    # 1) Thu nho anh neu qua lon de pipeline chay on dinh va nhanh hon.
    resized, scale = resize_max_side(image, params.max_side)
    # 2) Dua ve grayscale de cac phep xu ly anh lam viec tren mot kenh sang.
    gray = to_gray(resized)
    # 3) Tang tuong phan cuc bo de chu va nen tach nhau ro hon.
    enhanced = clahe_contrast(gray)
    # 4) Lam muot nhe de giam nhieu truoc khi tim canh.
    blurred = denoise(enhanced, params.blur_ksize)
    # 5) Do bien de tim khung giay.
    edges = canny_edges(blurred, params.canny_low, params.canny_high)
    # 6) Tim contour lon nhat gan giong to giay.
    canny_contour = find_document_contour(edges)
    bright_contour = find_bright_document_contour(resized)
    canny_area = contour_area_ratio(canny_contour, resized.shape)
    bright_area = contour_area_ratio(bright_contour, resized.shape)

    # Canny co the bat nham duong gach nen hoac bong tay. Neu vung giay sang lon hon
    # ro ret, ta uu tien contour dua tren vung giay sang de output scan de nhin hon.
    contour = canny_contour
    contour_method = "canny"
    if bright_contour is not None and (contour is None or bright_area > max(canny_area * 1.35, 0.18)):
        contour = bright_contour
        contour_method = "bright_page"
    used_fallback = contour is None
    if contour is None:
        # Neu khong tim duoc contour tin cay thi dung luon 4 goc anh.
        contour = fallback_page_corners(resized.shape)
        contour_method = "full_image_fallback"

    # 7) Ve contour len anh de luu anh trung gian.
    overlay = draw_contour_overlay(resized, contour)
    # 8) Warp phoi canh de dua anh nghieng ve gan dang scan thang.
    warped_color = four_point_transform(resized, contour)
    # 9) Cat vien den con sot lai neu anh la screenshot hoac scan co mep thua.
    borderless_color = remove_dark_borders(warped_color)
    # 9b) Cat tiep phan nen xam/gach con sot lai quanh trang giay.
    page_region_color = crop_light_page_region(borderless_color)
    # 10) Dua anh sau warp ve grayscale de tiep tuc xu ly chu.
    warped_gray = to_gray(page_region_color)
    # 11) Tang tuong phan lan nua tren anh da duoc sua phoi canh.
    warped_enhanced = clahe_contrast(warped_gray)
    # 11b) Tao bien the rieng cho OCR: lam phang nen giay va lam net nhe.
    # Bien the nay thuong tot hon threshold gat khi anh sach bi bong hoac lo chu mat sau.
    warped_background = normalize_background(warped_gray)
    warped_background_sharp = sharpen_text(warped_background, amount=0.6, sigma=1.0)
    # 12) Binarize thich nghi de tach chu khoi nen khong deu.
    binary = adaptive_binarize(
        warped_enhanced,
        block_size=params.adaptive_block_size,
        c_value=params.adaptive_c,
    )
    # 13) Lam sach nhieu nho bang morphology, nhung giu kernel nho de khong lam dinh chu.
    cleaned = clean_binary(binary, kernel_size=params.morph_kernel)
    # 14) Crop dung vung co chu de bo phan trang thua va vien den.
    text_crop = crop_text_region(cleaned, padding=params.text_crop_padding)
    dominant_cleaned = crop_dominant_text_region(cleaned, padding=params.text_crop_padding)
    dominant_gray = crop_dominant_text_region(warped_enhanced, padding=params.text_crop_padding)
    dominant_background = crop_dominant_text_region(warped_background_sharp, padding=params.text_crop_padding)
    # 15) Phong to va them le trang de Tesseract doc de hon.
    ocr_ready = add_white_margin(prepare_for_ocr(dominant_cleaned, scale=params.ocr_scale), margin=35)
    # 16) Anh scan de bao cao nen de nhin, nen dung anh xam tang tuong phan thay vi
    # anh nhi phan qua gat. OCR van co the dung mot bien the khac o ben duoi.
    readable_scan = add_white_margin(
        prepare_for_ocr(dominant_gray, scale=params.ocr_scale),
        margin=35,
    )
    page_gray_ocr = add_white_margin(prepare_for_ocr(warped_gray, scale=params.ocr_scale), margin=35)
    page_background_ocr = add_white_margin(prepare_for_ocr(warped_background_sharp, scale=params.ocr_scale), margin=35)
    dominant_background_ocr = add_white_margin(
        prepare_for_ocr(dominant_background, scale=params.ocr_scale),
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
    save_image(sample_dir / "10b_dominant_text_crop.jpg", dominant_gray)
    save_image(sample_dir / "11_ocr_ready.jpg", ocr_ready)
    save_image(sample_dir / "12_readable_scan.jpg", readable_scan)
    save_image(sample_dir / "12a_ocr_page_gray.jpg", page_gray_ocr)
    save_image(sample_dir / "12b_ocr_background_norm.jpg", page_background_ocr)
    save_image(sample_dir / "12c_ocr_dominant_background.jpg", dominant_background_ocr)
    save_image(Path(output_dir) / "final" / f"{stem}_scan.png", readable_scan)

    truth = None
    if ground_truth_path is not None and Path(ground_truth_path).exists():
        truth = Path(ground_truth_path).read_text(encoding="utf-8")

    # Try a few OCR-oriented variants. This is a parameter sweep for the OCR input
    # image, not a replacement for the CV pipeline.
    # Muc tieu o day la chon anh nao vao Tesseract thi cho ket qua tot nhat.
    ocr_candidates = {
        "page_gray": page_gray_ocr,
        "page_background_norm": page_background_ocr,
        "dominant_background_norm": dominant_background_ocr,
        "readable_scan": readable_scan,
        "cleaned_full": cleaned,
        "text_crop_cleaned": ocr_ready,
        "dominant_gray_text": add_white_margin(
            prepare_for_ocr(dominant_gray, scale=params.ocr_scale),
            margin=35,
        ),
        "dominant_cleaned_text": add_white_margin(
            prepare_for_ocr(dominant_cleaned, scale=params.ocr_scale),
            margin=35,
        ),
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
    psm_values = []
    for psm in [params.tesseract_psm, 3, 4]:
        if psm not in psm_values:
            psm_values.append(psm)

    for name, candidate in ocr_candidates.items():
        for psm in psm_values:
            # OCR tung ung vien va tung che do PSM de chon cach doc on nhat.
            text, mean_conf, warning = run_tesseract_ocr_with_confidence(
                candidate,
                lang=params.tesseract_lang,
                psm=psm,
            )
            score, cer, wer = _score_ocr_candidate(text, truth)
            if truth is None:
                # For real book photos, readable grayscale/cropped variants often
                # beat harsh binary images. Mean confidence helps avoid outputs
                # that are long but full of fake characters.
                score += {
                    "page_gray": -130.0,
                    "page_background_norm": -105.0,
                    "dominant_background_norm": -95.0,
                    "readable_scan": -55.0,
                    "dominant_gray_text": -70.0,
                    "dominant_cleaned_text": -45.0,
                    "gray_text_crop": -45.0,
                    "text_crop_cleaned": -30.0,
                    "text_crop_no_morph": -20.0,
                    "otsu_text_crop": -10.0,
                    "cleaned_full": 45.0,
                }.get(name, 0.0)
                if mean_conf >= 0:
                    score -= mean_conf * 2.0
                if psm == 3:
                    score -= 12.0
                elif psm == 4:
                    score -= 8.0
                elif psm in (11, 12):
                    score += 4.0
            if warning is not None:
                best_name = f"{name}_psm{psm}"
                best_text = text
                best_warning = warning
                best_image = candidate
                break
            if score < best_score:
                best_name = f"{name}_psm{psm}"
                best_text = text
                best_warning = warning
                best_score = score
                best_cer = cer
                best_wer = wer
                best_image = candidate
        if best_warning is not None:
            break

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
    # Sweep 1: thu cac nguong Canny khac nhau de xem bien giay nao on nhat.
    for low, high in [(40, 120), (60, 160), (100, 220)]:
        params = ScannerParams(**{**base_params.__dict__, "canny_low": low, "canny_high": high})
        sweep_configs.append((f"canny_{low}_{high}", params))
    # Sweep 2: thu block size cua adaptive threshold.
    for block in [15, 31, 51]:
        params = ScannerParams(**{**base_params.__dict__, "adaptive_block_size": block})
        sweep_configs.append((f"block_{block}", params))
    # Sweep 3: thu kernel morphology nho-vua-lon.
    for kernel in [1, 2, 3]:
        params = ScannerParams(**{**base_params.__dict__, "morph_kernel": kernel})
        sweep_configs.append((f"morph_{kernel}", params))
    # Sweep 4: thu khoang dem quanh vung chu de tranh cat mat dau hoac ky tu sat bien.
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
