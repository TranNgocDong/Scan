from __future__ import annotations

import cv2
import numpy as np


def resize_max_side(image: np.ndarray, max_side: int = 1200) -> tuple[np.ndarray, float]:
    """Resize image so the largest side is at most max_side.

    Returns the resized image and the scale factor from original to resized.
    """
    # Giu ty le anh goc, chi thu nho neu anh qua lon de pipeline nhanh hon.
    h, w = image.shape[:2]
    largest = max(h, w)
    if largest <= max_side:
        return image.copy(), 1.0

    scale = max_side / largest
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


def to_gray(image: np.ndarray) -> np.ndarray:
    """Convert BGR/RGB/grayscale image to grayscale."""
    # OCR va cac buoc tien xu ly thuong chi can mot kenh sang.
    if image.ndim == 2:
        return image.copy()
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def clahe_contrast(gray: np.ndarray, clip_limit: float = 2.0, tile_grid_size: int = 8) -> np.ndarray:
    """Improve local contrast with CLAHE."""
    # CLAHE tang tuong phan theo tung vung nho, tot cho anh sang khong deu.
    clahe = cv2.createCLAHE(
        clipLimit=float(clip_limit),
        tileGridSize=(int(tile_grid_size), int(tile_grid_size)),
    )
    return clahe.apply(gray)


def denoise(gray: np.ndarray, blur_ksize: int = 5) -> np.ndarray:
    """Denoise image before edge detection."""
    # Lam muot nhe de Canny bot bat nhieu lat vat.
    blur_ksize = int(blur_ksize)
    if blur_ksize <= 1:
        return gray.copy()
    if blur_ksize % 2 == 0:
        blur_ksize += 1
    return cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)


def adaptive_binarize(
    gray: np.ndarray,
    block_size: int = 31,
    c_value: int = 11,
) -> np.ndarray:
    """Binarize document using adaptive Gaussian threshold."""
    # block_size phai la so le; nguong duoc tinh cuc bo theo tung vung.
    block_size = int(block_size)
    if block_size % 2 == 0:
        block_size += 1
    block_size = max(block_size, 3)

    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        int(c_value),
    )


def otsu_binarize(gray: np.ndarray) -> np.ndarray:
    """Binarize document using Otsu threshold."""
    # Otsu chon nguong tu dong dua tren histogram toan anh.
    _, out = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return out


def clean_binary(binary: np.ndarray, kernel_size: int = 2, close_first: bool = True) -> np.ndarray:
    """Clean small noise in binarized text image with morphology."""
    # Kernel size = 1 nghia la hau nhu khong lam gi de tranh bao mon net chu.
    kernel_size = max(1, int(kernel_size))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    if kernel_size <= 1:
        return binary.copy()

    # Close truoc de lap cac lo nho trong net chu, roi open de bo cham nhieu.
    if close_first:
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
    else:
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    return cleaned


def remove_dark_borders(image: np.ndarray, threshold: int = 25, margin: int = 6) -> np.ndarray:
    """Crop away dark scanner/screenshot borders from a mostly white document image.

    The sample image is a screenshot of a page and contains black horizontal/vertical
    separators. This function keeps the largest region that is not dominated by very
    dark border pixels. It is used as a fallback cleanup step before OCR.
    """
    # Muc tieu la bo cac dai den o mep tren/duoi/trai/phai neu anh la screenshot.
    gray = to_gray(image)
    dark = gray < int(threshold)
    h, w = gray.shape

    row_dark_ratio = dark.mean(axis=1)
    col_dark_ratio = dark.mean(axis=0)

    good_rows = np.where(row_dark_ratio < 0.35)[0]
    good_cols = np.where(col_dark_ratio < 0.35)[0]
    if good_rows.size == 0 or good_cols.size == 0:
        return image.copy()

    y0 = max(0, int(good_rows[0]) - margin)
    y1 = min(h, int(good_rows[-1]) + margin + 1)
    x0 = max(0, int(good_cols[0]) - margin)
    x1 = min(w, int(good_cols[-1]) + margin + 1)
    return image[y0:y1, x0:x1].copy()


def crop_light_page_region(image: np.ndarray, margin: int = 10) -> np.ndarray:
    """Crop to the largest bright paper-like region after perspective correction.

    This removes gray floor/background that may remain around book pages. The crop
    is conservative: if it cannot find a large enough paper region, it returns the
    original image so the pipeline does not destroy the scan.
    """
    gray = to_gray(image)
    h, w = gray.shape
    if h <= 0 or w <= 0:
        return image.copy()

    # Paper is usually among the brighter pixels; the percentile adapts to each photo.
    threshold = max(135, int(np.percentile(gray, 55)))
    mask = (gray >= threshold).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if num_labels <= 1:
        return image.copy()

    image_area = h * w
    best_label = None
    best_area = 0
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area > best_area:
            best_label = label
            best_area = area

    if best_label is None or best_area < image_area * 0.18:
        return image.copy()

    x = int(stats[best_label, cv2.CC_STAT_LEFT])
    y = int(stats[best_label, cv2.CC_STAT_TOP])
    bw = int(stats[best_label, cv2.CC_STAT_WIDTH])
    bh = int(stats[best_label, cv2.CC_STAT_HEIGHT])

    # Avoid over-cropping to a tiny bright patch such as a blank margin.
    if bw < w * 0.30 or bh < h * 0.30:
        return image.copy()

    x0 = max(0, x - margin)
    y0 = max(0, y - margin)
    x1 = min(w, x + bw + margin)
    y1 = min(h, y + bh + margin)
    return image[y0:y1, x0:x1].copy()


def crop_text_region(binary_or_gray: np.ndarray, padding: int = 28) -> np.ndarray:
    """Crop to the bounding box of likely text pixels.

    Works on white-background document images by finding non-white connected text
    regions. This improves OCR when the page detector falls back to the full image
    and leaves large margins or screenshot separators.
    """
    # Dung cac cum pixel toi de tim vung co chu thay vi giu ca trang trang.
    gray = to_gray(binary_or_gray)
    h, w = gray.shape

    # Text is dark on a light page. Ignore tiny specks and very large page borders.
    mask = gray < 210
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    keep = np.zeros_like(mask, dtype=bool)
    image_area = h * w

    for label in range(1, num_labels):
        x, y, bw, bh, area = stats[label]
        # Bo nhieu qua nho va bo nhung khoi qua lon vi thuong khong phai chu.
        if area < 3:
            continue
        if area > image_area * 0.08:
            continue
        if bw > w * 0.95 or bh > h * 0.25:
            continue
        keep[labels == label] = True

    ys, xs = np.where(keep)
    if ys.size == 0 or xs.size == 0:
        return binary_or_gray.copy()

    y0 = max(0, int(ys.min()) - padding)
    y1 = min(h, int(ys.max()) + padding + 1)
    x0 = max(0, int(xs.min()) - padding)
    x1 = min(w, int(xs.max()) + padding + 1)

    crop_h = y1 - y0
    crop_w = x1 - x0
    crop_area = crop_h * crop_w
    # Chan loi crop nham thanh mot dai rat hep hoac gan nhu trang toan bo.
    # Voi anh sach that, mot crop qua hep thuong la mep giay/duong ke, khong phai vung chu.
    # A real text block on a page should not collapse to a very thin horizontal
    # strip. If that happens, it is usually floor texture/page shadow near the
    # border, so keeping the original page is safer and easier to explain.
    if crop_w < w * 0.20 or crop_h < h * 0.18 or crop_area < image_area * 0.03:
        return binary_or_gray.copy()

    return binary_or_gray[y0:y1, x0:x1].copy()


def add_white_margin(image: np.ndarray, margin: int = 30) -> np.ndarray:
    """Add a white margin around an OCR image so Tesseract sees complete lines."""
    # Margin trang giup Tesseract khong bi cat sat mep dong chu.
    if margin <= 0:
        return image.copy()
    value = 255 if image.ndim == 2 else (255, 255, 255)
    return cv2.copyMakeBorder(
        image,
        margin,
        margin,
        margin,
        margin,
        cv2.BORDER_CONSTANT,
        value=value,
    )
