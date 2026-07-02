from __future__ import annotations

import cv2
import numpy as np


def contour_area_ratio(contour: np.ndarray | None, image_shape: tuple[int, int] | tuple[int, int, int]) -> float:
    """Return contour area divided by image area."""
    if contour is None:
        return 0.0
    h, w = image_shape[:2]
    if h <= 0 or w <= 0:
        return 0.0
    pts = contour.reshape(-1, 1, 2).astype(np.float32)
    return float(cv2.contourArea(pts)) / float(h * w)


def canny_edges(gray: np.ndarray, low: int = 60, high: int = 160, dilate_iter: int = 1) -> np.ndarray:
    """Detect document edges with Canny and optional dilation."""
    # Do bien truoc, roi no nhe canh de contour giay de khep kin hon.
    edges = cv2.Canny(gray, int(low), int(high))
    if dilate_iter > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.dilate(edges, kernel, iterations=int(dilate_iter))
    return edges


def find_document_contour(edges: np.ndarray, min_area_ratio: float = 0.15) -> np.ndarray | None:
    """Find largest quadrilateral contour likely to be the paper boundary."""
    # Chi lay contour ngoai cung vi vien giay thuong la khung lon nhat.
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    h, w = edges.shape[:2]
    min_area = float(h * w * min_area_ratio)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for contour in contours:
        # Bo contour qua nho vi do thuong la nhieu hoac chi tiet chu.
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        # Thu nhieu muc xap xi de xem contour co gan giong tu giac khong.
        perimeter = cv2.arcLength(contour, True)
        for eps_ratio in (0.02, 0.03, 0.04, 0.06, 0.08):
            approx = cv2.approxPolyDP(contour, eps_ratio * perimeter, True)
            if len(approx) == 4:
                return approx.reshape(4, 2).astype(np.float32)
    return None


def find_bright_document_contour(image: np.ndarray, min_area_ratio: float = 0.12) -> np.ndarray | None:
    """Find a likely page contour by segmenting bright paper from darker background.

    Canny works well when page borders are sharp, but book photos often contain
    floor lines, fingers, and shadows that create stronger edges than the paper.
    This fallback uses the fact that most printed pages are brighter than the
    background, then returns a rotated rectangle around the largest bright region.
    """
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    h, w = gray.shape[:2]
    min_area = float(h * w * min_area_ratio)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        rect = cv2.minAreaRect(contour)
        (_, _), (rw, rh), _ = rect
        longer = max(rw, rh)
        shorter = min(rw, rh)
        if longer <= 1 or shorter / longer < 0.18:
            continue

        box = cv2.boxPoints(rect)
        return box.astype(np.float32)
    return None


def fallback_page_corners(image_shape: tuple[int, int] | tuple[int, int, int]) -> np.ndarray:
    """Use whole image corners when no document contour is found."""
    # Khi khong tim duoc contour tin cay thi dung luon khung anh de pipeline van chay.
    h, w = image_shape[:2]
    return np.array(
        [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]],
        dtype=np.float32,
    )


def draw_contour_overlay(image: np.ndarray, contour: np.ndarray | None) -> np.ndarray:
    """Draw detected document contour for lab-report intermediate output."""
    # Lop phu nay chi de nguoi doc nhin thay contour da tim duoc o buoc truoc.
    overlay = image.copy()
    if contour is not None:
        pts = contour.reshape(-1, 1, 2).astype(np.int32)
        cv2.polylines(overlay, [pts], isClosed=True, color=(0, 0, 255), thickness=3)
    return overlay
