from __future__ import annotations

import cv2
import numpy as np


def order_points(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as top-left, top-right, bottom-right, bottom-left."""
    # Sắp xếp 4 điểm theo thứ tự cố định để warpPerspective không bị lật sai.
    pts = np.asarray(pts, dtype=np.float32)
    rect = np.zeros((4, 2), dtype=np.float32)

    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1).ravel()

    rect[0] = pts[np.argmin(sums)]
    rect[2] = pts[np.argmax(sums)]
    rect[1] = pts[np.argmin(diffs)]
    rect[3] = pts[np.argmax(diffs)]
    return rect


def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Warp document to a top-down rectangle."""
    # Sau khi có 4 góc trang, ta ép ảnh về dạng nhìn thẳng từ trên xuống.
    rect = order_points(pts)
    tl, tr, br, bl = rect

    # Tính kích thước ảnh đích dựa trên độ dài các cạnh đối diện.
    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = max(int(round(width_a)), int(round(width_b)), 1)

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = max(int(round(height_a)), int(round(height_b)), 1)

    dst = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype=np.float32,
    )

    # Ma trận biến đổi phối cảnh từ 4 điểm gốc sang 4 điểm đích.
    matrix = cv2.getPerspectiveTransform(rect, dst)
    # Nội suy để tạo ảnh đã được “trải phẳng”.
    warped = cv2.warpPerspective(image, matrix, (max_width, max_height))
    return warped
