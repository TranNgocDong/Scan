from __future__ import annotations

import os
import shutil
from pathlib import Path

import cv2


def is_tesseract_available() -> bool:
    """Check whether the external Tesseract executable is available."""
    # Nếu tìm được executable thì mới chạy OCR thật.
    return _resolve_tesseract_cmd() is not None


def _resolve_tesseract_cmd() -> str | None:
    """Resolve Tesseract path from env, PATH, or the known local install folder."""
    # Ưu tiên biến môi trường do người dùng cấu hình.
    env_cmd = os.environ.get("TESSERACT_CMD")
    if env_cmd and Path(env_cmd).exists():
        return env_cmd

    # Nếu đã add vào PATH thì lấy trực tiếp từ PATH.
    path_cmd = shutil.which("tesseract")
    if path_cmd:
        return path_cmd

    # Cuối cùng mới thử đường dẫn cài đặt cục bộ mà project đang dùng.
    known = Path("E:/Visual_download/tesseract.exe")
    if known.exists():
        return str(known)
    return None


def run_tesseract_ocr(image, lang: str = "eng", psm: int = 6) -> tuple[str, str | None]:
    """Run pytesseract OCR if both pytesseract and Tesseract engine are available.

    Returns (text, warning). warning is None when OCR succeeds.
    """
    try:
        import pytesseract
    except ImportError:
        return "", "Python package pytesseract is not installed. Run: python -m pip install pytesseract"

    tesseract_cmd = _resolve_tesseract_cmd()
    if tesseract_cmd is None:
        return "", (
            "Tesseract executable was not found in PATH. Install Tesseract OCR "
            "and add it to PATH before running OCR."
        )
    # Gắn đúng đường dẫn executable để pytesseract biết gọi engine nào.
    pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    # Nếu chưa có TESSDATA_PREFIX thì tự set sang thư mục tessdata đã cài sẵn.
    tessdata = os.environ.get("TESSDATA_PREFIX")
    if not tessdata and Path("E:/Visual_download/tessdata").exists():
        tessdata = "E:/Visual_download/tessdata"
        os.environ["TESSDATA_PREFIX"] = tessdata

    # Tesseract thích ảnh xám hoặc RGB, nên ảnh màu BGR phải đổi sang RGB trước.
    if image.ndim == 2:
        ocr_image = image
    else:
        ocr_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # oem=3: dùng chế độ OCR mặc định mạnh nhất của Tesseract.
    # psm do mình chọn theo bố cục tài liệu, ở đây mặc định là 6.
    config = f"--oem 3 --psm {int(psm)}"
    text = pytesseract.image_to_string(ocr_image, lang=lang, config=config)
    return text, None
