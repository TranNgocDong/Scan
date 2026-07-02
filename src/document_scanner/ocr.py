from __future__ import annotations

import os
import shutil
from pathlib import Path

import cv2


def is_tesseract_available() -> bool:
    """Check whether the external Tesseract executable is available."""
    # Neu tim duoc executable thi moi chay OCR that.
    return _resolve_tesseract_cmd() is not None


def _resolve_tesseract_cmd() -> str | None:
    """Resolve Tesseract path from env, PATH, or the known local install folder."""
    # Uu tien bien moi truong do nguoi dung cau hinh.
    env_cmd = os.environ.get("TESSERACT_CMD")
    if env_cmd and Path(env_cmd).exists():
        return env_cmd

    # Neu da add vao PATH thi lay truc tiep tu PATH.
    path_cmd = shutil.which("tesseract")
    if path_cmd:
        return path_cmd

    # Cuoi cung moi thu duong dan cai dat cuc bo ma project dang dung.
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
    # Gan dung duong dan executable de pytesseract biet goi engine nao.
    pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    # Neu chua co TESSDATA_PREFIX thi tu set sang thu muc tessdata da cai san.
    tessdata = os.environ.get("TESSDATA_PREFIX")
    if not tessdata and Path("E:/Visual_download/tessdata").exists():
        tessdata = "E:/Visual_download/tessdata"
        os.environ["TESSDATA_PREFIX"] = tessdata

    # Tesseract thich anh xam hoac RGB, nen anh mau BGR phai doi sang RGB truoc.
    if image.ndim == 2:
        ocr_image = image
    else:
        ocr_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # oem=3: dung che do OCR mac dinh manh nhat cua Tesseract.
    # psm do minh chon theo bo cuc tai lieu, o day mac dinh la 6.
    config = f"--oem 3 --psm {int(psm)} --dpi 300 -c preserve_interword_spaces=1"
    text = pytesseract.image_to_string(ocr_image, lang=lang, config=config)
    return text, None
