from __future__ import annotations

import os
import re
import shutil
import unicodedata
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


def postprocess_ocr_text(text: str) -> str:
    """Clean common OCR mistakes in Vietnamese printed documents."""
    # Hau xu ly nhe: sua khoang trang va mot so loi OCR tieng Viet hay gap.
    # Day khong thay the OCR, chi lam sach nhung loi lap lai do dau/ky tu bi nham.
    text = unicodedata.normalize("NFC", text or "")
    text = text.replace("\x0c", "")
    text = text.replace("“", '"').replace("”", '"').replace("''", '"')
    text = text.replace("’", "'").replace("‘", "'")

    replacements = [
        ("kình doanh", "kinh doanh"),
        ("kình đoanh", "kinh doanh"),
        ("cẩu lợi", "cầu lợi"),
        ("câu lợi", "cầu lợi"),
        ("mâu thuần", "mâu thuẫn"),
        ("mâu thuả n", "mâu thuẫn"),
        ("mâu th uẫn", "mâu thuẫn"),
        ("mâu thị uẫn", "mâu thuẫn"),
        ("glai cấp", "giai cấp"),
        ("gỉai cấp", "giai cấp"),
        ("đục vọng", "dục vọng"),
        ("duc vọng", "dục vọng"),
        ("duc vong", "dục vọng"),
        ("quyển lực", "quyền lực"),
        ("quyển lự", "quyền lực"),
        ("đòn bảy", "đòn bẩy"),
        ("đòn bẫy", "đòn bẩy"),
        ("cập phạm trù", "cặp phạm trù"),
        ("điểu lợi", "điều lợi"),
        ("điểu lợ", "điều lợi"),
        ("đẻ xuất", "đề xuất"),
        ("chỉ ra răng", "chỉ ra rằng"),
        ("từng chỉ ra răng", "từng chỉ ra rằng"),
        ("xâu xa", "xấu xa"),
        ("vẻ lợi", "về lợi"),
        ("tơ bản", "cơ bản"),
        ("lo đanh", "lo danh"),
        ("Nhân đân", "Nhân dân"),
        ("lịch sứ", "lịch sử"),
        ("cọn người", "con người"),
    ]

    for wrong, right in replacements:
        pattern = re.compile(re.escape(wrong), flags=re.IGNORECASE)

        def repl(match: re.Match) -> str:
            found = match.group(0)
            if found[:1].isupper():
                return right[:1].upper() + right[1:]
            return right

        text = pattern.sub(repl, text)

    cleaned_lines = []
    for line in text.splitlines():
        line = re.sub(r"[ \t]+", " ", line).strip()
        # Bo mot so dong gan nhu chi la ky tu rac.
        letters = sum(ch.isalpha() for ch in line)
        if line and letters == 0 and len(line) <= 4:
            continue
        cleaned_lines.append(line)
    text = "\n".join(cleaned_lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + ("\n" if text.strip() else "")


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
    text = postprocess_ocr_text(pytesseract.image_to_string(ocr_image, lang=lang, config=config))
    return text, None


def run_tesseract_ocr_with_confidence(image, lang: str = "eng", psm: int = 6) -> tuple[str, float, str | None]:
    """Run OCR and return mean word confidence from Tesseract.

    Returns (text, mean_confidence, warning). Confidence is -1 when unavailable.
    """
    try:
        import pytesseract
    except ImportError:
        return "", -1.0, "Python package pytesseract is not installed. Run: python -m pip install pytesseract"

    tesseract_cmd = _resolve_tesseract_cmd()
    if tesseract_cmd is None:
        return "", -1.0, (
            "Tesseract executable was not found in PATH. Install Tesseract OCR "
            "and add it to PATH before running OCR."
        )
    pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    tessdata = os.environ.get("TESSDATA_PREFIX")
    if not tessdata and Path("E:/Visual_download/tessdata").exists():
        tessdata = "E:/Visual_download/tessdata"
        os.environ["TESSDATA_PREFIX"] = tessdata

    if image.ndim == 2:
        ocr_image = image
    else:
        ocr_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    config = f"--oem 3 --psm {int(psm)} --dpi 300 -c preserve_interword_spaces=1"
    text = postprocess_ocr_text(pytesseract.image_to_string(ocr_image, lang=lang, config=config))

    confidences = []
    data = pytesseract.image_to_data(
        ocr_image,
        lang=lang,
        config=config,
        output_type=pytesseract.Output.DICT,
    )
    for raw_conf in data.get("conf", []):
        try:
            conf = float(raw_conf)
        except (TypeError, ValueError):
            continue
        if conf >= 0:
            confidences.append(conf)

    mean_conf = sum(confidences) / len(confidences) if confidences else -1.0
    return text, mean_conf, None
