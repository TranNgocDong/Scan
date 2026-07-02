from __future__ import annotations

import re


def normalize_text(text: str) -> str:
    """Normalize OCR/ground-truth text for fairer CER/WER evaluation."""
    # Chuan hoa de tranh lech vi khac hoa/thuong hoac khoang trang thua.
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def levenshtein(a: list[str] | str, b: list[str] | str) -> int:
    """Compute Levenshtein edit distance."""
    # Day la so phep sua it nhat de doi chuoi a thanh chuoi b.
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n

    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(
                prev[j] + 1,
                cur[j - 1] + 1,
                prev[j - 1] + cost,
            )
        prev = cur
    return prev[m]


def character_error_rate(pred: str, truth: str) -> float:
    """CER = edit_distance(characters) / number_of_truth_characters."""
    # Do sai so theo tung ky tu, phu hop de xem OCR sai dau hay sai chu.
    pred_n = normalize_text(pred)
    truth_n = normalize_text(truth)
    if not truth_n:
        return 0.0 if not pred_n else 1.0
    return levenshtein(pred_n, truth_n) / len(truth_n)


def word_error_rate(pred: str, truth: str) -> float:
    """WER = edit_distance(words) / number_of_truth_words."""
    # Do sai so theo tung tu, nen nhay voi viec mat tu hoac tach tu sai.
    pred_words = normalize_text(pred).split()
    truth_words = normalize_text(truth).split()
    if not truth_words:
        return 0.0 if not pred_words else 1.0
    return levenshtein(pred_words, truth_words) / len(truth_words)
