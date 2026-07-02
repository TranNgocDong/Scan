from __future__ import annotations

import argparse
import csv
from pathlib import Path

from src.document_scanner.pipeline import ScannerParams, parameter_sweep, process_document_image


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def find_images(image_dir: Path) -> list[Path]:
    # Duyệt đệ quy để nhặt mọi ảnh thật trong thư mục đầu vào.
    return sorted(p for p in image_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def ground_truth_for(image_path: Path, gt_dir: Path) -> Path:
    # Mỗi ảnh được ghép với một file .txt cùng stem tên.
    return gt_dir / f"{image_path.stem}.txt"


def main() -> None:
    parser = argparse.ArgumentParser(description="Document scanner OCR pipeline")
    parser.add_argument("--images", default="data/images", help="Folder containing real document photos")
    parser.add_argument("--ground-truth", default="data/ground_truth", help="Folder containing .txt ground truth")
    parser.add_argument("--output", default="results", help="Output folder")
    parser.add_argument("--lang", default="eng", help="Tesseract language, e.g. eng or vie")
    parser.add_argument("--psm", type=int, default=6, help="Tesseract page segmentation mode")
    parser.add_argument("--sweep", action="store_true", help="Run parameter sweep on first image")
    args = parser.parse_args()

    image_dir = Path(args.images)
    gt_dir = Path(args.ground_truth)
    output_dir = Path(args.output)
    images = find_images(image_dir)

    if not images:
        print(f"No images found in {image_dir}. Put real document photos there first.")
        return

    # Chạy pipeline cho từng ảnh và thu kết quả để ghi summary.csv.
    rows = []
    params = ScannerParams(tesseract_lang=args.lang, tesseract_psm=args.psm)
    for image_path in images:
        gt_path = ground_truth_for(image_path, gt_dir)
        gt_arg = gt_path if gt_path.exists() else None
        result = process_document_image(image_path, output_dir, params=params, ground_truth_path=gt_arg)
        rows.append(result)
        print(f"Processed {image_path.name}")
        if result.get("ocr_warning"):
            print("  OCR warning:", result["ocr_warning"])
        if "cer" in result:
            print(f"  CER={result['cer']:.3f}, WER={result['wer']:.3f}")

    summary_path = output_dir / "summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image",
        "used_fallback_page",
        "contour_method",
        "ocr_variant",
        "final_scan",
        "ocr_text_path",
        "ocr_warning",
        "cer",
        "wer",
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Summary saved to {summary_path}")

    if args.sweep:
        # Sweep chỉ chạy trên ảnh đầu tiên để tiết kiệm thời gian.
        first = images[0]
        gt_path = ground_truth_for(first, gt_dir)
        gt_arg = gt_path if gt_path.exists() else None
        sweep_rows = parameter_sweep(first, output_dir, ground_truth_path=gt_arg, base_params=params)
        sweep_path = output_dir / "parameter_sweep.csv"
        with sweep_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=sorted({k for row in sweep_rows for k in row}), extrasaction="ignore")
            writer.writeheader()
            writer.writerows(sweep_rows)
        print(f"Parameter sweep saved to {sweep_path}")


if __name__ == "__main__":
    main()
