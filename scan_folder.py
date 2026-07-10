from __future__ import annotations

import argparse
import csv
import shutil
from datetime import datetime
from pathlib import Path

from src.document_scanner.pipeline import ScannerParams, parameter_sweep, process_document_image


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = ROOT / "data_Scan"
DEFAULT_WORKSPACE_DIR = ROOT / "scan_workspace"
DEFAULT_GT_DIR = ROOT / "data_Scan_ground_truth"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


TECHNIQUE_NOTES = """\
DOCUMENT SCANNER OCR - CAC KY THUAT DA DUNG

Muc tieu:
- Bien anh sach/tai lieu chup bang dien thoai thanh anh scan de nhin hon.
- Sau khi anh da duoc xu ly, moi dua vao Tesseract OCR de lay van ban.
- OCR chi la buoc cuoi; phan chinh cua project la pipeline Computer Vision truoc OCR.

Pipeline ky thuat:
1. Resize: giam kich thuoc anh lon de xu ly nhanh va on dinh hon.
2. Grayscale: dua anh mau ve mot kenh sang de de xu ly bien, threshold va contour.
3. CLAHE: tang tuong phan cuc bo, giup chu va nen tach nhau ro hon.
4. Gaussian blur: lam muot nhe de giam nhieu truoc khi do bien.
5. Canny edge detection: tim cac canh manh cua trang giay/vung tai lieu.
6. Contour detection: tim vung lon co kha nang la trang giay.
7. Bright-page contour: fallback thong minh dua vao vung giay sang neu Canny bat sai.
8. Perspective transform: sua meo phoi canh, dua trang nghieng ve dang scan thang hon.
9. Remove border/crop page: cat vien den, nen gach, nen thua quanh trang.
10. Adaptive threshold/Otsu: tach chu khoi nen khi anh sang khong deu.
11. Morphology nhe: lam sach nhieu nho nhung giu kernel nho de khong lam mat dau tieng Viet.
12. Text crop: cat dung vung co chu de OCR khong doc vao phan trang trong.
13. Background normalization: lam phang nen giay, giam bong va chu mat sau bi lo.
14. Upscale + white margin: phong to chu nho va them le trang de Tesseract doc de hon.
15. OCR candidate selection: thu nhieu anh dau vao OCR va nhieu PSM, chon ket qua co diem tot nhat.
16. Tesseract OCR + postprocess: trich xuat van ban va sua mot so loi OCR tieng Viet lap lai.

Lien he yeu cau cua thay:
- Du lieu that: anh dat trong data_Scan.
- It nhat 3 ky thuat: project dung nhieu ky thuat xu ly anh nhu Canny, contour, perspective, threshold, morphology.
- It nhat 2 ky thuat thuoc chuong 3/4/5: Canny/contour, threshold/morphology, OCR.
- Co anh trung gian: moi anh co thu muc rieng chua 01_resized den 13_best_ocr_input.
- Co danh gia dinh luong neu co ground truth .txt cung ten trong data_Scan_ground_truth.
- Co khao sat tham so neu chay them tuy chon --sweep-first.
"""


def find_images(image_dir: Path) -> list[Path]:
    # Duyet de quy de lay tat ca anh trong thu muc nguoi dung dua vao.
    return sorted(p for p in image_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def make_run_dir(workspace_dir: Path) -> Path:
    # Moi lan chay tao mot thu muc rieng theo thoi gian de khong ghi de ket qua cu.
    runs_dir = workspace_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = runs_dir / stamp
    counter = 1
    while run_dir.exists():
        run_dir = runs_dir / f"{stamp}_{counter:02d}"
        counter += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def ground_truth_for(image_path: Path, gt_dir: Path) -> Path | None:
    # Neu co file txt trung ten anh thi dung de tinh CER/WER.
    gt_path = gt_dir / f"{image_path.stem}.txt"
    return gt_path if gt_path.exists() else None


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def copy_final_files_to_image_folder(result: dict) -> dict:
    # Ngoai thu muc final chung, copy ket qua cuoi ve ngay thu muc cua tung anh cho de xem.
    image_dir = Path(result["output_dir"])
    final_scan = Path(result["final_scan"])
    ocr_text = Path(result["ocr_text_path"])
    image_scan = image_dir / "FINAL_SCAN.png"
    image_text = image_dir / "OCR_TEXT.txt"
    if final_scan.exists():
        shutil.copy2(final_scan, image_scan)
    if ocr_text.exists():
        shutil.copy2(ocr_text, image_text)
    result["image_final_scan"] = str(image_scan)
    result["image_ocr_text_path"] = str(image_text)
    return result


def write_image_report(result: dict, image_path: Path, params: ScannerParams) -> dict:
    # Ghi mot file tom tat ngan de khi van dap biet anh nay da di qua buoc nao.
    lines = [
        "BAO CAO XU LY CHO MOT ANH",
        "",
        f"Anh goc: {image_path}",
        f"Thu muc anh nay: {result.get('output_dir', '')}",
        f"Anh scan cuoi: {result.get('image_final_scan', result.get('final_scan', ''))}",
        f"Text OCR: {result.get('image_ocr_text_path', result.get('ocr_text_path', ''))}",
        "",
        "Ket qua ky thuat:",
        f"- contour_method: {result.get('contour_method', '')}",
        f"- used_fallback_page: {result.get('used_fallback_page', '')}",
        f"- ocr_variant: {result.get('ocr_variant', '')}",
        f"- ocr_warning: {result.get('ocr_warning', '')}",
        "",
        "Tham so chinh:",
        f"- max_side: {params.max_side}",
        f"- blur_ksize: {params.blur_ksize}",
        f"- canny_low/canny_high: {params.canny_low}/{params.canny_high}",
        f"- adaptive_block_size/adaptive_c: {params.adaptive_block_size}/{params.adaptive_c}",
        f"- morph_kernel: {params.morph_kernel}",
        f"- text_crop_padding: {params.text_crop_padding}",
        f"- ocr_scale: {params.ocr_scale}",
        f"- tesseract_lang: {params.tesseract_lang}",
        f"- tesseract_psm: {params.tesseract_psm}",
        "",
        "Cac anh trung gian can xem khi bao cao:",
        "01_resized.jpg, 02_gray.jpg, 03_enhanced.jpg, 04_edges.jpg,",
        "05_document_contour.jpg, 06_warped.jpg, 07_borderless.jpg,",
        "07b_page_region.jpg, 08_threshold.jpg, 09_cleaned.jpg,",
        "10_text_crop.jpg, 11_ocr_ready.jpg, 12_readable_scan.jpg,",
        "12a_ocr_page_gray.jpg, 12b_ocr_background_norm.jpg,",
        "12c_ocr_dominant_background.jpg, 13_best_ocr_input.jpg,",
        "FINAL_SCAN.png, OCR_TEXT.txt",
    ]
    if "cer" in result:
        lines.extend(["", "Danh gia voi ground truth:", f"- CER: {result['cer']}", f"- WER: {result['wer']}"])
    report_path = Path(result["output_dir"]) / "SCAN_REPORT.txt"
    write_text(report_path, "\n".join(lines))
    result["scan_report"] = str(report_path)
    return result


def write_summary(run_dir: Path, rows: list[dict]) -> Path:
    summary_path = run_dir / "summary.csv"
    fieldnames = [
        "image",
        "output_dir",
        "used_fallback_page",
        "contour_method",
        "ocr_variant",
        "final_scan",
        "image_final_scan",
        "ocr_text_path",
        "image_ocr_text_path",
        "scan_report",
        "ocr_warning",
        "cer",
        "wer",
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return summary_path


def write_run_report(run_dir: Path, input_dir: Path | str, rows: list[dict], params: ScannerParams) -> Path:
    ok_count = sum(1 for row in rows if not row.get("ocr_warning"))
    fallback_count = sum(1 for row in rows if row.get("used_fallback_page"))
    report = [
        "BAO CAO LAN SCAN",
        "",
        f"Thu muc anh dau vao: {input_dir}",
        f"Thu muc ket qua: {run_dir}",
        f"So anh da xu ly: {len(rows)}",
        f"So anh OCR khong bao loi: {ok_count}",
        f"So anh phai fallback full image: {fallback_count}",
        "",
        "Lenh chay lai:",
        "python scan_folder.py --input data_Scan --lang vie",
        "",
        "Tham so mac dinh:",
        f"- max_side={params.max_side}",
        f"- canny=({params.canny_low}, {params.canny_high})",
        f"- adaptive_block_size={params.adaptive_block_size}",
        f"- morph_kernel={params.morph_kernel}",
        f"- text_crop_padding={params.text_crop_padding}",
        f"- ocr_scale={params.ocr_scale}",
        f"- lang={params.tesseract_lang}",
        f"- psm={params.tesseract_psm}",
        "",
        "Cach doc ket qua:",
        "- Moi anh co mot thu muc rieng nam truc tiep trong run nay.",
        "- Mo FINAL_SCAN.png de xem anh scan cuoi.",
        "- Mo OCR_TEXT.txt de xem noi dung OCR.",
        "- Mo SCAN_REPORT.txt de xem anh do da dung contour/fallback/ocr_variant nao.",
        "- Mo summary.csv de xem tong hop tat ca anh.",
    ]
    report_path = run_dir / "RUN_REPORT.txt"
    write_text(report_path, "\n".join(report))
    return report_path


def scan_images(
    images: list[Path],
    workspace_dir: Path,
    gt_dir: Path,
    params: ScannerParams,
    *,
    input_label: Path | str,
    sweep_first: bool = False,
) -> Path:
    # Ham dung chung cho ca che do dong lenh va che do chon anh bang cua so.
    run_dir = make_run_dir(workspace_dir)
    write_text(run_dir / "TECHNIQUES_USED.txt", TECHNIQUE_NOTES)

    rows = []
    for index, image_path in enumerate(images, start=1):
        print(f"[{index}/{len(images)}] Dang scan: {image_path.name}")
        gt_path = ground_truth_for(image_path, gt_dir)
        result = process_document_image(image_path, run_dir, params=params, ground_truth_path=gt_path)
        result = copy_final_files_to_image_folder(result)
        result = write_image_report(result, image_path, params)
        rows.append(result)
        print(f"  -> Thu muc anh: {result['output_dir']}")
        print(f"  -> Scan cuoi: {result['image_final_scan']}")
        print(f"  -> OCR text: {result['image_ocr_text_path']}")
        if result.get("ocr_warning"):
            print(f"  -> OCR warning: {result['ocr_warning']}")

    summary_path = write_summary(run_dir, rows)
    run_report = write_run_report(run_dir, input_label, rows, params)
    write_text(workspace_dir / "LATEST_RUN.txt", str(run_dir))

    if sweep_first:
        print("[sweep] Dang chay khao sat tham so tren anh dau tien...")
        first = images[0]
        sweep_rows = parameter_sweep(
            first,
            run_dir / "parameter_sweep",
            ground_truth_path=ground_truth_for(first, gt_dir),
            base_params=params,
        )
        sweep_path = run_dir / "parameter_sweep.csv"
        with sweep_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=sorted({k for row in sweep_rows for k in row}), extrasaction="ignore")
            writer.writeheader()
            writer.writerows(sweep_rows)
        print(f"[sweep] Da luu: {sweep_path}")

    print("")
    print("HOAN TAT")
    print(f"Thu muc ket qua: {run_dir}")
    print(f"Summary: {summary_path}")
    print(f"Bao cao lan scan: {run_report}")
    print(f"File ghi nho lan moi nhat: {workspace_dir / 'LATEST_RUN.txt'}")
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan all document photos from a folder into timestamped results.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT_DIR), help="Folder chua anh can scan")
    parser.add_argument("--workspace", default=str(DEFAULT_WORKSPACE_DIR), help="Thu muc luu cac lan scan")
    parser.add_argument("--ground-truth", default=str(DEFAULT_GT_DIR), help="Thu muc .txt cung ten anh de tinh CER/WER neu co")
    parser.add_argument("--lang", default="vie", help="Ngon ngu Tesseract, vi du vie hoac eng")
    parser.add_argument("--psm", type=int, default=6, help="Page segmentation mode mac dinh cua Tesseract")
    parser.add_argument("--max-side", type=int, default=1200, help="Canh dai nhat sau resize")
    parser.add_argument("--sweep-first", action="store_true", help="Chay khao sat tham so tren anh dau tien")
    parser.add_argument("--limit", type=int, default=0, help="Chi xu ly N anh dau tien de test nhanh, 0 la xu ly tat ca")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input)
    workspace_dir = Path(args.workspace)
    gt_dir = Path(args.ground_truth)
    images = find_images(input_dir)
    if args.limit and args.limit > 0:
        images = images[: args.limit]

    if not images:
        print(f"Khong tim thay anh trong: {input_dir}")
        print("Hay dat anh .jpg/.png vao thu muc data_Scan roi chay lai.")
        return

    params = ScannerParams(max_side=args.max_side, tesseract_lang=args.lang, tesseract_psm=args.psm)
    scan_images(
        images,
        workspace_dir,
        gt_dir,
        params,
        input_label=input_dir,
        sweep_first=args.sweep_first,
    )


if __name__ == "__main__":
    main()
