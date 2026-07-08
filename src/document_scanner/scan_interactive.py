from __future__ import annotations

from pathlib import Path
import sys
from tkinter import Tk, filedialog, messagebox

# Khi bam Run truc tiep file nay trong VS Code, Python chi thay thu muc
# src/document_scanner. Them project root vao sys.path de import scan_folder.py.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scan_folder import DEFAULT_GT_DIR, IMAGE_EXTS, find_images, scan_images

try:
    from .pipeline import ScannerParams
except ImportError:
    from src.document_scanner.pipeline import ScannerParams


def make_root() -> Tk:
    # Tao cua so an de dung hop thoai chon file/thu muc cua Windows.
    root = Tk()
    root.withdraw()
    root.update()
    return root


def choose_images(root: Tk) -> tuple[list[Path], str] | None:
    # Hoi nguoi dung muon chon tung anh hay quet ca mot thu muc.
    choose_files = messagebox.askyesno(
        "Chon dau vao",
        "Ban muon chon tung anh de scan khong?\n\n"
        "Yes: chon mot hoac nhieu anh.\n"
        "No: chon ca mot thu muc anh.",
        parent=root,
    )

    if choose_files:
        filetypes = [
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff"),
            ("All files", "*.*"),
        ]
        selected = filedialog.askopenfilenames(
            title="Chon anh can scan",
            filetypes=filetypes,
            parent=root,
        )
        images = [Path(p) for p in selected if Path(p).suffix.lower() in IMAGE_EXTS]
        if not images:
            return None
        return images, f"{len(images)} anh duoc chon"

    folder = filedialog.askdirectory(title="Chon thu muc chua anh can scan", parent=root)
    if not folder:
        return None
    input_dir = Path(folder)
    images = find_images(input_dir)
    if not images:
        messagebox.showwarning(
            "Khong co anh",
            "Thu muc vua chon khong co file anh hop le (.jpg, .png, .bmp, .webp, .tif).",
            parent=root,
        )
        return None
    return images, str(input_dir)


def choose_output_dir(root: Tk) -> Path | None:
    # Chon noi luu ket qua. Chuong trinh se tao them runs/<thoi_gian> ben trong.
    folder = filedialog.askdirectory(title="Chon noi luu ket qua scan", parent=root)
    if not folder:
        return None
    return Path(folder)


def main() -> None:
    root = make_root()
    try:
        selected = choose_images(root)
        if selected is None:
            messagebox.showinfo("Da huy", "Ban chua chon anh hoac thu muc dau vao.", parent=root)
            return
        images, input_label = selected

        output_dir = choose_output_dir(root)
        if output_dir is None:
            messagebox.showinfo("Da huy", "Ban chua chon noi luu ket qua.", parent=root)
            return

        confirm = messagebox.askyesno(
            "Bat dau scan",
            f"Se scan {len(images)} anh.\n\n"
            f"Noi luu ket qua:\n{output_dir}\\runs\\<thoi_gian>\n\n"
            "Bat dau scan bay gio?",
            parent=root,
        )
        if not confirm:
            return

        params = ScannerParams(tesseract_lang="vie", tesseract_psm=6)
        run_dir = scan_images(
            images,
            output_dir,
            DEFAULT_GT_DIR,
            params,
            input_label=input_label,
            sweep_first=False,
        )

        messagebox.showinfo(
            "Scan xong",
            "Da scan xong.\n\n"
            f"Ket qua nam o:\n{run_dir}\n\n"
            "Trong moi thu muc anh co FINAL_SCAN.png va OCR_TEXT.txt.",
            parent=root,
        )
    finally:
        root.destroy()


if __name__ == "__main__":
    main()
