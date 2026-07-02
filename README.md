# Document Scanner OCR

Project nay xay dung pipeline xu ly anh tai lieu chup bang dien thoai de tao anh scan de nhin hon va trich xuat van ban bang Tesseract OCR. Muc tieu chinh khong phai chi goi OCR, ma la ap dung cac ky thuat Computer Vision de lam sach anh truoc khi dua vao OCR.

De tai phu hop voi bai tap lon mon Xu ly anh va Thi giac may tinh vi giai quyet mot bai toan thuc te: bien anh tai lieu chu in bi nghieng, sang khong deu, co nen xau hoac vien thua thanh anh scan va text.

## 1. Project Lam Gi?

Input la anh tai lieu chu in chup bang dien thoai.

Output gom:

- Anh scan cuoi, de doc hon anh goc.
- File text OCR.
- Anh trung gian sau tung buoc xu ly.
- Bang tong hop ket qua va tham so.

Pipeline xu ly:

```text
Anh goc
-> resize
-> grayscale
-> tang tuong phan CLAHE
-> Gaussian blur
-> Canny edge
-> tim contour trang giay
-> bright-page contour fallback
-> perspective transform
-> cat vien va vung nen thua
-> adaptive threshold / Otsu threshold
-> morphology nhe
-> crop vung chu
-> upscale anh OCR
-> thu nhieu bien the OCR
-> Tesseract OCR
-> danh gia CER/WER neu co ground truth
```

## 2. Cau Truc Code

```text
.
|-- run_pipeline.py
`-- src/
    `-- document_scanner/
        |-- __init__.py
        |-- preprocess.py
        |-- document_detect.py
        |-- perspective.py
        |-- ocr.py
        |-- evaluation.py
        `-- pipeline.py
```

Y nghia tung file:

| File | Vai tro |
|---|---|
| `run_pipeline.py` | File chay chinh tu command line, doc anh dau vao, goi pipeline, ghi `summary.csv`. |
| `preprocess.py` | Resize, grayscale, CLAHE, blur, threshold, morphology, crop vung chu. |
| `document_detect.py` | Canny edge, tim contour trang giay, bright-page contour, ve contour trung gian. |
| `perspective.py` | Sap xep 4 goc va thuc hien perspective transform de sua anh nghieng. |
| `ocr.py` | Goi Tesseract OCR, cau hinh duong dan Tesseract, lay confidence OCR. |
| `evaluation.py` | Tinh CER va WER bang edit distance. |
| `pipeline.py` | Noi tat ca buoc xu ly anh thanh mot pipeline hoan chinh. |

## 3. Ky Thuat Computer Vision Su Dung

| Ky thuat | Dung de lam gi |
|---|---|
| Grayscale | Dua anh ve mot kenh sang de xu ly don gian hon. |
| CLAHE | Tang tuong phan cuc bo, giup chu noi ro hon tren nen giay. |
| Gaussian blur | Giam nhieu truoc khi do bien. |
| Canny edge detection | Tim canh manh cua trang giay hoac vung tai lieu. |
| Contour detection | Tim bien/vung trang giay de cat va warp. |
| Bright-page contour | Fallback khi Canny bat nham nen gach, bong tay hoac duong thua. |
| Perspective transform | Sua meo phoi canh, dua trang nghieng ve gan dang scan thang. |
| Adaptive threshold | Tach chu khoi nen khi do sang khong deu. |
| Otsu threshold | Thu them cach tach nguong tu dong dua tren histogram. |
| Morphology | Lam sach nhieu nho, nhung kernel phai nho de khong lam mat dau tieng Viet. |
| Text crop + upscale | Cat vung co chu va phong to truoc OCR. |
| OCR confidence | Chon bien the OCR tot hon dua tren do tin cay cua Tesseract. |
| CER/WER | Danh gia dinh luong ket qua OCR neu co file ground truth. |

### Ky Thuat Duoc Cai Dat O File Nao?

| Ky thuat | File code | Ham chinh |
|---|---|---|
| Resize anh | `src/document_scanner/preprocess.py` | `resize_max_side()` |
| Chuyen grayscale | `src/document_scanner/preprocess.py` | `to_gray()` |
| Tang tuong phan CLAHE | `src/document_scanner/preprocess.py` | `clahe_contrast()` |
| Gaussian blur | `src/document_scanner/preprocess.py` | `denoise()` |
| Adaptive threshold | `src/document_scanner/preprocess.py` | `adaptive_binarize()` |
| Otsu threshold | `src/document_scanner/preprocess.py` | `otsu_binarize()` |
| Morphology | `src/document_scanner/preprocess.py` | `clean_binary()` |
| Cat vien den | `src/document_scanner/preprocess.py` | `remove_dark_borders()` |
| Cat vung giay sang | `src/document_scanner/preprocess.py` | `crop_light_page_region()` |
| Cat vung van ban | `src/document_scanner/preprocess.py` | `crop_text_region()` |
| Them le trang cho OCR | `src/document_scanner/preprocess.py` | `add_white_margin()` |
| Canny edge detection | `src/document_scanner/document_detect.py` | `canny_edges()` |
| Tim contour trang giay | `src/document_scanner/document_detect.py` | `find_document_contour()` |
| Bright-page contour | `src/document_scanner/document_detect.py` | `find_bright_document_contour()` |
| Fallback khi khong tim duoc trang | `src/document_scanner/document_detect.py` | `fallback_page_corners()` |
| Ve contour trung gian | `src/document_scanner/document_detect.py` | `draw_contour_overlay()` |
| Sap xep 4 goc trang | `src/document_scanner/perspective.py` | `order_points()` |
| Perspective transform | `src/document_scanner/perspective.py` | `four_point_transform()` |
| Goi Tesseract OCR | `src/document_scanner/ocr.py` | `run_tesseract_ocr()` |
| Lay confidence OCR | `src/document_scanner/ocr.py` | `run_tesseract_ocr_with_confidence()` |
| CER | `src/document_scanner/evaluation.py` | `character_error_rate()` |
| WER | `src/document_scanner/evaluation.py` | `word_error_rate()` |
| Pipeline tong | `src/document_scanner/pipeline.py` | `process_document_image()` |
| Khao sat tham so | `src/document_scanner/pipeline.py` | `parameter_sweep()` |
| Chay bang command line | `run_pipeline.py` | `main()` |

Neu thay hoi "ky thuat nay nam o dau", co the tra loi theo bang tren. Vi du: Canny nam trong `document_detect.py`, perspective transform nam trong `perspective.py`, OCR va confidence nam trong `ocr.py`, con CER/WER nam trong `evaluation.py`.

### Ky Thuat Dua Theo File PDF Nao?

| Ky thuat trong project | File PDF lien quan | Ly do lien quan |
|---|---|---|
| Muc tieu project, pipeline, parameter sweep, bao cao, metric | `Computer_Vision 25.pdf` | Day la file huong dan bai tap lon, yeu cau bai toan thuc te, it nhat 3 ky thuat, co anh trung gian, khao sat tham so va danh gia dinh luong. |
| Grayscale, histogram/contrast, thresholding co ban | `Computer_Vision 18 (1).pdf` | Bai 1 gioi thieu xu ly anh co ban, point operators, histogram equalization va thresholding. |
| Gaussian blur, convolution, Sobel co ban | `Computer_Vision 18 (1).pdf` | Slide co phan Gaussian, convolution va Sobel/gradient de lam nen cho xu ly anh va do bien. |
| Geometric transform, affine/warp, inverse warping, bilinear interpolation | `Computer_Vision 18 (1).pdf` | Slide co phan affine, warping nguoc va noi suy song tuyen. Project dung y tuong nay cho perspective/warp trang tai lieu. |
| Morphology, erosion/dilation/opening/closing | `Computer_Vision 19 (2).pdf` | File ghi chu bo sung co phan Phep bien doi hinh thai, phan tu cau truc, erosion va dilation. Project dung morphology de lam sach nhiu nho sau threshold. |
| Canny edge detection, Gaussian, Sobel, NMS, hysteresis | `Computer_Vision 23.pdf` va `Computer_Vision 24 (2).pdf` | Hai file nay trinh bay phat hien canh, gradient, Canny va cac buoc lien quan. Project dung Canny de tim bien trang giay. |
| Contour va document scanning | `Computer_Vision 23.pdf` | Slide co phan edge/contour va ung dung document scanning: tim bien trang giay de crop va perspective-correct. |
| Otsu threshold / nguong tu dong | `Computer_Vision 23.pdf` va `Computer_Vision 24 (2).pdf` | Slide/tai lieu bo sung co nhac Otsu auto-threshold trong phan Canny/threshold baseline. Project thu Otsu nhu mot bien the dau vao OCR. |
| Segmentation/fallback theo vung sang cua trang giay | `Computer_Vision 27.pdf` hoac `Computer_Vision 27 (1).pdf` | File nay ve phan doan co ban, region/contour. Project van dung y tuong phan vung de tach vung giay sang khoi nen khi Canny bat sai. |
| OCR / recognition | Lien he Chuong 5 va project requirement trong `Computer_Vision 25.pdf` | Tesseract OCR la buoc nhan dang cuoi cua pipeline. Phan nay la ung dung thuc chien, khong phai thuat toan tu cai dat tu slide. |
| CER/WER | `Computer_Vision 25.pdf` | File bai tap lon yeu cau danh gia dinh luong bang metric. Project chon CER/WER vi phu hop bai toan OCR. |

Tom lai, project lay yeu cau tong tu `Computer_Vision 25.pdf`, lay nen tang xu ly anh/warp tu `Computer_Vision 18 (1).pdf`, lay morphology tu `Computer_Vision 19 (2).pdf`, lay Canny/edge/contour/Otsu tu `Computer_Vision 23.pdf` va `Computer_Vision 24 (2).pdf`, va lien he them phan segmentation tu `Computer_Vision 27.pdf`.

## 4. Cai Dat Moi Truong

Can Python va cac thu vien:

```bash
pip install opencv-python numpy matplotlib pillow pytesseract pandas
```

Neu co file `requirements.txt`, co the chay:

```bash
pip install -r requirements.txt
```

Project dung Tesseract OCR. Neu may chua cai Tesseract, tai ban Windows tai link:

```text
https://github.com/UB-Mannheim/tesseract/wiki
```

Sau khi cai, kiem tra:

```powershell
tesseract --version
```

Tren may dang lam project nay, Tesseract dat tai:

```text
E:\Visual_download\tesseract.exe
E:\Visual_download\tessdata
```

Can co language pack tieng Viet. File can co la:

```text
E:\Visual_download\tessdata\vie.traineddata
```

Neu luc cai Tesseract chua co tieng Viet, co the tai file `vie.traineddata` tu tessdata cua Tesseract:

```text
https://github.com/tesseract-ocr/tessdata
```

Sau do copy `vie.traineddata` vao thu muc `tessdata` cua Tesseract.

Neu Windows chua nhan Tesseract trong PATH, set bien moi truong truoc khi chay:

```powershell
$env:PATH = 'E:\Visual_download;' + $env:PATH
$env:TESSDATA_PREFIX = 'E:\Visual_download\tessdata'
```

## 5. Cach Chuan Bi Du Lieu

Trong thu muc project, tao cau truc:

```text
data/
|-- images/
|   `-- doc_001.png
`-- ground_truth/
    `-- doc_001.txt
```

Luu y:

- Anh trong `data/images` la anh tai lieu that, khong phai anh tong hop.
- File ground truth co cung ten voi anh, chi khac duoi `.txt`.
- Neu khong co ground truth, pipeline van chay OCR nhung se khong tinh CER/WER.
- Anh chup nen uu tien mot trang, chup gan, ro chu, anh sang deu, it cong trang.

## 6. Cach Chay

Chay pipeline co tinh CER/WER neu co ground truth:

```powershell
python run_pipeline.py --images data/images --ground-truth data/ground_truth --output results --lang vie
```

Chay kem parameter sweep:

```powershell
python run_pipeline.py --images data/images --ground-truth data/ground_truth --output results --lang vie --sweep
```

Neu OCR tieng Anh:

```powershell
python run_pipeline.py --images data/images --ground-truth data/ground_truth --output results --lang eng
```

## 7. Ket Qua Dau Ra

Sau khi chay, ket qua nam trong `results/`.

```text
results/
|-- summary.csv
|-- parameter_sweep.csv
|-- final/
|   |-- doc_001_scan.png
|   `-- doc_001_ocr.txt
`-- doc_001/
    |-- 01_resized.jpg
    |-- 02_gray.jpg
    |-- 03_enhanced.jpg
    |-- 04_edges.jpg
    |-- 05_document_contour.jpg
    |-- 06_warped.jpg
    |-- 07_borderless.jpg
    |-- 07b_page_region.jpg
    |-- 08_threshold.jpg
    |-- 09_cleaned.jpg
    |-- 10_text_crop.jpg
    |-- 11_ocr_ready.jpg
    |-- 12_readable_scan.jpg
    `-- 13_best_ocr_input.jpg
```

Y nghia:

- `final/*_scan.png`: anh scan cuoi de xem/dua vao bao cao.
- `final/*_ocr.txt`: noi dung text OCR.
- `summary.csv`: tong hop moi anh, cach detect trang, bien the OCR, CER/WER.
- `parameter_sweep.csv`: ket qua khao sat tham so neu chay `--sweep`.
- Anh `01` den `13`: anh trung gian de giai thich trong bao cao va van dap.

## 8. Parameter Sweep

Pipeline hien thu cac nhom tham so:

| Nhom | Gia tri thu |
|---|---|
| Canny threshold | `(40,120)`, `(60,160)`, `(100,220)` |
| Adaptive block size | `15`, `31`, `51` |
| Morphology kernel | `1`, `2`, `3` |
| Text crop padding | `12`, `28`, `44` |

Nhan xet quan trong:

- Canny threshold qua thap co the bat nhieu nen/thua.
- Canny threshold qua cao co the mat bien trang.
- Morphology kernel lon co the lam mat dau tieng Viet hoac lam dinh net chu.
- Anh chu nho thuong nen dung morphology nhe.

## 9. CER Va WER

CER la ty le loi theo ky tu:

```text
CER = edit_distance(text_ocr, text_dung) / so_ky_tu_text_dung
```

WER la ty le loi theo tu:

```text
WER = edit_distance(tu_ocr, tu_dung) / so_tu_text_dung
```

CER/WER cang thap thi OCR cang tot.

Vi du:

- CER thap: OCR it sai ky tu.
- WER cao: OCR co the sai dau, tach tu sai, mat tu hoac them tu.

## 10. Nhung Dieu Can Luu Y

OCR khong doc hoan hao moi anh. Anh nhin bang mat nguoi co the ro, nhung OCR van co the sai vi:

- Chu bi nghieng hoac font in nghieng.
- Dau tieng Viet nho, de bi nham.
- Trang sach cong, khong phang.
- Co bong giay, nen xam, chu mat sau hien len.
- Chu qua nho hoac anh hoi mo.
- Anh chup sat gay sach, dong chu bi cong.
- Mot anh co hai trang hoac nhieu cot van ban.

Project nay tap trung vao pipeline scan va tien xu ly truoc OCR. Neu muon OCR chinh xac hon nua, co the cai tien them:

- Dewarp trang cong.
- Tach tung dong chu truoc OCR.
- Tach tung cot/trang khi anh co nhieu vung van ban.
- Hau xu ly text bang tu dien tieng Viet.
- Dung model OCR manh hon Tesseract cho tieng Viet.
