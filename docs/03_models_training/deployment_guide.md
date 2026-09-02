# SỔ TAY TRIỂN KHAI ĐA NỀN TẢNG — TRAIN SONG SONG COLAB + KAGGLE

Tài liệu này hướng dẫn cách chia 3 mô hình ra chạy **đồng thời** trên hai nền tảng miễn phí (Google Colab và Kaggle) để rút ngắn tổng thời gian huấn luyện, thay vì xếp hàng chạy tuần tự từng mô hình một.

**Áp dụng cho:** 3 notebook trong `Traffic-Sign-Detection-ZaloAI/notebooks/v3/`.

---

## PHẦN 0: PHÂN CÔNG MÔ HÌNH CHO TỪNG NỀN TẢNG

### 0.1. Bảng phân công

| Mô hình | Notebook V3 | Nền tảng | GPU | Lý do phân công |
|---|---|---|---|---|
| **YOLOv8s-P2** | `train_yolov8_v3.ipynb` | **Kaggle** | P100 | Chạy `imgsz=1280` nên rất ngốn VRAM. Kaggle cấp 30 giờ GPU/tuần với thời hạn 12 giờ liên tục mỗi phiên — dư sức nuốt 100 epoch ảnh độ phân giải cao |
| **Faster R-CNN** | `train_faster_rcnn_v3.ipynb` | **Kaggle** | P100 hoặc T4 | Kiến trúc Two-stage nặng, lại cần đo mAP sau mỗi epoch. Cùng nằm trên Kaggle để dùng chung `/kaggle/input/` đã mount sẵn dataset, không tốn thời gian tải lại |
| **RT-DETR-L** | `train_rtdetr_v3.ipynb` | **Google Colab** | T4 | Chạy `imgsz=640` nên nhẹ nhất trong ba mô hình, hợp với hạn mức khắt khe của Colab. Đẩy sang Colab để giải phóng slot Kaggle cho hai mô hình nặng |

### 0.2. Vì sao chia như vậy?

Kaggle cho phép chạy **2 phiên notebook song song** trên cùng một tài khoản. Nghĩa là YOLOv8 và Faster R-CNN có thể chạy cùng lúc bên Kaggle, trong khi RT-DETR chạy độc lập bên Colab. Cả ba khởi động gần như đồng thời.

Điểm mấu chốt khiến việc này khả thi: **cả ba đều gọi chung một script `split_dataset.py` với `random_seed=42`**. Seed cố định nên dù chạy trên máy nào, lúc nào, phép chia dữ liệu vẫn ra kết quả y hệt. Không cần đồng bộ file dữ liệu giữa hai nền tảng, cũng không sợ ba mô hình học trên ba tập khác nhau.

> **Cảnh báo về hạn mức:** Kaggle cấp 30 giờ GPU mỗi tuần và reset vào thứ Bảy. Hai mô hình chạy song song sẽ **đốt hạn mức gấp đôi tốc độ** — chạy 6 giờ đồng hồ thực tế là tiêu 12 giờ hạn mức. Hãy tính toán trước khi bấm nút.

---

## PHẦN 1: THIẾT LẬP KAGGLE (YOLOv8 + FASTER R-CNN)

### Bước 1.1: Tạo notebook và gắn dataset

1. Vào [kaggle.com](https://www.kaggle.com) $\rightarrow$ **Create** $\rightarrow$ **New Notebook**.
2. Thanh bên phải, mục **Input** $\rightarrow$ **Add Input**.
3. Tìm `za-traffic-2020` (tác giả `phhasian0710`) $\rightarrow$ bấm **Add**.
4. Xác nhận đã thấy dữ liệu tại `/kaggle/input/za-traffic-2020/`.

> **Vì sao mount Kaggle Dataset nhanh hơn tải về?** Kaggle Dataset được gắn vào phiên chạy dưới dạng ổ đĩa chỉ đọc gắn sẵn, tốc độ đọc gần như đọc ổ cứng cục bộ, và **không tốn một giây nào** cho việc tải hay giải nén. Nếu tải file zip về rồi giải nén như bên Colab, bạn mất khoảng 5-10 phút mỗi lần khởi động lại phiên. Đây chính là lý do nên ưu tiên Kaggle cho hai mô hình nặng.

### Bước 1.2: Cấu hình phần cứng

Mục **Settings** ở thanh bên phải:

| Mục | Giá trị | Ghi chú |
|---|---|---|
| **Accelerator** | `GPU P100` | P100 có 16GB VRAM, mạnh hơn một GPU T4 đơn lẻ cho tác vụ train |
| **Internet** | `On` | Bắt buộc — cần tải `split_dataset.py` từ GitHub, trọng số pretrained và thư viện `torchmetrics` |
| **Persistence** | `Files only` | Giữ lại file output giữa các lần chạy |
| **Environment** | `Always use latest` | Đảm bảo `ultralytics` đủ mới |

> **Về lựa chọn `T4 x2`:** Ultralytics hỗ trợ chạy 2 GPU bằng `device=[0, 1]`, nhưng với dataset chỉ ~4500 ảnh thì chi phí đồng bộ giữa hai GPU thường ăn hết phần lợi. Cứ để `P100` cho đơn giản và dễ giải thích số liệu.

### Bước 1.3: Nạp notebook và chạy

1. **File** $\rightarrow$ **Import Notebook** $\rightarrow$ tải `train_yolov8_v3.ipynb` từ máy lên.
2. Bấm **Run All**.
3. Khi thấy log epoch đầu tiên chạy ổn định, **mở tab mới** và lặp lại toàn bộ Bước 1.1 đến 1.3 cho `train_faster_rcnn_v3.ipynb`. Đây chính là phiên chạy song song thứ hai.

### Bước 1.4: Bật "Save Version" để chạy nền

Đây là mẹo quan trọng nhất khi train dài trên Kaggle. Nếu chỉ bấm **Run All** rồi để tab trình duyệt mở, phiên sẽ chết ngay khi máy bạn mất mạng hoặc sập nguồn.

Thay vào đó: bấm **Save Version** $\rightarrow$ chọn **Save & Run All (Commit)**. Kaggle sẽ chạy notebook trên máy chủ của họ hoàn toàn độc lập với trình duyệt của bạn. Tắt máy đi ngủ vẫn không sao, sáng mai vào xem kết quả.

> **Giới hạn cần nhớ:** mỗi phiên Commit tối đa **12 giờ**. Nếu 100 epoch của YOLOv8 ở `imgsz=1280` có nguy cơ vượt mốc này, hãy hạ `imgsz` xuống `960` hoặc giảm trần `epochs` — và nhớ cập nhật lại `models_specs.md` cho khớp.

### Bước 1.5: Lấy kết quả về

Sau khi chạy xong, vào tab **Output** của phiên và tải các file:

```text
yolov8_v3_results.zip          <- weights + results.csv + JSON + biểu đồ
faster_rcnn_v3_results.zip
```

---

## PHẦN 2: THIẾT LẬP GOOGLE COLAB (RT-DETR)

### Bước 2.1: Chuẩn bị `kaggle.json`

Colab không có sẵn dataset như Kaggle, phải tải về qua Kaggle API.

1. Vào Kaggle $\rightarrow$ ảnh đại diện $\rightarrow$ **Settings** $\rightarrow$ mục **API** $\rightarrow$ **Create New Token**. Trình duyệt tải xuống file `kaggle.json`.
2. Mở [colab.research.google.com](https://colab.research.google.com) $\rightarrow$ **File** $\rightarrow$ **Upload notebook** $\rightarrow$ chọn `train_rtdetr_v3.ipynb`.
3. Trong Colab, mở biểu tượng **thư mục** ở thanh trái $\rightarrow$ bấm nút **Upload** $\rightarrow$ chọn file `kaggle.json` vừa tải.

> **Cảnh báo bảo mật:** `kaggle.json` chứa API key cá nhân của bạn. Tuyệt đối không commit file này lên GitHub. Repo đã có `.gitignore`, nhưng vẫn nên kiểm tra lại bằng `git status` trước khi commit.

### Bước 2.2: Bật GPU

**Runtime** $\rightarrow$ **Change runtime type** $\rightarrow$ **Hardware accelerator** = `T4 GPU` $\rightarrow$ **Save**.

Kiểm tra nhanh bằng cách chạy `!nvidia-smi` — phải thấy dòng `Tesla T4`.

### Bước 2.3: Mount Google Drive (bắt buộc, không được bỏ qua)

Cell 1 của notebook làm việc này:

```python
from google.colab import drive
import os

drive.mount('/content/drive')

SAVE_DIR = '/content/drive/MyDrive/DoAn_NhanDienBienBao'
os.makedirs(SAVE_DIR, exist_ok=True)
```

Khi chạy, Colab hiện popup yêu cầu chọn tài khoản Google và cấp quyền. Sau khi đồng ý, thư mục Drive của bạn xuất hiện tại `/content/drive/MyDrive/`.

**Vì sao đây là bước bắt buộc?** Colab bản miễn phí ngắt phiên bất kỳ lúc nào (thường sau 4-6 giờ, hoặc sớm hơn nếu bạn không tương tác). Khi ngắt, **toàn bộ thư mục `/content/` bị xóa sạch** — bao gồm cả checkpoint đã train được nửa chừng. Ghi thẳng vào Drive thì dù phiên có chết, trọng số vẫn còn nguyên.

Notebook đã cấu hình sẵn `project=f'{SAVE_DIR}/zalo_traffic'`, nên Ultralytics ghi trực tiếp `best.pt` vào Drive sau mỗi lần cải thiện, không cần copy thủ công.

### Bước 2.4: Chạy và theo dõi

Bấm **Runtime** $\rightarrow$ **Run all**. Thứ tự các cell: mount Drive $\rightarrow$ tải dataset $\rightarrow$ chia 70/10/20 $\rightarrow$ cài `ultralytics` $\rightarrow$ train $\rightarrow$ xuất JSON $\rightarrow$ vẽ Learning Curve.

**Chống ngắt phiên:** cứ khoảng 30-60 phút quay lại tab Colab và tương tác một chút (cuộn chuột, bấm vào một cell). Colab phát hiện tab để yên quá lâu sẽ coi là bỏ bê và thu hồi GPU.

> **Nếu phiên vẫn bị ngắt giữa chừng:** không mất gì cả. Chạy lại notebook từ đầu, `split_dataset.py` với seed 42 sẽ tái tạo đúng tập dữ liệu cũ. Muốn train tiếp từ checkpoint thay vì làm lại từ đầu thì đổi cell train thành `model = RTDETR(f'{SAVE_DIR}/zalo_traffic/rtdetr_v3/weights/last.pt')` và thêm `resume=True`.

### Bước 2.5: Lấy kết quả về

Vào Google Drive, thư mục `DoAn_NhanDienBienBao/`:

```text
DoAn_NhanDienBienBao/
├── zalo_traffic/rtdetr_v3/weights/best.pt
├── rtdetr_training_history.json
└── rtdetr_learning_curve.png
```

---

## PHẦN 3: SAU KHI CẢ BA ĐÃ TRAIN XONG

### Bước 3.1: Gom trọng số về một chỗ

Tạo một thư mục trên máy và đặt tên phẳng, rõ ràng:

```text
D:\zalo_traffic_3_models\
├── 1_YOLOv8_P2_HighRes.pt              <- từ Kaggle
├── 2_FasterRCNN_ResNet50.pth           <- từ Kaggle
├── 3_RTDETR_Large_Transformer.pt       <- từ Google Drive
├── yolov8_training_history.json
├── faster_rcnn_training_history.json
└── rtdetr_training_history.json
```

### Bước 3.2: Đối chiếu số epoch thực tế

Trước khi chạy đánh giá, mở ba file JSON và ghi lại con số này — nó sẽ vào thẳng bảng kết quả trong báo cáo:

```python
import json

for ten_file in ['yolov8', 'faster_rcnn', 'rtdetr']:
    du_lieu = json.load(open(f'{ten_file}_training_history.json', encoding='utf-8'))
    print(f"{du_lieu['model']:25s} chạy {du_lieu['epochs_thuc_te']:3d}/{du_lieu['epochs_du_kien']} epochs")
```

Con số này trả lời một câu hỏi hay cho phần phân tích: **mô hình nào hội tụ nhanh hơn trên bộ dữ liệu này?** Nếu RT-DETR chạy hết gần 100 epoch trong khi YOLOv8 dừng ở khoảng 40, đó là bằng chứng thực nghiệm cho luận điểm "Transformer hội tụ chậm hơn CNN" mà bạn nêu trong `models_specs.md` mục 0.2.

### Bước 3.3: Chuyển sang bước đánh giá

Đóng gói ba trọng số lên Kaggle Dataset rồi chạy `evaluate_3_models.ipynb`. Quy trình chi tiết ở `docs/05_testing_evaluation/test_KeHoach_Kaggle.md`.

---

## PHẦN 4: CHECKLIST TRƯỚC KHI BẤM RUN

**Kaggle (làm 2 lần, cho 2 notebook):**

- [ ] Đã Add Input dataset `za-traffic-2020`.
- [ ] Accelerator = `GPU P100`, Internet = `On`.
- [ ] Đã bấm **Save Version → Save & Run All (Commit)** thay vì chỉ Run All.
- [ ] Còn đủ hạn mức GPU trong tuần (kiểm tra ở trang **Settings** của tài khoản).

**Colab:**

- [ ] Đã upload `kaggle.json` vào thư mục làm việc.
- [ ] Runtime type = `T4 GPU`, đã chạy `!nvidia-smi` xác nhận.
- [ ] Đã mount Drive thành công và thấy thư mục `DoAn_NhanDienBienBao`.

**Chung cho cả ba:**

- [ ] Cell chạy `split_dataset.py` in ra đúng thứ tự **Test 20.0% → Val 10.0% → Train 70.0%**.
- [ ] File `data.yaml` in ra chỉ có `path`, `train`, `val`, `names` — **không có khóa `test`**.
- [ ] Cell train hiển thị `epochs=100` và `patience=15`.
