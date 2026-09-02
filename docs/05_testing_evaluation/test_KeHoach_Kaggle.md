# KẾ HOẠCH TRIỂN KHAI LUỒNG TESTING & EVALUATION TRÊN KAGGLE

Tài liệu này hướng dẫn từng bước chuyển luồng đánh giá (Evaluation) từ máy Local lên Kaggle để chấm điểm 3 mô hình trên **cùng một tập Test**, phục vụ bảng so sánh trong báo cáo môn học.

**File code đi kèm:** `Traffic-Sign-Detection-ZaloAI/notebooks/evaluate_3_models.ipynb`

---

## PHẦN 0: HAI ĐIỀU BẮT BUỘC PHẢI BIẾT TRƯỚC KHI LÀM

Đây là hai giới hạn có thật của dự án. Phải hiểu rõ để không báo cáo sai với hội đồng.

### 0.1. Thư mục `traffic_public_test/` KHÔNG dùng để tính mAP được
Cấu trúc dataset gốc trên Kaggle như sau (trích `ke_hoach_de_1.md`):

```text
/kaggle/input/za-traffic-2020/za_traffic_2020/
├── traffic_train/
│   ├── images/                           # ~4500 ảnh có nhãn
│   └── train_traffic_sign_dataset.json   # Toàn bộ nhãn COCO nằm ở đây
└── traffic_public_test/
    └── images/                           # CHỈ CÓ ẢNH, KHÔNG CÓ NHÃN
```

Vì `traffic_public_test/` không có file nhãn, ta **không thể** tính `mAP@50` hay vẽ Confusion Matrix trên đó (không có Ground Truth để đối chiếu). Nó chỉ dùng để xem kết quả bằng mắt (qualitative demo).

$\Rightarrow$ **Kết luận:** Tập Test dùng để chấm điểm bắt buộc phải cắt ra từ 4500 ảnh có nhãn trong `traffic_train/`.

### 0.2. Điều kiện đánh giá của 3 mô hình không hoàn toàn đồng nhất

Ta tái lập lại đúng phép chia dữ liệu đã dùng lúc huấn luyện (`random.seed(42)`, shuffle, lấy 20% cuối) để làm tập Test. Cả 3 mô hình đều chỉ đọc dữ liệu từ `traffic_train/`, **không mô hình nào đụng tới `traffic_public_test/`**. Vì vậy ở đây **không có Data Leakage** theo nghĩa "học lỏm tập test của cuộc thi" — bản thân tập test đó còn không có nhãn.

Vấn đề nằm ở chỗ khác và nhẹ hơn: cách chia train/val của 3 mô hình không giống nhau.

| Mô hình | Cách chia lúc train | Vị thế của mô hình trên tập 20% này |
|---|---|---|
| YOLOv8-P2 | `random.seed(42)` + shuffle, cắt 80/20 | Đây đúng là tập **Validation** của nó — chưa từng học trực tiếp, chỉ dùng để chọn `best.pt` |
| RT-DETR | Cùng đoạn code với YOLOv8 | Trùng khớp hoàn toàn với YOLOv8 |
| Faster R-CNN | `random_split(...)` cắt 90/10, **không set seed** | ⚠️ Không tái lập được. Về mặt xác suất, khoảng 90% ảnh trong tập 20% này đã nằm trong phần train của nó |

**Ý nghĩa thực tế:** Đây là hiện tượng **Overfitting** lộ ra chứ không phải gian lận dữ liệu. Faster R-CNN được chấm trên những bức ảnh nó đã học thuộc, còn YOLOv8 và RT-DETR bị chấm trên ảnh hoàn toàn mới. Nên nếu Faster R-CNN có mAP cao bất thường thì đó phần lớn là điểm "học thuộc lòng", không phản ánh khả năng tổng quát hóa.

Hai cách xử lý:

- **Cách nhanh (chấp nhận được cho đồ án):** Cứ chạy đánh giá và **ghi rõ chú thích này** dưới bảng kết quả. Trung thực, tốn ít thời gian, và bản thân việc chỉ ra được chênh lệch điều kiện này cũng là một điểm phân tích tốt trong báo cáo.
- **Cách chuẩn mực hơn:** Sửa `train_faster_rcnn.ipynb` dùng `random.seed(42)` và cắt 80/20 y hệt 2 mô hình kia, rồi train lại. Tốn thêm thời gian GPU nhưng bảng so sánh sẽ đặt cả 3 lên cùng một vạch xuất phát.

---

## PHẦN 1: ĐÓNG GÓI 3 BỘ TRỌNG SỐ VÀ ĐẨY LÊN KAGGLE DATASETS

Vì dataset ảnh đã có sẵn công khai trên Kaggle, ta **chỉ cần upload trọng số (weights)**.

### Bước 1.1: Gom 3 file weights về một thư mục trên máy Local

Sau khi train xong, các file trọng số nằm rải rác ở những đường dẫn sau (trích trực tiếp từ mã nguồn 3 notebook huấn luyện):

| Mô hình | Đường dẫn file gốc sau khi train | Dung lượng ước tính |
|---|---|---|
| YOLOv8s-P2 | `zalo_traffic/yolov8s_p2_highres/weights/best.pt` | ~20-25 MB |
| Faster R-CNN | `faster_rcnn_highres/faster_rcnn_best.pth` | ~160-170 MB |
| RT-DETR-L | `zalo_traffic/rtdetr_kaggle_640/weights/best.pt` (bản Kaggle)<br>hoặc `zalo_traffic/rtdetr_colab_highres/weights/best.pt` (bản Colab) | ~63-66 MB |

Tạo một thư mục sạch trên máy và copy 3 file vào, **đặt tên phẳng và rõ ràng** để tránh nhầm lẫn:

```text
D:\zalo_traffic_3_models\
├── 1_YOLOv8_P2_HighRes.pt
├── 2_FasterRCNN_ResNet50.pth
└── 3_RTDETR_Large_Transformer.pt
```

> **Vì sao để phẳng mà không tạo thư mục con?** Lệnh `kaggle datasets create` mặc định **bỏ qua các thư mục con** (`--dir-mode skip`). Nếu bạn tạo 3 thư mục con thì upload xong sẽ thấy dataset rỗng. Để file phẳng là cách an toàn nhất. (Nếu bạn thích cấu trúc thư mục thì phải upload bằng giao diện Web ở Cách B bên dưới).

### Bước 1.2 — Cách A: Upload bằng Kaggle CLI (khuyên dùng, nhanh và lặp lại được)

Mở PowerShell tại thư mục vừa tạo:

```powershell
# Cài công cụ dòng lệnh của Kaggle
pip install kaggle

# Đặt file kaggle.json (tải từ Kaggle > Settings > API > Create New Token) vào đúng chỗ
mkdir "$env:USERPROFILE\.kaggle" -Force
copy "$env:USERPROFILE\Downloads\kaggle.json" "$env:USERPROFILE\.kaggle\kaggle.json"

# Sinh ra file mô tả dataset
cd "D:\zalo_traffic_3_models"
kaggle datasets init -p .
```

Lệnh trên tạo file `dataset-metadata.json`. Mở nó ra và sửa lại (thay `your-username` bằng username Kaggle của bạn):

```json
{
  "title": "Zalo Traffic Sign - 3 Trained Models",
  "id": "your-username/zalo-traffic-3-models",
  "licenses": [{ "name": "CC0-1.0" }]
}
```

Đẩy lên Kaggle:

```powershell
kaggle datasets create -p . -m "Weights 3 model: YOLOv8-P2, Faster R-CNN, RT-DETR"
```

Sau này nếu train lại và muốn cập nhật weights mới, **không tạo dataset mới** mà tạo phiên bản mới:

```powershell
kaggle datasets version -p . -m "Cap nhat weights sau khi train lai 50 epochs"
```

### Bước 1.2 — Cách B: Upload bằng giao diện Web (nếu ngại cài CLI)

1. Vào [kaggle.com/datasets](https://www.kaggle.com/datasets) $\rightarrow$ bấm **New Dataset**.
2. Đặt tiêu đề: `Zalo Traffic Sign - 3 Trained Models`.
3. Kéo thả cả 3 file (hoặc kéo nguyên thư mục — cách này **giữ được** cấu trúc thư mục con).
4. Bấm **Create**. Chờ Kaggle xử lý xong (trạng thái chuyển từ *Processing* sang *Ready*).

> **Mẹo:** Để chế độ **Private** nếu chưa muốn công khai. Notebook của chính bạn vẫn mount được dataset private bình thường.

### Bước 1.3: Kiểm tra lại đường dẫn sau khi mount
Khi dataset được gắn vào notebook, nó sẽ nằm ở `/kaggle/input/<slug-dataset>/`. Ví dụ:

```text
/kaggle/input/zalo-traffic-3-models/
├── 1_YOLOv8_P2_HighRes.pt
├── 2_FasterRCNN_ResNet50.pth
└── 3_RTDETR_Large_Transformer.pt
```

Notebook đánh giá đã được viết để **tự động dò tìm** (`glob`) file `.pt` / `.pth` theo từ khóa tên, nên dù bạn upload phẳng hay theo thư mục thì code vẫn chạy được.

---

## PHẦN 2: CHUẨN BỊ TẬP HOLD-OUT TEST (KHÔNG CẦN UPLOAD)

Vì bộ dữ liệu gốc đã có sẵn công khai tại [phhasian0710/za-traffic-2020](https://www.kaggle.com/datasets/phhasian0710/za-traffic-2020), ta **không phải upload gì thêm**. Chỉ cần gắn dataset đó vào notebook rồi tái lập lại đúng phép chia đã dùng lúc train.

### Nguyên tắc: Tái lập bằng Seed thay vì upload file
Cả `train_yolov8.ipynb` và `train_rtdetr.ipynb` đều chia dữ liệu bằng đúng đoạn code sau:

```python
image_ids = list(images_info.keys())
random.seed(42)
random.shuffle(image_ids)
split_idx = int(len(image_ids) * 0.8)
train_ids = image_ids[:split_idx]
val_ids   = image_ids[split_idx:]   # <-- 20% này chính là tập Hold-out Test của ta
```

Vì `random.seed(42)` là cố định, chạy lại đoạn code này **ở bất kỳ đâu cũng cho ra đúng danh sách ảnh giống hệt**. Đây chính là lý do ta không cần đóng gói và upload tập test — bản thân con số seed đã là "bản hợp đồng" đảm bảo tính tái lập.

**Ưu điểm:** Không tốn dung lượng, không sợ upload nhầm phiên bản, và người chấm có thể tự kiểm chứng lại bằng cách chạy đúng đoạn code trên.

**Điều kiện bắt buộc:** Danh sách `images_info.keys()` phải giữ nguyên thứ tự như lúc train. Điều này luôn đúng vì cả hai đều đọc từ cùng một file `train_traffic_sign_dataset.json` bất biến, và `dict` trong Python 3.7+ giữ nguyên thứ tự chèn.

### (Tùy chọn) Khi nào mới nên đóng băng tập test thành Dataset riêng?
Chỉ nên làm khi bạn **train lại cả 3 model** theo Cách chuẩn mực ở mục 0.2. Lúc đó hãy tách 15% ảnh ra thành một dataset riêng, không cho bất kỳ mô hình nào nhìn thấy, rồi upload y hệt quy trình ở Phần 1. Với hiện trạng đồ án (model đã train xong), việc này không còn ý nghĩa vì dữ liệu đã bị "nhìn" mất rồi.

---

## PHẦN 3: CẤU HÌNH KAGGLE NOTEBOOK

### Bước 3.1: Tạo Notebook và gắn dữ liệu
1. Vào Kaggle $\rightarrow$ **Create** $\rightarrow$ **New Notebook**.
2. Ở thanh bên phải, mở mục **Input** $\rightarrow$ bấm **Add Input**.
3. Gắn **hai** nguồn dữ liệu vào cùng một phiên chạy:
   - Tìm `za-traffic-2020` (tác giả `phhasian0710`) $\rightarrow$ **Add**. Đây là ảnh + nhãn.
   - Chuyển sang tab **Your Datasets** $\rightarrow$ chọn `zalo-traffic-3-models` $\rightarrow$ **Add**. Đây là trọng số.
4. Kiểm tra lại cây thư mục `/kaggle/input/` phải thấy đủ 2 thư mục con.

### Bước 3.2: Chọn phần cứng (quan trọng cho phần đo FPS)

Vào **Settings** ở thanh bên phải:

| Mục | Giá trị cần chọn | Lý do |
|---|---|---|
| **Accelerator** | `GPU P100` | **Chọn P100, không chọn T4 x2.** Bài toán đo FPS chạy suy luận từng ảnh một (`batch=1`) nên GPU thứ hai hoàn toàn không giúp tăng tốc, mà còn khiến số liệu khó giải thích. Một GPU duy nhất cho kết quả FPS sạch và dễ bảo vệ trước hội đồng. Nếu hết hạn mức P100 thì chọn `T4 x2` nhưng phải khóa cứng `device='cuda:0'` trong code. |
| **Internet** | `On` | Cần tải thư viện `torchmetrics` và trọng số backbone ResNet-50 của torchvision. |
| **Persistence** | `Files only` (hoặc để mặc định) | Giữ lại file CSV kết quả giữa các lần chạy cho tiện. |
| **Environment** | `Always use latest` | Đảm bảo `ultralytics` đủ mới để đọc được file `best.pt`. |

> **Cảnh báo hạn mức:** Kaggle cấp 30 giờ GPU miễn phí mỗi tuần. Luồng đánh giá này chạy rất nhanh (chỉ suy luận, không train), nên hãy **tắt notebook ngay sau khi chạy xong** để dành GPU cho việc train lại nếu cần.

### Bước 3.3: Nạp file notebook đánh giá
Có 2 cách đưa `evaluate_3_models.ipynb` lên Kaggle:
- **Cách nhanh:** Trong notebook trống vừa tạo, chọn **File** $\rightarrow$ **Import Notebook** $\rightarrow$ tải file `.ipynb` từ máy lên.
- **Cách thủ công:** Mở file `.ipynb` trên máy, copy nội dung từng cell rồi dán sang Kaggle.

### Bước 3.4: Chạy và lấy kết quả
1. Bấm **Run All**. Toàn bộ quá trình gồm: dựng tập test $\rightarrow$ chạy 3 model $\rightarrow$ tính mAP $\rightarrow$ đo FPS $\rightarrow$ vẽ Confusion Matrix.
2. Kết quả cuối cùng được lưu ra `/kaggle/working/`:
   - `final_comparison_table.csv` — bảng số liệu tổng hợp để dán thẳng vào báo cáo.
   - `confusion_matrix_<tên_model>.png` — 3 ảnh ma trận nhầm lẫn.
3. Tải về qua mục **Output** ở thanh bên phải.

---

## PHẦN 4: CHECKLIST TRƯỚC KHI BẤM RUN ALL

- [ ] Đã gắn đủ **2 dataset** vào Input (`za-traffic-2020` và dataset weights của mình).
- [ ] Accelerator đã bật **GPU P100** (không phải None, không phải CPU).
- [ ] **Internet = On**.
- [ ] Cả 3 file weights đều hiện diện trong `/kaggle/input/`, không file nào bị lỗi upload 0 KB.
- [ ] Đã đọc và hiểu cảnh báo ở mục **0.2** để ghi chú thích trung thực dưới bảng kết quả trong báo cáo.
