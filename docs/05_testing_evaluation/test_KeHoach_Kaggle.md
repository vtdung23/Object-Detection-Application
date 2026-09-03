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

### 0.2. Tập Hold-out Test 20% được tách ra và giấu đi từ đầu

Từ phiên bản **V3** trở đi, tập Test không còn được "tái lập lại từ tập Validation" như trước nữa, mà được cắt ra và giấu đi ngay từ khâu chuẩn bị dữ liệu.

Danh sách ảnh được xáo trộn bằng seed 42, rồi cắt lần lượt theo thứ tự **Test → Val → Train**:

| Thứ tự cắt | Tập | Tỷ lệ | Ai được nhìn thấy |
|---|---|---|---|
| 1 | **Hold-out Test** | **20% đầu** | **Không mô hình nào.** Chỉ mở ra đúng một lần tại notebook đánh giá |
| 2 | Validation | 10% tiếp theo | Cả 3 mô hình, chỉ để chọn `best.pt` và kích hoạt Early Stopping |
| 3 | Train | ~70% còn lại | Cả 3 mô hình, để cập nhật trọng số |

Cắt tập Test ra trước tiên là có chủ đích: nhờ vậy tập Test luôn là 20% đầu của danh sách đã xáo trộn, nên sau này có chỉnh tỷ lệ Train/Val thế nào thì nó vẫn giữ nguyên đúng những bức ảnh cũ — kết quả các lần chạy khác nhau vẫn so sánh được với nhau.

Ba cơ chế bảo đảm tính sạch của tập Hold-out:

1. **Một nguồn chia duy nhất.** Cả 3 notebook V3 đều gọi chung `data_preparation/split_dataset.py` với `random_seed = 42`. Không notebook nào tự viết đoạn chia riêng, nên không thể lệch nhau.
2. **Thư mục vật lý tách rời.** Tập Test nằm ở `holdout_test/`, hoàn toàn bên ngoài `dataset_train_val/`. File `data.yaml` **không có khóa `test:`** — nếu có, Ultralytics sẽ tự động đánh giá trên đó và làm rò rỉ thông tin vào quá trình chọn mô hình.
3. **Biên bản đối chiếu.** Script xuất `split_manifest.json` ghi lại danh sách `image_id` của cả 3 tập, kèm bước kiểm tra chéo (`assert`) chắc chắn không có ảnh nào xuất hiện ở hai tập cùng lúc.

> **So với phiên bản cũ:** Bản V1/V2 chia 80/20 cho YOLOv8 và RT-DETR (không hề có tập Test), còn Faster R-CNN chia 90/10 bằng `random_split()` không set seed. Hệ quả là ba mô hình học trên ba tập khác nhau, và tập dùng để chấm điểm chính là tập Validation — tức là tập đã được dùng để chọn `best.pt`. Điểm số vì thế lạc quan hơn thực tế. Bản V3 sinh ra để khắc phục đúng vấn đề này. Chi tiết kỹ thuật ở `models_specs.md` mục 0.1.

### 0.3. Early Stopping và kế hoạch vẽ Learning Curve

Cả 3 mô hình đều đặt trần `epochs = 100` và dừng sớm khi 15 epoch liên tiếp không cải thiện mAP trên tập Validation.

**Vì sao không dùng số epoch cố định?** CNN và Transformer hội tụ với tốc độ rất khác nhau. YOLOv8 bắt nhịp nhanh nhờ có sẵn quy nạp cục bộ về không gian; RT-DETR phải tự học quan hệ không gian từ con số 0 nên chậm hơn hẳn ở giai đoạn đầu. Ép cả ba chạy cứng 50 epoch thì mô hình chậm bị cắt ngang lúc chưa chín, mô hình nhanh thì thừa ra hàng chục epoch chỉ để overfitting. Cho cả ba cùng trần 100 epoch rồi để Early Stopping tự quyết định mới là so sánh công bằng.

**Số epoch thực tế mỗi mô hình dùng cũng là một số liệu đáng đưa vào báo cáo** — nó cho biết mô hình nào "học nhanh" hơn trên bộ dữ liệu này.

Sau khi train, mỗi mô hình sinh ra một file nhật ký JSON:

| Mô hình | File nhật ký | Cách sinh ra |
|---|---|---|
| YOLOv8s-P2 | `yolov8_training_history.json` | Parse `results.csv` của Ultralytics |
| RT-DETR-L | `rtdetr_training_history.json` | Parse `results.csv` (dùng chung hàm với YOLOv8) |
| Faster R-CNN | `faster_rcnn_training_history.json` | Ghi thẳng trong vòng lặp `for epoch`, `json.dump()` sau mỗi epoch |

Mỗi file chứa mảng theo từng epoch với 5 trường: `epoch_id`, `train_loss`, `val_loss`, `mAP_50`, `mAP_50_95`.

**Kế hoạch dùng Learning Curve trong báo cáo:**

- **Biểu đồ Loss** (`train_loss` và `val_loss` chung một trục): dùng để chứng minh mô hình **không bị overfitting**. Dấu hiệu overfitting là `val_loss` quay đầu đi lên trong khi `train_loss` vẫn tiếp tục giảm — nếu Early Stopping làm đúng việc thì điểm dừng phải rơi ngay quanh chỗ đường `val_loss` chạm đáy.
- **Biểu đồ mAP** (`mAP_50` và `mAP_50_95`): dùng để xác định **điểm hội tụ thật sự**, và để đối chiếu xem `best.pt` được chọn có đúng là đỉnh của đường cong không.
- **Biểu đồ chồng 3 mô hình**: vẽ `mAP_50_95` của cả ba lên cùng một trục để so sánh trực quan tốc độ hội tụ giữa CNN và Transformer. Đây là hình minh họa mạnh nhất cho luận điểm ở mục 0.3.

Hàm `ve_learning_curve()` đã được viết sẵn trong cả 3 notebook V3, tự động lưu ảnh PNG cạnh file trọng số.

---

## PHẦN 1: ĐÓNG GÓI 3 BỘ TRỌNG SỐ VÀ ĐẨY LÊN KAGGLE DATASETS

Vì dataset ảnh đã có sẵn công khai trên Kaggle, ta **chỉ cần upload trọng số (weights)**.

### Bước 1.1: Gom 3 file weights về một thư mục trên máy Local

Sau khi train xong bằng các notebook trong `notebooks/v3/`, file trọng số nằm ở những đường dẫn sau (trích trực tiếp từ mã nguồn):

| Mô hình | Đường dẫn sau khi train (bản V3) | Nền tảng | Dung lượng ước tính |
|---|---|---|---|
| YOLOv8s-P2 | `/kaggle/working/zalo_traffic/yolov8s_p2_v3/weights/best.pt` | Kaggle | ~20-25 MB |
| Faster R-CNN | `/kaggle/working/faster_rcnn_v3/faster_rcnn_best.pth` | Kaggle | ~160-170 MB |
| RT-DETR-L | `<Drive>/DoAn_NhanDienBienBao/zalo_traffic/rtdetr_v3/weights/best.pt` | Colab | ~63-66 MB |

Tạo một thư mục sạch trên máy và copy 3 file vào, **đặt tên phẳng và rõ ràng** để tránh nhầm lẫn. Nên gom luôn 3 file nhật ký JSON vào cùng chỗ để tiện vẽ Learning Curve về sau:

```text
D:\zalo_traffic_3_models\
├── 1_YOLOv8_P2_HighRes.pt
├── 2_FasterRCNN_ResNet50.pth
├── 3_RTDETR_Large_Transformer.pt
├── yolov8_training_history.json
├── faster_rcnn_training_history.json
└── rtdetr_training_history.json
```

> **Lưu ý về định dạng checkpoint của Faster R-CNN:** từ bản V3, file `.pth` lưu dưới dạng dictionary gồm `model_state_dict`, `anchor_sizes`, `aspect_ratios`, `num_classes` chứ không phải `state_dict` trần như bản cũ. Lý do là bộ Anchor K-Means giờ chỉ tính trên tập Train, nên phải lưu kèm mới nạp lại đúng được. Notebook đánh giá đã xử lý được cả hai định dạng.

> **Vì sao để phẳng mà không tạo thư mục con?** Lệnh `kaggle datasets create` mặc định **bỏ qua các thư mục con** (`--dir-mode skip`). Nếu bạn tạo 3 thư mục con thì upload xong sẽ thấy dataset rỗng. Để file phẳng là cách an toàn nhất. (Nếu bạn thích cấu trúc thư mục thì phải upload bằng giao diện Web ở Cách B bên dưới).

### Bước 1.2 — Cách A: Upload bằng Kaggle CLI (khuyên dùng, nhanh và lặp lại được)

Mở PowerShell tại thư mục vừa tạo:

```powershell
# Cài công cụ dòng lệnh của Kaggle
pip install kaggle

# Đăng nhập qua trình duyệt, CLI tự lưu thông tin xác thực
python -m kaggle auth login

# Sinh ra file mô tả dataset
cd "D:\zalo_traffic_3_models"
python -m kaggle datasets init -p .
```

> **Về việc xác thực:** Kaggle đã bỏ cơ chế tự tải `kaggle.json` khi tạo token. Bây giờ trang `kaggle.com/settings/api` chỉ hiện ra một chuỗi `KGAT_...` (chỉ hiện một lần duy nhất). Nếu `kaggle auth login` không chạy được trên bản `kaggle` đang cài, có hai cách thay thế: đặt biến môi trường `KAGGLE_API_TOKEN` bằng chuỗi đó, hoặc kéo xuống mục **Legacy API Credentials** trên cùng trang, bấm **Create Legacy API Key** để tải file `kaggle.json` như kiểu cũ rồi chép vào `%USERPROFILE%\.kaggle\`.

Lệnh trên tạo file `dataset-metadata.json`. Mở nó ra và sửa lại (thay `your-username` bằng username Kaggle của bạn):

```json
{
  "title": "Zalo Traffic Sign - Weights 3 Models V3",
  "id": "vtdungfitus/zalo-traffic-3-models-v3",
  "licenses": [{"name": "CC0-1.0"}]
}
```

Đẩy lên Kaggle:

```powershell
python -m kaggle datasets create -p . -t
```

Sau này nếu train lại và muốn cập nhật weights mới, **không tạo dataset mới** mà tạo phiên bản mới:

```powershell
python -m kaggle datasets version -p . -m "Cap nhat trong so RT-DETR"
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

Notebook đánh giá chỉ cần chạy lại đúng script mà 3 notebook huấn luyện đã dùng:

```bash
python split_dataset.py --output-root /kaggle/working/data_v3
```

Script này chia dữ liệu bằng `random.seed(42)` cố định, nên chạy ở bất kỳ đâu, bất kỳ lúc nào cũng cho ra **đúng danh sách ảnh giống hệt**. Con số seed chính là "bản hợp đồng" đảm bảo tính tái lập — không cần đóng gói và upload tập test.

Kết quả sinh ra hai nhánh tách rời:

```text
/kaggle/working/data_v3/
├── dataset_train_val/     <- 3 model đã dùng nhánh này để train
│   └── data.yaml          (chỉ có train + val, KHÔNG có khóa test)
├── holdout_test/          <- notebook đánh giá CHỈ đọc nhánh này
│   ├── images/, labels/
│   └── holdout_test_annotations.json
└── split_manifest.json    <- biên bản đối chiếu
```

**Ưu điểm:** Không tốn dung lượng, không sợ upload nhầm phiên bản, và người chấm có thể tự kiểm chứng bằng cách chạy lại đúng script đó rồi đối chiếu `split_manifest.json`.

**Điều kiện bắt buộc:** Thứ tự `images` trong file JSON gốc phải giữ nguyên. Điều này luôn đúng vì mọi thứ đều đọc từ cùng một file `train_traffic_sign_dataset.json` bất biến trên Kaggle, và `dict` trong Python 3.7+ giữ nguyên thứ tự chèn.

### Kiểm tra chéo: tập Hold-out có thật sự sạch không?

Trước khi chạy đánh giá, nên xác minh nhanh bằng `split_manifest.json`:

```python
import json

bien_ban = json.load(open('/kaggle/working/data_v3/split_manifest.json', encoding='utf-8'))
tap_train = set(bien_ban['image_ids']['train'])
tap_val = set(bien_ban['image_ids']['val'])
tap_test = set(bien_ban['image_ids']['holdout_test'])

print('Train ∩ Test:', len(tap_train & tap_test))   # phải bằng 0
print('Val   ∩ Test:', len(tap_val & tap_test))     # phải bằng 0
print('Seed đã dùng:', bien_ban['random_seed'])     # phải bằng 42
```

Bản thân `split_dataset.py` cũng đã có sẵn ba lệnh `assert` kiểm tra điều này và sẽ dừng ngay nếu phát hiện trùng lặp.

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
| **Accelerator** | `GPU T4 x2` | **Chọn T4 x2, không chọn P100.** Bản PyTorch của image Kaggle hiện tại dùng CUDA 12.8 nên đã bỏ hỗ trợ kiến trúc Pascal `sm_60` của P100 — chọn P100 là gặp lỗi `no kernel image is available for execution on the device`. Chọn `T4 x2` nhưng code phải khóa cứng `device='cuda:0'` để **chỉ dùng một GPU**: bài toán đo FPS chạy suy luận từng ảnh một (`batch=1`) nên GPU thứ hai không giúp tăng tốc, mà còn khiến số liệu khó giải thích trước hội đồng. |
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
- [ ] Accelerator đã bật **GPU T4 x2** (không phải P100, không phải None, không phải CPU).
- [ ] **Internet = On**.
- [ ] Cả 3 file weights đều hiện diện trong `/kaggle/input/`, không file nào bị lỗi upload 0 KB.
- [ ] Cả 3 weights đều là **bản V3** (train bằng notebook trong `notebooks/v3/`). Nếu trộn lẫn weights V2 cũ vào thì bảng so sánh mất giá trị, vì model V2 đã học trên tập dữ liệu khác.
- [ ] Đã chạy `split_dataset.py` và kiểm tra `split_manifest.json` cho thấy 3 tập không giao nhau.
- [ ] Đã tải về đủ 3 file `*_training_history.json` để vẽ Learning Curve cho báo cáo.
