# Giải thích chi tiết mã nguồn: train_rtdetr.ipynb

Tài liệu này giải thích một cách cặn kẽ và chuyên sâu từng đoạn code trong file `train_rtdetr.ipynb`. Mục tiêu là đối chiếu và làm rõ cách mã nguồn hiện thực hóa các thông số kỹ thuật tiên tiến nhất được định nghĩa tại `models_specs.md` (Đặc biệt là phần **3. RT-DETR (Real-Time DEtection TRansformer)**) và `train_GiaiThich_Hyperparams.md`.

Khác với YOLOv8 hay Faster R-CNN, RT-DETR là một kiến trúc dựa trên **Transformer**, mang trong mình sức mạnh của cơ chế **Self-Attention** nhưng lại đòi hỏi cực kỳ khắt khe về bộ nhớ VRAM. Notebook này được sinh ra để chạy trên **Google Colab** nhằm tận dụng tối đa tài nguyên phần cứng, đồng thời tích hợp các thủ thuật "chống tràn bộ nhớ" đỉnh cao.

---

## Cell 1: Tải dữ liệu siêu tốc bằng Kaggle API

```python
# Tải bộ dữ liệu Zalo AI từ Kaggle về Google Colab bằng API
!pip install -q kaggle
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download -d phhasian0710/za-traffic-2020
!unzip -q -n za-traffic-2020.zip -d /content/dataset
```
- **Ý nghĩa Kỹ thuật**: Đoạn code này thiết lập một "đường ống" truyền dữ liệu trực tiếp từ máy chủ Kaggle sang ổ cứng SSD nội bộ (`/content/`) của máy ảo Google Colab. 
- **Giải quyết nút thắt cổ chai (Bottleneck)**: Nếu tải dữ liệu thủ công lên Google Drive rồi mount sang Colab, tốc độ đọc file ảnh trong lúc train sẽ cực kỳ chậm, làm nghẽn GPU (GPU đói dữ liệu). Việc dùng API tải thẳng file ZIP và giải nén ngay tại local ảo giúp tốc độ đọc (I/O) đạt mức tối đa.

---

## Cell 2 & 3: Tiền xử lý Dữ liệu và Chia tập Train/Val (80/20)

```python
import glob
json_paths = glob.glob('/content/dataset/**/train_traffic_sign_dataset.json', recursive=True)
...
train_dataset, val_dataset = random_split(...) # Cắt 80% Train, 20% Val
```
- **Ý nghĩa (Tránh Data Leakage)**: Tương tự như quy trình chuẩn của nhóm, code sử dụng `random.seed(42)` để xáo trộn toàn bộ danh sách 4500 tấm ảnh, sau đó cắt vạch ranh giới rõ ràng: 80% đưa vào `train` và 20% đưa vào `val`. Việc này **ngăn chặn 100% hiện tượng Rò rỉ dữ liệu (Data Leakage)** - một lỗi sơ đẳng nhưng chí mạng nếu để mô hình học thi bằng chính đề kiểm tra.

```python
def convert_coco_to_yolo(bbox, img_width, img_height):
    x_center = (x_min + w / 2) / img_width
...
```
- **Ý nghĩa Kỹ thuật**: Chuyển đổi hệ tọa độ COCO `[x_min, y_min, w, h]` (pixel tuyệt đối) sang chuẩn YOLO format `[x_center, y_center, w_norm, h_norm]` (chuẩn hóa tỷ lệ từ 0 đến 1). Mặc dù là mô hình Transformer, cấu trúc thư viện `ultralytics` vẫn đòi hỏi dữ liệu đầu vào phải tuân thủ nghiêm ngặt định dạng text này để tối ưu hóa việc nạp batch. Nhãn (label) cũng được lùi 1 đơn vị (`class_id - 1`) để khớp với index mảng bắt đầu từ 0.

---

## Cell 5: Kết nối Google Drive làm "Két sắt"

```python
from google.colab import drive
drive.mount('/content/drive')
save_dir = '/content/drive/MyDrive/DoAn_NhanDienBienBao'
```
- **Ý nghĩa DevOps**: Đây là chiến thuật **Zero-Data Loss** (Không mất mát dữ liệu). Google Colab là một nền tảng điện toán đám mây cấp phát tạm thời. Nếu người dùng tắt trình duyệt hoặc máy chủ hết phiên (Session Timeout), toàn bộ dữ liệu ở `/content/` sẽ bốc hơi sạch sẽ. Đoạn code này mở một kênh lưu trữ vĩnh viễn sang Google Drive, đảm bảo trọng số (weights) nặng hàng trăm MB được cất giữ an toàn tuyệt đối ngay cả khi máy ảo bị sập.

---

## Cell 6: Khởi tạo Kiến trúc Transformer (RT-DETR-L) và Cấu hình chống OOM

Đây là ô code đắt giá nhất và chứa nhiều hàm lượng chất xám nhất của toàn bộ quá trình huấn luyện RT-DETR.

```python
from ultralytics import RTDETR
model = RTDETR('rtdetr-l.pt')
```
- **Ý nghĩa**: Hiện thực hóa mục **3. RT-DETR** trong `models_specs.md`. Khởi tạo mạng Transformer với kích thước **Large (L)**. Thay vì dùng nhánh backbone CNN truyền thống, RT-DETR dùng cơ chế **Multi-Head Self-Attention (Q-K-V)** để quét toàn cục bức ảnh, giúp nó hiểu được bối cảnh (Context) hoàn hảo hơn hẳn YOLO. Khối lượng tham số khổng lồ (pretrained) giúp mô hình "thông minh" sẵn.

```python
results = model.train(
    data='/content/dataset.yaml',
    epochs=50,
    imgsz=640,          # BẮT BUỘC ĐỂ 640 (Nếu để 1280 sẽ mất 15 tiếng như Kaggle)
```
- **Ý nghĩa (`imgsz=640`)**: Đây là thông số đã được **hạ từ `1280` xuống `640`** so với bản kế hoạch ban đầu. Lý do kỹ thuật (biện luận đầy đủ tại mục **3.6** của `models_specs.md`):
  - **Chống tràn bộ nhớ (OOM):** Lõi Transformer phải tính ma trận Attention giữa mọi cặp token. Ảnh `1280` sinh ra lưới đặc trưng $40 \times 40 = 1{,}600$ token, tức ma trận Attention nặng $1{,}600^2 \approx 2{,}56$ triệu ô. Ảnh `640` chỉ còn lưới $20 \times 20 = 400$ token, ma trận $400^2 = 160{,}000$ ô — **nhẹ hơn 16 lần**. Đây là khác biệt sống còn giữa "train được" và "văng CUDA Out Of Memory".
  - **Thời gian chạy thực tế:** Chính comment trong notebook đã ghi lại kết quả thực nghiệm — để `1280` thì một lượt train ngốn tới **15 tiếng**, vượt xa hạn mức 30 giờ GPU/tuần mà Kaggle cấp cho cả 3 mô hình.
  - **Vì sao không sợ mất vật thể nhỏ?** Nhiệm vụ "soi biển báo li ti" đã được giao cho **YOLOv8s-P2** (`imgsz=1280` + nhánh P2 Stride 4), còn khi lên Web App thì cả 3 mô hình đều được bọc trong **SAHI** để zoom cận cảnh. RT-DETR giữ đúng vai trò của nó: chứng minh sức mạnh **Nhận thức Ngữ cảnh Toàn cục**.

```python
    batch=4,            # Tăng batch vì ảnh nhỏ lại
```
- **Ý nghĩa Kỹ thuật (Vì sao bỏ được Gradient Accumulation)**: Kế hoạch cũ ở mức `1280` buộc phải ép `batch=2` rồi bù lại bằng `accumulate=4` để tạo ra "batch ảo" bằng 8 (cộng dồn đạo hàm 4 chu kỳ mới cập nhật trọng số 1 lần).
  - Sau khi hạ xuống `640`, bộ nhớ Attention giảm 16 lần nên GPU thừa sức nạp thẳng nhiều ảnh thật cùng lúc. Notebook Colab dùng `batch=4`, còn bản Kaggle (`train_rtdetr_Kaggle.ipynb`) đẩy lên `batch=8` nhờ chạy song song 2 GPU T4 (`device=[0, 1]`).
  - **Kết quả**: Batch thật luôn tốt hơn batch ảo — thống kê BatchNorm chuẩn hơn, code cũng gọn hơn vì bỏ được một tham số dễ gây lỗi API.

```python
    optimizer='AdamW',
    cos_lr=True,
    project='/content/drive/MyDrive/DoAn_NhanDienBienBao/zalo_traffic',
    name='rtdetr_colab_highres',
```
- **Ý nghĩa**: 
  - `AdamW`: Khác với SGD dùng cho Faster R-CNN, mô hình Transformer cực kỳ nhạy cảm với việc tinh chỉnh Learning Rate và hiện tượng bùng nổ trọng số. Thuật toán `AdamW` (Adam với Weight Decay) là tiêu chuẩn vàng của ngành công nghiệp dành cho Transformer.
  - `cos_lr=True`: Hạ nhiệt độ Learning Rate theo đường cong hình sin (Cosine Annealing), tránh việc mô hình đi lệch khỏi điểm cực tiểu toàn cục ở những epoch cuối.
  - Kết quả cuối cùng được xuất thẳng sang Google Drive thông qua biến `save_dir`.

---

## Cell 7: Thuật toán SAHI (Hỗ trợ Web App)

```python
from sahi import AutoDetectionModel
...
    detection_model = AutoDetectionModel.from_pretrained(
        model_type='yolov8',  # SAHI gọi RT-DETR qua type yolov8
        model_path='/content/drive/MyDrive/DoAn_NhanDienBienBao/zalo_traffic/rtdetr_colab_highres/weights/best.pt',
...
```
- **Ý nghĩa MLOps**: Chuẩn bị sẵn sàng cho quá trình Deploy (Triển khai lên Web). Trong ứng dụng thực tế (Camera giao thông), ảnh có thể là 4K. Kỹ thuật **SAHI (Slicing Aided Hyper Inference)** giúp băm tấm ảnh 4K ra làm nhiều mảnh nhỏ (ví dụ 4 mảnh 1080p), cho RT-DETR dự đoán trên từng mảnh, rồi tự động nối (Merge) các Bounding Box lại với nhau. Điều thú vị là thư viện SAHI đóng gói API chung cho RT-DETR qua khai báo `model_type='yolov8'`.

---

## Tổng kết: Bảng đối chiếu Code ↔ Spec

| Thông số kỹ thuật | Code trong notebook | `models_specs.md` | Trạng thái |
|---|---|---|---|
| Phân loại kiến trúc | `RTDETR('rtdetr-l.pt')` | Transformer (Self-Attention) Large | ✅ Khớp |
| Chia Dữ liệu | `random.shuffle` (80% Train / 20% Val) | 80% Train / 20% Val | ✅ Khớp |
| Kích thước ảnh (Resolution)| `imgsz=640` | `640 x 640` (hạ từ 1280 để chống OOM) | ✅ Khớp |
| Tối ưu hóa bộ nhớ (VRAM) | `batch=4` (Colab) / `batch=8` (Kaggle) | Batch thật, bỏ Gradient Accumulation | ✅ Khớp |
| Thuật toán Tối ưu | `optimizer='AdamW'` | AdamW (Chuẩn công nghiệp cho Transformer)| ✅ Khớp |
| Hạ nhiệt độ LR | `cos_lr=True` | Cosine Annealing | ✅ Khớp |
| Triển khai (Deploy) | Gọi thư viện `sahi` | Slicing Aided Hyper Inference | ✅ Khớp |
