# KẾ HOẠCH TRIỂN KHAI CODE TRAINING VÀ WEB APP (BẢN THIẾT KẾ CHUẨN)

Tài liệu này là bản thiết kế code (Code Blueprint) tổng quát và chi tiết nhất để áp dụng chính xác **10 chiến thuật đã chốt từ Phần IV của file EDA Tổng kết** vào quá trình huấn luyện 3 mô hình và xây dựng ứng dụng Web App Inference.

---

## TỔNG QUAN PHÂN BỔ KỸ THUẬT

Dưới đây là ma trận phân bổ các kỹ thuật tinh chỉnh cho từng giai đoạn và từng mô hình cụ thể, giúp bạn có cái nhìn toàn cảnh trước khi code:

### 🌐 1. Dành cho phần Web App (Giai đoạn Suy luận / Inference)
Phần Web App (Dự đoán 1 ảnh tĩnh tải lên) được ưu tiên tối đa hóa độ chính xác (mAP) và không bị giới hạn bởi tốc độ Real-time. Các kỹ thuật "nặng đô" này sẽ được code thẳng trên Server Web:
- **SAHI (Slicing Aided Hyper Inference)**: Cắt ảnh trượt đè lấp để soi bằng được các biển báo siêu li ti ở xa. Áp dụng chung cho cả 3 model khi load weights dự đoán.
- **TTA (Test-Time Augmentation)**: Bật tính năng lật/phóng to ảnh lúc test để lấy kết quả trung bình, vá các góc nhìn mù.
- **Tối ưu NMS & Soft-NMS**: Hạ `max_det=50`, nới lỏng `IoU=0.6` hoặc dùng `Soft-NMS` để giữ lại các biển báo cắm chùm, đứng sát mép nhau mà không bị xóa nhầm.

### 🤖 2. Phân bổ cho mô hình YOLOv8
- **Cấu trúc mạng**: Bật nhánh P2 Layer (`yolov8s-p2.yaml`).
- **Hàm Loss**: Bật Focal Loss, cấu hình tăng mạnh `cls_gain` cao hơn `box_gain`.
- **Augmentation**: Kích hoạt Mosaic cường độ cao, Random Shift (thông qua `translate` để trị Center Bias).
- **Siêu tham số Training**: `imgsz=1280`, `max_det=50`, dùng Optimizer `AdamW` + `Cosine Annealing LR`.

### 🤖 3. Phân bổ cho mô hình Faster R-CNN
- **Cấu trúc mạng**: Ghi đè Anchor Box mặc định của mạng RPN bằng bộ **K-Means Anchor 1:1** sinh ra từ EDA. Lưu ý cấu hình **num_classes = 8** (7 class biển báo + 1 class Background bắt buộc của PyTorch).
- **Hàm Loss**: *Lưu ý:* Thư viện `torchvision` không hỗ trợ sẵn Focal Loss cho Faster R-CNN. Việc sửa mã nguồn (source code) C++ của PyTorch để ép vào là quá rủi ro cho đồ án sinh viên. Do đó ta đành chấp nhận dùng Cross-Entropy gốc và bù đắp bằng các kỹ thuật khác.
- **Augmentation**: Sử dụng thư viện `Albumentations` để áp dụng **BBox-Safe Crop** (với `min_visibility=0.5`).
- **Siêu tham số Training**: Code cũ sử dụng Optimizer `SGD (momentum=0.9)` hoạt động cực kỳ ổn định với kiến trúc ResNet của PyTorch. Ta sẽ giữ nguyên `SGD` thay vì ép dùng `AdamW` như 2 model kia để tránh hỏng model.

### 🤖 4. Phân bổ cho mô hình RT-DETR
- **Xử lý tài nguyên (OOM)**: RT-DETR ngốn VRAM khủng khiếp. Phải áp dụng thuật toán **Gradient Accumulation** (Ép batch size thật nhỏ và tích lũy đạo hàm) để model học được ảnh `imgsz=1280` mà không làm chết GPU máy chủ.

---

## HƯỚNG DẪN CODE CHI TIẾT (MÃ NGUỒN MẪU)

### PHẦN I: CODE CHUẨN DÀNH CHO WEB APP (INFERENCE)
Khối code này được dùng ở phía Backend của ứng dụng Web (ví dụ: Flask/FastAPI) để xử lý ảnh do người dùng upload.

```python
# Cài đặt thư viện: !pip install sahi
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

# 1. Load model với các thông số tối ưu [TỪ E1, E2]
detection_model = AutoDetectionModel.from_pretrained(
    model_type='yolov8', # Sửa thành 'torchvision' nếu dùng Faster R-CNN
    model_path='weights/best.pt',
    confidence_threshold=0.25,
    device="cuda:0", # Dùng GPU để tăng tốc (hoặc cpu)
)

# 2. Sử dụng SAHI kết hợp Soft-NMS [TỪ E3, M4.2]
result = get_sliced_prediction(
    "uploaded_image.jpg",
    detection_model,
    slice_height=512,
    slice_width=512,
    overlap_height_ratio=0.2, # Đè lấp 20% để không làm đứt đôi biển báo
    overlap_width_ratio=0.2,
    postprocess_type="SOFTNMS", # Kích hoạt Soft-NMS bảo vệ biển báo đứng sát nhau
    postprocess_match_metric="IOU",
    postprocess_match_threshold=0.6 # Nới lỏng IoU
)

# 3. Xuất kết quả trả về Web
result.export_visuals(export_dir="static/output/")
```

### PHẦN II: CODE HUẤN LUYỆN YOLOv8
**File áp dụng:** `notebooks/train_yolov8.ipynb`

```python
import yaml
from ultralytics import YOLO

# 1. Tạo và load kiến trúc P2 Layer [TỪ E3]
# Có thể dùng trực tiếp file yolov8s-p2.yaml từ ultralytics
model = YOLO('yolov8s-p2.yaml') 
model.load('yolov8s.pt') # Load pretrain để học nhanh

# 2. Thiết lập Hyperparameters khổng lồ đúc kết từ EDA
results = model.train(
    data='dataset.yaml',
    epochs=50,
    imgsz=1280,         # High-res bảo toàn pixel vật thể nhỏ
    batch=8,
    max_det=50,         # Tối ưu luồng NMS, tăng tốc luồng xử lý
    iou=0.6,            # Giữ biển báo đứng cạnh nhau
    optimizer='AdamW',  # Lịch trình học thuật
    cos_lr=True,        # Hạ nhiệt độ Learning Rate bằng Cosine Annealing
    
    # --- Kỹ thuật Hàm Loss ---
    fl_gamma=2.0,       # [SỬA LỖI API] Kích hoạt Focal Loss bằng cách đặt gamma > 0
    cls=2.0,            # Tăng cls_gain, ép soi kỹ hình vẽ bên trong biển báo
    box=1.0,
    
    # --- Augmentation & Khắc phục Center Bias ---
    mosaic=1.0,         # Trộn ảnh liên tục
    degrees=10.0,       # Xoay nhẹ
    translate=0.2,      # Random Shift văng biển báo ra mép ảnh
    
    project='zalo_traffic',
    name='yolov8_optimized'
)
```

### PHẦN III: CODE HUẤN LUYỆN FASTER R-CNN
**File áp dụng:** `notebooks/train_faster_rcnn.ipynb`

```python
import torch
import torchvision
from torchvision.models.detection.anchor_utils import AnchorGenerator
import albumentations as A
from albumentations.pytorch import ToTensorV2

# 1. BBox-Safe Augmentation [TỪ E4]
def get_transform():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        # min_visibility=0.5: Bộ lọc thông minh tự động hủy nhát cắt nếu xóa mất quá 50% biển báo
        A.RandomSizedBBoxSafeCrop(width=1280, height=1280, erosion_rate=0.0, p=0.3),
        A.Normalize(),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'], min_visibility=0.5))

# 2. Thay thế Anchor bằng K-Means [TỪ E6]
def get_model(num_classes):
    # Load model gốc
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    
    # 5 kích thước chuẩn của Zalo AI (Giá trị này lấy từ output chạy thực tế của K-Means)
    anchor_sizes = ((15, 25, 45, 70, 120), ) 
    aspect_ratios = ((1.0,),) # Ép dùng hoàn toàn hình vuông 1:1
    
    # Ghi đè vào mạng RPN
    anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)
    model.rpn.anchor_generator = anchor_generator
    return model

# 3. Khởi tạo mô hình và Optimizer (Kế thừa ưu điểm của code cũ)
num_classes = 8  # [QUAN TRỌNG] 7 class biển báo + 1 class Background bắt buộc của PyTorch
model = get_model(num_classes)

params = [p for p in model.parameters() if p.requires_grad]
# Giữ nguyên SGD (momentum=0.9) thay vì dùng AdamW để đảm bảo ResNet hội tụ tốt nhất
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
```

### PHẦN IV: CODE HUẤN LUYỆN RT-DETR
**File áp dụng:** `notebooks/train_rtdetr.ipynb`

```python
from ultralytics import RTDETR

model = RTDETR('rtdetr-l.pt')

# Chống tràn RAM với Gradient Accumulation
results = model.train(
    data='dataset.yaml',
    epochs=50,
    imgsz=1280,         # Vẫn bắt buộc giữ độ phân giải 1280
    batch=2,            # Ép batch=2 để GPU Google Colab không bị chết (OOM)
    accumulate=4,       # Tích lũy 4 step đạo hàm -> Batch ảo = 2 x 4 = 8
    optimizer='AdamW',
    cos_lr=True,
    project='zalo_traffic',
    name='rtdetr_optimized'
)
```



# GIẢI THÍCH CHIẾN THUẬT VÀ HYPERPARAMETERS 

Tài liệu này đóng vai trò là "Luận cứ bảo vệ" (Thesis Defense Arguments). Nếu hội đồng giáo viên thắc mắc tại sao bạn lại sử dụng tham số A thay vì tham số B, bạn chỉ cần mở tài liệu này ra. Mọi con số trong Code Training đều được biện luận chặt chẽ từ kết quả của quá trình Phân tích Dữ liệu (EDA).


---

## 0. Biện luận Lựa chọn Kiến trúc Mô hình (Model Selection)

Đề bài (Document: `Object-Detection-Application.pdf`) yêu cầu sinh viên phải huấn luyện và so sánh ít nhất 3 kiến trúc mô hình đại diện cho 3 trường phái khác nhau. Dưới đây là lý do chúng ta chọn **Faster R-CNN, YOLOv8, và RT-DETR** mà từ chối các mô hình được gợi ý khác (như SSD, RetinaNet, DETR nguyên bản).

### 0.1 Nhóm 1: Traditional CNN-based (Lựa chọn: Faster R-CNN)
*   *Các ứng viên bị loại (Từ gợi ý đề bài):* 
    *   **SSD (Single Shot MultiBox Detector):** Kiến trúc one-stage (một giai đoạn) đời đầu. Ưu điểm là nhanh, nhưng nhược điểm chí mạng là **nhận diện vật thể nhỏ (Small Object) cực kỳ kém** do trích xuất đặc trưng ở các feature map độ phân giải thấp. Bài toán biển báo Zalo AI có diện tích vật thể $<1\%$, dùng SSD chắc chắn sẽ có chỉ số Recall chạm đáy.
    *   **RetinaNet:** Tốt hơn SSD nhờ có Focal Loss (chuyên trị dữ liệu mất cân bằng). Tuy nhiên tốc độ huấn luyện chậm và kiến trúc FPN chưa tối ưu mạnh mẽ như các model đời mới.
*   *Lý do chốt chọn Faster R-CNN:* Dù chạy chậm (Two-stage network), nhưng cơ chế chốt hạ bằng hộp neo **RPN (Region Proposal Network)** của nó mang lại độ chính xác (mAP) cao nhất trong họ CNN cổ điển. Hơn nữa, việc chọn Faster R-CNN tạo cơ hội cực tốt để trình diễn kỹ năng **K-Means Clustering** (tự gom cụm Anchor box từ file EDA), giúp ghi điểm kỹ thuật tuyệt đối với Hội đồng. Nó đóng vai trò làm "Đường cơ sở" (Baseline) hoàn hảo để so sánh.

### 0.2 Nhóm 2: YOLO-based (Lựa chọn: YOLOv8)
*   *Lý do chốt chọn YOLOv8:* 
    *   Đề bài yêu cầu xây dựng Web-app (Requirement 2). YOLOv8 cân bằng hoàn hảo nhất giữa tốc độ (Real-time) và độ chính xác, vô cùng nhẹ để nhúng vào Web.
    *   Đây là kiến trúc **Anchor-free** tiên tiến, giải quyết được rườm rà của hộp neo.
    *   Quan trọng nhất: Cấu trúc của YOLOv8 cho phép ta can thiệp sâu vào kiến trúc mạng (sửa file YAML để mở thêm nhánh **P2 Layer**). Đây là đòn bẩy kỹ thuật trị dứt điểm bài toán biển báo siêu nhỏ siêu xa mà các bản YOLO cũ (v5, v7) làm rất yếu.

### 0.3 Nhóm 3: Transformer-based (Lựa chọn: RT-DETR)
*   *Các ứng viên bị loại (Từ gợi ý đề bài):* 
    *   **DETR nguyên bản (Original DETR):** Điểm yếu chí mạng là hội tụ cực kỳ chậm (cần tới 500 epochs) và khả năng tìm vật nhỏ cực kỳ tệ do giới hạn của cơ chế Attention tiêu chuẩn.
    *   **Deformable DETR / ViTDet:** Khắc phục được vật thể nhỏ và tăng tốc hội tụ nhờ Sparse Attention. Tuy nhiên, việc thiết lập môi trường build C++ (Multi-Scale Deformable Attention) trên Window/Colab rất phức tạp, dễ gây lỗi hệ thống (Dependency Hell) khi sinh viên làm đồ án.
*   *Lý do chốt chọn RT-DETR (Real-Time DETR):* Đây là mô hình Vision Transformer sinh ra để "hủy diệt" YOLO. Nó đập tan định kiến "Transformer là phải chạy chậm", đạt tốc độ Real-time trong khi vẫn giữ nguyên sức mạnh **"Nhận thức ngữ cảnh toàn cục" (Global Context)** của cơ chế Self-Attention. Dùng RT-DETR vừa giải quyết được hiện tượng "Đồng xuất hiện" (E5 trong EDA), vừa tạo ra ấn tượng về sự cập nhật công nghệ mới nhất (State-of-the-Art) cho đồ án.

---
