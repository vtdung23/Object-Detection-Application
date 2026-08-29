# THÔNG SỐ KỸ THUẬT MÔ HÌNH (MODEL DATASHEETS)

Tài liệu này đóng vai trò như một **Datasheet tiêu chuẩn công nghiệp**, cung cấp thông số kỹ thuật chi tiết nhất (cấp độ mạng nơ-ron và hàm toán học) cho 3 mô hình được lựa chọn. Những thông số này là vũ khí đắc lực để báo cáo trước Hội đồng khoa học nhằm chứng minh độ hiểu biết sâu sắc về kiến trúc mạng.

---

## 1. YOLOv8s-P2 (Custom Architecture)
**Phân loại:** One-stage Anchor-free Detector
**Mục tiêu thiết kế:** Đạt tốc độ mượt mà nhất cho Web App nhưng vẫn bắt được vật thể cực nhỏ (nhờ P2 Layer).

### 1.1 Thông số Mạng (Architecture)
- **Backbone:** Modified CSPDarknet53. Sử dụng cấu trúc `C2f` (Cross Stage Partial Bottleneck với 2 chập) thay cho `C3` cũ, giúp dòng gradient chảy mượt hơn và giảm thiểu suy hao đặc trưng.
- **Neck:** PANet (Path Aggregation Network). Nối các đặc trưng từ dưới lên trên (Bottom-up) và từ trên xuống (Top-down).
- **Head (Custom):** 
  - Mở rộng nhánh **P2 Layer (Stride 4)**. Ở ảnh `1280x1280`, nhánh P2 xuất ra Feature Map kích thước khổng lồ `320x320` pixel.
  - Hỗ trợ phát hiện ở 4 cấp độ (P2, P3, P4, P5) thay vì 3 cấp độ mặc định.
- **Số lượng tham số (Parameters):** ~11.5 Triệu. (Nhỉnh hơn bản gốc 11.1M do thêm nhánh P2).
- **Khối lượng tính toán (GFLOPs):** ~30 GFLOPs.

### 1.2 Thiết lập Huấn luyện (Training Config)
- **Độ phân giải đầu vào (Resolution):** `1280 x 1280` (High-resolution).
- **Thuật toán Tối ưu (Optimizer):** `AdamW` (Tự động thích ứng Learning Rate và phạt Weight Decay chuẩn xác hơn Adam gốc).
- **Lịch trình LR (Learning Rate Scheduler):** `Cosine Annealing`. Khởi đầu (Warmup) ở mức rất nhỏ, sau đó vọt lên và giảm dần theo đường cong hình sin lượn sóng.
- **Kích thước Lô (Batch Size):** 8.

### 1.3 Cơ chế Hàm độ lỗi (Loss Functions)
Tổng Loss = $\lambda_1 L_{cls} + \lambda_2 L_{box} + \lambda_3 L_{dfl}$
- **Classification Loss ($L_{cls}$):** Dùng **BCE (Binary Cross-Entropy)** kết hợp với **Focal Loss** (`fl_gamma=2.0`). Trọng số `cls_gain` được ép lên 2.0 để trừng phạt thật nặng các lỗi nhận diện nhầm biển báo có viền đỏ giống nhau.
- **Bounding Box Loss ($L_{box}$):** Sử dụng **CIoU Loss** (Complete IoU). Không chỉ xét diện tích đè lấp, mà còn đo khoảng cách giữa 2 tâm (Center distance) và tỷ lệ khung hình (Aspect Ratio).
- **Distribution Focal Loss ($L_{dfl}$):** Tối ưu hóa xác suất ranh giới mờ của các hộp (Fuzzy boundaries).

### 1.4 Yêu cầu Phần cứng & Hiệu năng
- **Tài nguyên Training:** Yêu cầu GPU tối thiểu 12GB VRAM (như T4, RTX 3060). Thời gian train ~ 3-4 phút / Epoch.
- **Tốc độ Inference:** ~80-100 FPS trên GPU, và ~15-20 FPS nếu chạy thuần CPU (Hoàn toàn vượt chuẩn cho Web App).

---

## 2. Faster R-CNN (Baseline Two-stage)
**Phân loại:** Two-stage Anchor-based Detector
**Mục tiêu thiết kế:** Đóng vai trò là đường cơ sở (Baseline) mang tính học thuật cao nhất, chuẩn mực của Object Detection truyền thống.

### 2.1 Thông số Mạng (Architecture)
- **Backbone:** ResNet-50 (Residual Network với 50 lớp). Sử dụng các khối Skip Connection để chống hiện tượng tiêu biến đạo hàm (Vanishing Gradient) khi mạng quá sâu.
- **Neck:** FPN (Feature Pyramid Network).
- **RPN (Region Proposal Network):**
  - Mạng lưới sinh hộp neo (Anchor Generator). 
  - **Sự khác biệt cốt lõi:** Thay vì dùng hộp neo mặc định, mô hình này được nhúng **5 cụm Anchor Box 1:1** sinh ra trực tiếp từ thuật toán K-Means trên tập dữ liệu Zalo AI (15x15, 25x25, 45x45, 70x70, 120x120).
- **RoI Heads:** Dùng RoIAlign để trích xuất đặc trưng chính xác tới cấp độ số thập phân, khắc phục lỗi lệch pixel của RoIPool cũ.
- **Số lượng tham số (Parameters):** ~41 Triệu.
- **Khối lượng tính toán (GFLOPs):** ~130 GFLOPs.

### 2.2 Thiết lập Huấn luyện (Training Config)
- **Độ phân giải đầu vào:** Tự động scale sao cho cạnh nhỏ nhất là 800px.
- **Thuật toán Tối ưu (Optimizer):** `SGD` (Stochastic Gradient Descent). Với ResNet, SGD (kèm Momentum=0.9) luôn mang lại sự hội tụ ổn định và sâu hơn so với các thuật toán họ Adam.
- **Batch Size:** 4 (Do kiến trúc Two-stage rất tốn VRAM).

### 2.3 Cơ chế Hàm độ lỗi (Loss Functions)
- **RPN Loss:** Gồm 2 hàm: Objectness Loss (BCE) để phân biệt có vật thể hay nền, và RPN Box Loss (Smooth L1).
- **RoI Loss:** Gồm 2 hàm: Classification Loss (**Cross-Entropy**, do giới hạn mã nguồn không dùng được Focal Loss) và Box Regression Loss (Smooth L1).

### 2.4 Yêu cầu Phần cứng & Hiệu năng
- **Tài nguyên Training:** Yêu cầu VRAM >= 12GB. Chạy khá lâu do phải huấn luyện cả 2 mạng (RPN và Fast R-CNN độc lập).
- **Tốc độ Inference:** ~10 FPS trên GPU. (Khá chậm, chỉ phù hợp xử lý ảnh tĩnh).

---

## 3. RT-DETR-L (State-of-the-Art Transformer)
**Phân loại:** End-to-End Transformer-based Detector
**Mục tiêu thiết kế:** "Cỗ máy hủy diệt" dùng để phô diễn sức mạnh công nghệ mới nhất. Giải quyết bài toán ngữ cảnh toàn cục (Biển cấm ngược chiều thường đi với biển cấm quẹo).

### 3.1 Thông số Mạng (Architecture)
- **Backbone:** HGNetv2 (Hierarchical Graph Network). Cực kỳ mạnh mẽ trong việc trích xuất đặc trưng cấp thấp.
- **Neck / Encoder:** Hybrid Encoder thay thế cho Transformer Encoder tiêu chuẩn. Giảm bớt số lớp Attention để tăng tốc, kết hợp với các khối chập (Conv) cục bộ.
- **Decoder:** Transformer Decoder.
  - Sử dụng cơ chế Attention để mô hình tự động "nhìn" vào các mối liên kết toàn cục của bức ảnh.
  - Bỏ hoàn toàn thuật toán NMS. Mô hình tự động xuất ra số lượng hộp cố định và dùng thuật toán **Bipartite Matching** (Hungarian Algorithm) để khớp 1-1 với vật thể thật.
- **Số lượng tham số (Parameters):** ~32 Triệu (Bản L).
- **Khối lượng tính toán (GFLOPs):** ~114 GFLOPs.

### 3.2 Thiết lập Huấn luyện (Training Config)
- **Độ phân giải đầu vào:** `1280 x 1280`.
- **Thủ thuật Chống OOM (Out-Of-Memory):** **Gradient Accumulation**. Ép `batch_size = 2` nhưng cài `accumulate = 4`. Mô hình sẽ cộng dồn đạo hàm qua 4 chu kỳ liên tiếp mới cập nhật trọng số 1 lần, tạo ra hiệu ứng batch ảo là 8.
- **Thuật toán Tối ưu (Optimizer):** `AdamW` + `Cosine Annealing`.

### 3.3 Cơ chế Hàm độ lỗi (Loss Functions)
- Không dùng kiến trúc Anchor nên không có khái niệm IoU lúc Training đơn thuần.
- Sử dụng **Hungarian Loss**: Là sự kết hợp của Classification Loss (Focal Loss) và Box Loss (GIoU + L1 Loss). Thuật toán Hungarian tính toán ma trận chi phí (Cost Matrix) để khớp dự đoán với nhãn thực tế sao cho chi phí là rẻ nhất.

### 3.4 Yêu cầu Phần cứng & Hiệu năng
- **Tài nguyên Training:** Cực kỳ "ăn" VRAM do tính chất ma trận vuông khổng lồ của Attention. Nếu không dùng Gradient Accumulation, card 16GB sẽ nổ tung ở ảnh 1280.
- **Tốc độ Inference:** ~60-80 FPS trên GPU. Cực kỳ xuất sắc, có thể dùng thay thế YOLO trong hệ thống thực tế.
