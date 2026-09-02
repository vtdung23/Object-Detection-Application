# THÔNG SỐ KỸ THUẬT MÔ HÌNH (MODEL DATASHEETS)

Tài liệu này đóng vai trò như một **Datasheet tiêu chuẩn công nghiệp**, cung cấp thông số kỹ thuật chi tiết nhất (cấp độ mạng nơ-ron và hàm toán học) cho 3 mô hình được lựa chọn. Những thông số này là vũ khí đắc lực để báo cáo trước Hội đồng khoa học nhằm chứng minh độ hiểu biết sâu sắc về kiến trúc mạng.

---

## 0. Giao thức Dữ liệu & Huấn luyện dùng chung (Phiên bản V3)

Ba mô hình chỉ so sánh được với nhau nếu chúng được huấn luyện và chấm điểm trong **cùng một điều kiện**. Mục này chốt các quy ước bắt buộc mà cả 3 phải tuân theo, được hiện thực hóa trong thư mục `Traffic-Sign-Detection-ZaloAI/notebooks/v3/`.

### 0.1 Phép chia dữ liệu 70-10-20 (một nguồn duy nhất)

Danh sách ảnh được xáo trộn bằng `random_seed = 42`, sau đó cắt lần lượt theo thứ tự **Test → Val → Train**:

| Thứ tự cắt | Tập | Tỷ lệ | Vai trò |
|---|---|---|---|
| 1 | **Hold-out Test** | **20% đầu** | **Bị giấu hoàn toàn.** Chỉ mở ra đúng một lần ở bước đánh giá cuối cùng |
| 2 | Validation | 10% tiếp theo | Chọn `best.pt` và kích hoạt Early Stopping |
| 3 | Train | ~70% còn lại | Dữ liệu duy nhất mà mô hình được nhìn thấy để cập nhật trọng số |

**Vì sao phải cắt tập Test ra trước tiên?** Nếu cắt theo thứ tự Train → Val → Test thì tập Test nằm ở cuối danh sách. Sau này chỉ cần chỉnh tỷ lệ Train một chút (ví dụ đổi từ 70% sang 75%) là toàn bộ tập Test bị xê dịch theo, và mọi kết quả đã chạy trước đó không còn so sánh được với kết quả mới. Cắt Test ra đầu tiên thì nó luôn là 20% đầu của danh sách đã xáo trộn — dù có đổi tỷ lệ Train/Val thế nào, tập Test vẫn đúng y nguyên những bức ảnh cũ.

- Toàn bộ phép chia do một script duy nhất đảm nhiệm: `data_preparation/split_dataset.py`, với `random_seed = 42` cố định. Cả 3 notebook V3 đều gọi chung script này thay vì mỗi notebook tự viết một đoạn chia riêng — chỉ cần lệch một dòng là cả bảng so sánh mất giá trị.
- **Tập Hold-out không được khai báo trong `data.yaml`.** Nếu thêm khóa `test:` trỏ tới thư mục hold-out, Ultralytics sẽ tự động chạy đánh giá trên đó và làm rò rỉ thông tin vào quá trình chọn mô hình.
- Script xuất kèm `split_manifest.json` ghi lại danh sách `image_id` của cả 3 tập, đóng vai trò biên bản để đối chiếu về sau.
- Vì Faster R-CNN đọc COCO JSON chứ không đọc file `.txt` của YOLO, script xuất song song cả hai định dạng (`train_annotations.json`, `val_annotations.json`) từ đúng danh sách ảnh đó.

> **Khác biệt so với phiên bản cũ:** Bản V1/V2 chia 80/20 cho YOLOv8 và RT-DETR (không có tập Test), còn Faster R-CNN chia 90/10 bằng `random_split()` **không set seed** nên không tái lập được. Hệ quả là ba mô hình học trên ba tập dữ liệu khác nhau và không có tập Test thật sự sạch. Bản V3 sinh ra để khắc phục đúng vấn đề này.

### 0.2 Early Stopping: `epochs = 100`, `patience = 15`

Cả 3 mô hình đều đặt trần `epochs = 100` và dừng sớm khi 15 epoch liên tiếp không cải thiện trên tập Validation.

**Lý do không dùng số epoch cố định:** CNN và Transformer hội tụ với tốc độ rất khác nhau. YOLOv8 (mạng tích chập, có sẵn quy nạp cục bộ về không gian) thường bắt nhịp rất nhanh; RT-DETR phải tự học quan hệ không gian từ con số 0 nên chậm hơn hẳn ở giai đoạn đầu. Nếu ép cả ba chạy cứng 50 epoch thì mô hình hội tụ chậm bị cắt ngang lúc chưa chín, còn mô hình hội tụ nhanh thì thừa ra hàng chục epoch chỉ để overfitting. Cho cả ba cùng trần 100 epoch rồi để Early Stopping tự quyết định điểm dừng mới là so sánh công bằng, và bản thân số epoch thực tế mỗi mô hình dùng cũng trở thành một số liệu đáng phân tích.

**Tiêu chí dừng thống nhất là mAP trên tập Validation**, không phải `val_loss`. Ultralytics vốn dừng theo fitness (hàm trọng số của mAP@50 và mAP@50-95), nên Faster R-CNN cũng phải theo mAP@50-95 thì ba mô hình mới cùng một thước đo.

### 0.3 Nhật ký huấn luyện dạng JSON

Sau khi train, mỗi mô hình sinh ra một file `<tên_model>_training_history.json` chứa mảng dữ liệu theo từng epoch với 5 trường: `epoch_id`, `train_loss`, `val_loss`, `mAP_50`, `mAP_50_95`.

- **YOLOv8 và RT-DETR:** Ultralytics đã ghi sẵn `results.csv`, chỉ cần một hàm parse chuyển sang JSON. Hàm này dò cột theo từ khóa `loss` thay vì gõ cứng tên cột, vì hai mô hình dùng bộ loss khác nhau (YOLOv8: `box`/`cls`/`dfl`; RT-DETR: `giou`/`cls`/`l1`).
- **Faster R-CNN:** không có sẵn cơ chế nào, phải nhúng thẳng một list vào vòng lặp `for epoch in range(...)` rồi `json.dump()` sau **mỗi** epoch (không đợi train xong) để không mất nhật ký khi Kaggle ngắt phiên giữa chừng.

File JSON này là nguyên liệu để vẽ **Learning Curve** trong báo cáo — xem `docs/05_testing_evaluation/test_KeHoach_Kaggle.md`.

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
- **Số vòng lặp (Epochs):** `100` với **Early Stopping `patience = 15`** (xem mục 0.2). Ultralytics tự theo dõi fitness trên tập Val, dừng khi 15 epoch liên tiếp không cải thiện và giữ lại `best.pt` của epoch tốt nhất.
- **Tỷ lệ chia dữ liệu:** 70% Train / 10% Val / 20% Hold-out Test ẩn (xem mục 0.1).
- **Tham số dọn rác (NMS):** Hạ `max_det = 50` (Giới hạn tối đa 50 vật thể/ảnh để tối ưu luồng xử lý Web) và `iou = 0.6` (Bảo vệ các biển báo cắm sát nhau).

### 1.3 Cơ chế Tiền xử lý & Augmentation (CPU DataLoader)
- **Mosaic (`1.0`):** Kỹ thuật đập vụn 4 bức ảnh và nén vào 1 lưới (Grid). Vô tình thu nhỏ kích thước thật của vật thể, ép mạng P2 Layer phải học cách nhìn xa. Đồng thời hack dung lượng VRAM (Batch 8 mang bối cảnh của 32 ảnh).
- **Random Shift (`translate=0.2`):** Sử dụng ma trận biến đổi Affine để dịch chuyển toàn bộ tọa độ điểm ảnh ngẫu nhiên 20%. Kỹ thuật này sinh ra để triệt tiêu hội chứng **Center Bias** (Khi thống kê cho thấy 67.4% biển báo Zalo tập trung ở chính giữa bức ảnh).
- **Xoay nhẹ (`degrees=10.0`):** Xoay ảnh ngẫu nhiên trong khoảng ±10° để tăng tính đa dạng dữ liệu, mô phỏng góc nghiêng thực tế của camera Dashcam.

### 1.4 Cơ chế Hàm độ lỗi (Loss Functions)
Tổng Loss = $\lambda_1 L_{cls} + \lambda_2 L_{box} + \lambda_3 L_{dfl}$
- **Classification Loss ($L_{cls}$):** Dùng **BCE (Binary Cross-Entropy)**. Trọng số `cls_gain` được ép lên 2.0 (gấp đôi `box_gain`) để trừng phạt thật nặng các lỗi nhận diện nhầm biển báo có viền đỏ giống nhau — đây chính là biện pháp chống mất cân bằng dữ liệu của mô hình này.

> **Đính chính quan trọng về Focal Loss (cập nhật V3).** Các phiên bản tài liệu trước ghi rằng YOLOv8 bật Focal Loss thông qua tham số `fl_gamma=2.0`. Điều này **không đúng với thực tế cài đặt của thư viện**. `fl_gamma` là di sản từ file siêu tham số của YOLOv5; Ultralytics có giữ lại tên khóa này một thời gian nhưng lớp `v8DetectionLoss` chưa bao giờ đọc tới nó — nhánh phân loại của YOLOv8 dùng thuần `BCEWithLogitsLoss`. Từ các bản Ultralytics gần đây khóa này đã bị xóa hẳn, truyền vào sẽ báo `SyntaxError: 'fl_gamma' is not a valid YOLO argument`.
>
> Hệ quả: kể cả những lần chạy trước đó, Focal Loss cũng **chưa từng được kích hoạt**. Việc gỡ tham số này khỏi mã nguồn V3 do đó **không làm thay đổi kết quả huấn luyện**, chỉ khiến tài liệu phản ánh đúng những gì thực sự chạy. Nhiệm vụ xử lý mất cân bằng dữ liệu được giao hoàn toàn cho `cls_gain=2.0`. Điều này chấp nhận được vì chính phần EDA đã kết luận tỷ lệ mất cân bằng của bộ dữ liệu chỉ khoảng **1:5.5** (Moderate Imbalance), tức mức nhẹ trong Object Detection.
- **Bounding Box Loss ($L_{box}$):** Sử dụng **CIoU Loss** (Complete IoU). Không chỉ xét diện tích đè lấp, mà còn đo khoảng cách giữa 2 tâm (Center distance) và tỷ lệ khung hình (Aspect Ratio).
- **Distribution Focal Loss ($L_{dfl}$):** Tối ưu hóa xác suất ranh giới mờ của các hộp (Fuzzy boundaries).

### 1.5 Yêu cầu Phần cứng & Hiệu năng
- **Tài nguyên Training:** Yêu cầu GPU tối thiểu 12GB VRAM (như T4, RTX 3060). Thời gian train ~ 3-4 phút / Epoch.
- **Tốc độ Inference:** ~80-100 FPS trên GPU, và ~15-20 FPS nếu chạy thuần CPU (Hoàn toàn vượt chuẩn cho Web App).

### 1.6 Bản chất Kỹ thuật: Quy trình vòng lặp học tập của YOLOv8
Để thấu hiểu sự hiệp đồng của tất cả các kỹ thuật trên, ta cần mổ xẻ quy trình một vòng lặp huấn luyện (Training Iteration) theo đúng trình tự thời gian:
0. **Tiền xử lý (DataLoader trên CPU):** Trước khi nạp vào mạng, CPU sẽ bốc ngẫu nhiên 4 bức ảnh Zalo (dạng Panorama ngang 1622x626) khác nhau. Thay vì cắt xén làm xói mòn dữ liệu, DataLoader dùng kỹ thuật **Letterboxing (Đệm viền xám)** để bóp tỷ lệ ảnh lọt thỏm vào khung vuông 1280x1280. Sau đó, nó áp dụng ma trận Affine để dịch chuyển tọa độ (văng vật thể ra rìa), rồi dán đè 4 ảnh lại bằng lưới Mosaic. Kết quả tạo ra một bức ảnh "giả lập" 1280x1280 hoàn toàn mới mẻ, xóa bỏ Center Bias mà vẫn bảo toàn 100% pixel gốc.
1. **Forward Pass (Lan truyền tiến trên GPU):** Bức ảnh 1280x1280 (sau khi Augment) đi qua mạng Backbone (CSPDarknet). Tại nhánh P2 (tầng nông nhất), mạng trích xuất ra một ma trận đặc trưng khổng lồ 320x320 chứa nguyên vẹn điểm ảnh của các biển báo siêu li ti (đã bị thu nhỏ thêm bởi Mosaic). Thông tin này qua cổ chai PANet rồi đẩy ra Head để đưa ra 2 dự đoán: Tọa độ viền hộp và Xác suất phân loại của từng lớp ($p_t$).
2. **Calculate Loss (Đo lường sai số ĐỒNG THỜI):** 
   - Sai số tọa độ được đo bằng thuật toán **CIoU** ($L_{box}$).
   - Sai số phân loại được đo bằng **BCE (Binary Cross-Entropy)** ($L_{cls}$), tính độc lập cho từng lớp theo cơ chế đa nhãn của YOLOv8.
   - Hàm Loss tổng hợp được tính toán ngay lập tức: **$Loss_{total} = 1.0 \cdot L_{box} + 2.0 \cdot L_{cls} + 1.5 \cdot L_{dfl}$**.
3. **Backward Pass (Lan truyền ngược):** PyTorch tính đạo hàm (Gradient) từ $Loss_{total}$ truyền ngược về lại mạng Backbone. Nhờ hệ số nhân `cls_gain=2.0`, dòng thác Gradient dội về từ nhánh Phân loại mạnh gấp đôi so với nhánh Vẽ khung, mang theo mệnh lệnh: *"Ưu tiên sửa lỗi đọc nhầm loại biển báo hơn là lỗi vẽ lệch vài pixel"*.
4. **Weight Update (Cập nhật trọng số):** Thuật toán tối ưu **AdamW** tiếp nhận dòng thác Gradient này. Nó dùng thuật toán tự động thích ứng để tinh chỉnh (update) hàng triệu ma trận tham số trong mạng nơ-ron theo đúng hướng dẫn của Gradient. Kết hợp với chiến thuật phanh **Cosine Annealing LR**, AdamW sẽ hạ cánh lướt êm các trọng số này đáp chính xác xuống đáy của hố Loss (Global Minima) mà không bị học vẹt.
---

## 2. Faster R-CNN (Baseline Two-stage)
**Phân loại:** Two-stage Anchor-based Detector
**Mục tiêu thiết kế:** Đóng vai trò là đường cơ sở (Baseline) mang tính học thuật cao nhất, chuẩn mực của Object Detection truyền thống.

### 2.1 Thông số Mạng (Architecture)
- **Backbone:** ResNet-50 (Residual Network với 50 lớp). Sử dụng các khối Skip Connection để chống hiện tượng tiêu biến đạo hàm (Vanishing Gradient) khi mạng quá sâu.
- **Neck:** FPN (Feature Pyramid Network).
- **RPN (Region Proposal Network):**
  - Mạng lưới sinh hộp neo (Anchor Generator). 
  - **Sự khác biệt cốt lõi (K-Means 1:1 Anchor cho FPN):** Chạy thuật toán K-Means Clustering để phân tích toàn bộ kích thước biển báo trong kho dữ liệu Zalo, gom chúng thành 5 cụm tối ưu nhất: 10px, 24px, 44px, 77px, 133px. Vì biển báo giao thông (Tròn, Tam giác) có tính đối xứng cao, ta ép cứng tỷ lệ khung thành 1:1 (Khung vuông tuyệt đối). Dữ liệu này được định dạng thành một Tuple chứa 5 Tuple con `((10,), (24,), (44,), (77,), (133,))` nhằm phân phối chính xác 5 loại khung mồi này cho 5 tầng của tháp FPN (P2, P3, P4, P5, P6).
- **RoI Heads:** Dùng RoIAlign để trích xuất đặc trưng chính xác tới cấp độ số thập phân, khắc phục lỗi lệch pixel của RoIPool cũ.
- **Số lượng tham số (Parameters):** ~41 Triệu.
- **Khối lượng tính toán (GFLOPs):** ~130 GFLOPs.

### 2.2 Thiết lập Huấn luyện (Training Config)
- **Tỷ lệ chia dữ liệu:** 70% Train / 10% Val / 20% Hold-out Test ẩn (xem mục 0.1), dùng chung `split_dataset.py` với hai mô hình kia.
- **Độ phân giải đầu vào:** Tự động scale sao cho cạnh nhỏ nhất là 800px (`min_size=800`, `max_size=1333` — đây là giá trị mặc định của torchvision, mã nguồn không ghi đè).
- **Thuật toán Tối ưu (Optimizer):** `SGD` (`lr=0.005`, `momentum=0.9`, `weight_decay=0.0005`). Với ResNet, SGD kèm Momentum luôn mang lại sự hội tụ ổn định và sâu hơn so với các thuật toán họ Adam.
- **Lịch trình LR (Learning Rate Scheduler):** `CosineAnnealingLR` (`T_max = 100`), giảm mượt theo đường cong sin suốt 100 epoch.
- **Batch Size:** 4 (Do kiến trúc Two-stage rất tốn VRAM).
- **Số vòng lặp (Epochs):** `100` với **Early Stopping `patience = 15`** tự cài bằng tay (PyTorch thuần không có sẵn cơ chế này như Ultralytics).
- **Tiêu chí chọn Best Model:** `mAP@50-95` trên tập Validation, đo bằng `torchmetrics` sau mỗi epoch.
- **Augmentation tập Validation:** Không áp dụng. Tập Val chỉ đi qua `A.Normalize()`, không có `RandomSizedBBoxSafeCrop`.
- **Phạm vi chạy K-Means Anchor:** Chỉ trên bounding box của **tập Train**.

> **Bốn thay đổi so với phiên bản cũ và lý do:**
>
> 1. **`StepLR` → `CosineAnnealingLR`.** Bản cũ dùng `StepLR(step_size=10, gamma=0.1)` cho 15 epoch nên không có vấn đề gì. Nhưng khi nâng trần lên 100 epoch, Learning Rate sẽ bị chia 10 tổng cộng 10 lần, tức là teo từ `0.005` xuống cỡ $5 \times 10^{-13}$ — mô hình ngừng học hẳn từ khoảng epoch 30 và Early Stopping sẽ cắt ngang một cách vô nghĩa. Đổi sang Cosine cũng đồng bộ luôn với `cos_lr=True` của hai mô hình kia.
> 2. **Chọn Best Model theo `val_loss` → theo `mAP@50-95`.** Ultralytics chọn best theo mAP, nên nếu Faster R-CNN chọn theo loss thì ba mô hình đang được tuyển chọn bằng hai tiêu chí khác nhau. Đổi lại cho thống nhất. Bản V3 vẫn ghi `val_loss` vào nhật ký để vẽ Learning Curve.
> 3. **Tập Validation không còn bị Augment.** Bản cũ truyền chung `get_transform()` cho cả hai `Dataset`, nghĩa là tập Val cũng bị `RandomSizedBBoxSafeCrop` với `p=0.3` — điểm đánh giá vì thế bị nhiễu ngẫu nhiên qua từng epoch. Bản V3 tách `get_transform_train()` và `get_transform_val()` riêng.
> 4. **K-Means chỉ chạy trên tập Train.** Bản cũ chạy K-Means trên toàn bộ annotation của dataset, kể cả phần sau này trở thành tập Test. Kích thước Anchor là một tham số học được từ dữ liệu, nên làm vậy là để thông tin về tập Test rò rỉ vào thiết kế mạng — Data Leakage thật sự, dù mức độ nhẹ.
>
> Vì bộ Anchor giờ phụ thuộc vào tập Train, checkpoint V3 lưu dạng dictionary kèm luôn `anchor_sizes` bên trong thay vì chỉ lưu `state_dict`. Lúc nạp lại để test không phải chạy lại K-Means để đoán anchor nữa.

### 2.3 Cơ chế Hàm độ lỗi (Loss Functions)
- **RPN Loss:** Gồm 2 hàm: Objectness Loss (BCE) để phân biệt có vật thể hay nền, và RPN Box Loss (Smooth L1).
- **RoI Loss:** Gồm 2 hàm: Classification Loss (**Cross-Entropy**, do giới hạn mã nguồn không dùng được Focal Loss) và Box Regression Loss (Smooth L1).

### 2.4 Yêu cầu Phần cứng & Hiệu năng
- **Tài nguyên Training:** Yêu cầu VRAM >= 12GB. Chạy khá lâu do phải huấn luyện cả 2 mạng (RPN và Fast R-CNN độc lập).
- **Tốc độ Inference:** ~10 FPS trên GPU. (Khá chậm, chỉ phù hợp xử lý ảnh tĩnh).

### 2.5 Bản chất Kỹ thuật: Quy trình vòng lặp học tập của Faster R-CNN
Dây chuyền Two-stage của Faster R-CNN trải qua 5 bước cực kỳ cồng kềnh nhưng độ chính xác lại vươn lên đỉnh cao của học thuật:
0. **Tiền xử lý (BBox-Safe Crop trên CPU):** Khác với YOLO dùng Mosaic, mô hình này dùng `Albumentations` cắt ảnh ngẫu nhiên (On-the-fly Random Crop). Kích thước khuôn cắt chốt cứng là `512x512` (Vì ảnh gốc Zalo AI chỉ cao 626px, 512 là giới hạn Power of 2 an toàn nhất). Cụ thể cắt bao nhiêu lần? Mỗi lần nạp 1 bức ảnh vào GPU (ở mỗi Epoch), CPU chỉ tung xúc xắc cắt đúng **1 lần duy nhất** với xác suất `p=0.3` (tức là 30% khả năng bị cắt 512x512, 70% giữ nguyên ảnh gốc). Trải qua 10 Epochs, 1 bức ảnh gốc sẽ sinh ra khoảng 3 phiên bản bị cắt ở 3 tọa độ khác nhau, và 7 lần giữ nguyên. Điều này giúp AI vừa học được chi tiết cục bộ (khi bị cắt phóng to), vừa học được bối cảnh toàn cục (khi giữ nguyên). Nếu nhát cắt chém mất $> 50\%$ diện tích biển báo (`min_visibility=0.5`), hàm `RandomSizedBBoxSafeCrop` lập tức vứt bỏ và tung xúc xắc cắt lại, đảm bảo AI không bao giờ học nhầm "biển báo cụt".
1. **Trích xuất Đặc trưng (ResNet-50 & FPN):** Ảnh nạp vào bị hàng ngàn ma trận Tích chập (Convolutional Kernel 3x3) trượt qua để tìm viền và hình khối, nén lại thành khối ma trận đặc trưng.
   - **Skip Connection:** Khắc phục lỗi Vanishing Gradient (Khi tín hiệu lùi qua 50 lớp nơ-ron, Gradient bị nhân liên tiếp với số $<1$ và tiêu biến về 0). Nhờ đường vòng $F(x)+x$, đạo hàm bằng 1 giữ cho Gradient truyền thẳng về gốc mạng.
   - **FPN (Feature Pyramid):** Lớp nông rất Nét nhưng Ngu ngốc, lớp sâu rất Thông minh nhưng Mờ nhòe. FPN dùng Lateral Connection cộng dồn ma trận, đúc ra Kim tự tháp đặc trưng vừa hiểu ngữ nghĩa vừa sắc nét tọa độ.
2. **Mạng lưới đề xuất (RPN - Giai đoạn 1):** RPN trượt cửa sổ 3x3 trên ma trận đặc trưng. Tại mỗi điểm $(x, y)$, nó chiếu ngược về tâm ảnh gốc và "phóng ra" 5 Khung mồi ảo (Anchor Boxes). Kích thước chốt cứng từ thuật toán **K-Means 1:1** trên Zalo Dataset: $10\times10, 24\times24, 44\times44, 77\times77, 133\times133$. Thông qua phép nhân vô hướng (Dot Product) tạo ra sự cộng hưởng, RPN đánh giá 2 thứ: Điểm Objectness (BCE Loss) và độ lệch Box Deltas, từ đó "ghim" lại tọa độ những vùng CÓ THỂ chứa vật thể (Lọc bỏ 99% rác nền).
3. **Trạm kiểm duyệt (RoI Align & Head - Giai đoạn 2):** Các vùng đề xuất từ RPN được đẩy vào Mạng Head để trả lời "Đó là biển báo gì?". 
   - **RoI Align (Nội suy điểm ảnh lẻ):** Tọa độ RPN đẩy ra thường bị lẻ (VD: $15.6$). RoIPool cũ ép làm tròn thành $15$. Khi phóng ngược lên ảnh thật (nhân tỷ lệ nén 32 lần), sai số $0.6$ bị khuếch đại thành lệch $19.2$ pixel gốc (đủ chém đứt nửa biển báo). RoI Align cấm làm tròn, dùng Nội suy song tuyến tính (Bilinear Interpolation) tính chính xác ma trận của điểm ảo $15.6$. Khối ma trận trích xuất hoàn hảo này được đưa vào hàm Softmax chốt xác suất $p_t$ cho 7 nhóm biển báo.
4. **Tính Loss Kép:** Vì là 2 mạng nơ-ron chạy nối tiếp, model BẮT BUỘC phải tính 2 loại Loss độc lập cùng lúc cho RPN và RoI Head. 
   - **Phân loại:** Dùng **Cross-Entropy** (phóng to điểm phạt bằng Logarit nếu mô hình đoán sai nhưng lại tự tin mù quáng 99%). 
   - **Tọa độ:** Dùng **Smooth L1**. Xa đích thì chạy như hàm L1 thẳng tắp để chống nổ Gradient (Exploding Gradient), về sát đích thì uốn cong thành hàm bậc 2 (Parabol L2) để hạ cánh êm ái chống rung lắc.
5. **Cập nhật Trọng số (SGD Momentum):** Gradient kép dội về thuật toán **SGD (Momentum=0.9)**. Mạng Two-stage rất dễ vỡ nên kén AdamW. Phương trình động lượng $v_{t+1} = \mu v_t + \nabla L$ hoạt động như một cỗ xe lu. Nó giữ lại 90% vận tốc cũ ($\mu = 0.9$), tạo lực quán tính khổng lồ. Nếu gặp rãnh zig-zag, lực đối nghịch tự triệt tiêu. Nếu sụp vào "ổ gà" (Local Minima), đà quá khứ sẽ hất tung xe lu vọt qua miệng hố, lướt êm ái xuống đáy tối ưu vĩ mô.

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
- **Độ phân giải đầu vào:** `640 x 640`. (Đã hạ từ `1280` xuống `640` — xem biện luận đầy đủ tại mục **3.6**. Lý do ngắn gọn: ma trận Self-Attention phình theo $O(N^2)$ nên ảnh 1280 gây tràn VRAM, đồng thời thời gian train/inference vượt quá khuôn khổ đồ án).
- **Kích thước Lô (Batch Size):** `8` trên Kaggle (`train_rtdetr_Kaggle.ipynb`, chạy song song 2 GPU T4 qua `device=[0, 1]`) và `4` trên Colab (`train_rtdetr.ipynb`, 1 GPU).
- **Thủ thuật Chống OOM (Out-Of-Memory):** Sau khi hạ về `640`, bộ nhớ Attention giảm 16 lần nên **không còn phải dùng Gradient Accumulation** nữa. Kế hoạch ban đầu (`batch=2` + `accumulate=4` để tạo batch ảo bằng 8) chỉ là giải pháp chữa cháy cho mức `1280`; nay ta nạp thẳng `batch=8` thật, vừa đơn giản hơn vừa cho đạo hàm ổn định hơn batch ảo.
- **Thuật toán Tối ưu (Optimizer):** `AdamW` + `Cosine Annealing` (`cos_lr=True`).
- **Số vòng lặp (Epochs):** `100` với **Early Stopping `patience = 15`** (xem mục 0.2). Đây là mô hình hưởng lợi nhiều nhất từ cơ chế này: Transformer nổi tiếng hội tụ chậm hơn CNN ở giai đoạn đầu vì phải tự học quan hệ không gian từ con số 0, nên nếu bị cắt cứng ở 50 epoch rất dễ bị đánh giá thấp oan.
- **Tỷ lệ chia dữ liệu:** 70% Train / 10% Val / 20% Hold-out Test ẩn (xem mục 0.1).

### 3.3 Cơ chế Hàm độ lỗi (Loss Functions)
- Bỏ hẳn tư duy Anchor Box (Khung mồi).
- Sử dụng **Hungarian Loss**: Là sự kết hợp của Focal Loss (Phân loại) và L1/GIoU Loss (Tọa độ). 

### 3.4 Bản chất Kỹ thuật: Bức tranh toàn cảnh 5 bước của RT-DETR
Khác hoàn toàn với tư duy "Trượt cửa sổ" của CNN (như YOLO, R-CNN), RT-DETR mang tư duy "Nhìn toàn cục" của Transformer:
0. **Tiền xử lý (Nạp ảnh Panorama 640px):** Trái ngược với Faster R-CNN phải cắt vụn ảnh ra 512x512, RT-DETR nuốt trọn bức ảnh toàn cảnh `640x640` để giữ lại 100% bối cảnh không gian (Ví dụ: Mô hình tự hiểu biển báo cấm rẽ thường đứng chung cột với biển cấm ngược chiều). Ta chấp nhận đánh đổi độ phân giải để giữ được toàn cảnh mà không làm nổ VRAM (mục 3.6).
1. **Trích xuất cục bộ (HGNetv2 Backbone):** Dù là mạng Transformer, lớp đầu tiên của nó BẮT BUỘC phải là mạng Tích chập (CNN) HGNetv2. Lý do: Transformer rất ngu ngốc trong việc nhận diện viền/góc cạnh ở giai đoạn đầu. Mạng CNN sẽ giải quyết phần "chân tay" này và nén ảnh thành ma trận.
2. **Cầu nối đa tầng (CCFM Neck):** Thay vì dùng FPN, RT-DETR dùng mô-đun lai CCFM (Cross-Scale Feature-fusion Module) để trộn lẫn ma trận từ tầng nông và tầng sâu, chuẩn bị "thức ăn" tinh gọn nhất trước khi tống vào lõi Transformer.
3. **Bộ não Transformer (Self-Attention Decoder):** Đây là lõi sức mạnh. Mô hình ném vào không gian đúng 300 "Hạt giống" (Object Queries). Chẳng cần trượt cái cửa sổ nào cả! Mỗi hạt giống phóng tầm mắt bao quát toàn bộ 400 mảnh ghép (ảnh `640x640` nén 32 lần còn lưới `20x20`) bằng cơ chế toán học **Q-K-V (Query-Key-Value)**. Thao tác này hoàn toàn dựa trên Đại số tuyến tính: Nó dùng phép Nhân vô hướng (Dot Product) để đo **Cosine Similarity** (Độ tương đồng) giữa lệnh truy nã ($Q$) và biển quảng cáo của pixel ($K$). Sau khi trúng mục tiêu, nó dùng phép Cộng ma trận (Residual Connection) để "nuốt" trọn khối lượng dữ liệu thật ($V$) vào bản thân hạt giống. Trải qua 6 vòng lặp Decoder, 300 hạt giống này tự động bù trừ Offset (Độ lệch) và nở thành đúng 300 Khung dự đoán.
4. **Khớp nối 1-1 & Tính Loss (Hungarian Algorithm):** Đây là cuộc cách mạng chấm dứt kỷ nguyên của NMS! R-CNN hay YOLO phọt ra 10,000 khung rồi phải dùng NMS xóa bớt. RT-DETR chỉ xuất đúng 300 khung. Nó dùng thuật toán **Bipartite Matching** (Kuhn-Munkres) trong thời gian đa thức $O(N^3)$ để lập Ma trận chi phí (Dựa trên $\mathcal{L}_{\text{L1}}$, $\mathcal{L}_{\text{GIoU}}$ và $\text{Focal Loss}$). Toán học giải bài toán Tối ưu Tổ hợp sao cho 5 cái biển báo thật được gán cho đúng 5 khung dự đoán với chi phí RẺ NHẤT (1-kèm-1). Sự thanh lịch tuyệt đối nằm ở chỗ: 5 khung khớp nhất được lôi ra tính Loss để bay về đích. 295 khung rớt đài còn lại bị ép gán nhãn "$\varnothing$" (Background) và lập tức chịu sự trừng phạt tàn khốc của **Focal Loss**, sinh ra dòng Gradient âm khổng lồ đè bẹp trọng số của chúng về 0. Toàn bộ rác nền tự động bị triệt tiêu bằng Toán học thuần túy mà không cần NMS!

### 3.5 Yêu cầu Phần cứng & Hiệu năng
- **Tài nguyên Training:** Cực kỳ "ăn" VRAM do tính chất ma trận vuông khổng lồ của Attention. Nếu không dùng Gradient Accumulation, card 16GB sẽ nổ tung ở ảnh 1280. Đây chính là lý do ta chốt hạ độ phân giải xuống `640` (mục 3.6) thay vì cố đấm ăn xôi ở 1280.
- **Tốc độ Inference:** ~60-80 FPS trên GPU. Cực kỳ xuất sắc, có thể dùng thay thế YOLO trong hệ thống thực tế.

### 3.6 Biện luận: Tại sao hạ Độ phân giải từ `1280` xuống `640`?

Đây là quyết định kỹ thuật quan trọng nhất của mô hình RT-DETR trong đồ án. Ba lý do chính:

**1. Chi phí Self-Attention phình theo bình phương $O(N^2)$ — nguyên nhân trực tiếp gây OOM.**
Khác với YOLOv8 (mạng tích chập, chi phí chỉ tăng tuyến tính theo số pixel), lõi Transformer phải tính ma trận điểm số $QK^T$ giữa **mọi cặp token** với nhau. Số token được lấy từ feature map sau khi nén 32 lần:
- Ở `imgsz=1280`: lưới $40 \times 40 = 1{,}600$ token $\rightarrow$ ma trận Attention $1{,}600 \times 1{,}600 \approx 2{,}56$ triệu ô.
- Ở `imgsz=640`: lưới $20 \times 20 = 400$ token $\rightarrow$ ma trận Attention $400 \times 400 = 160{,}000$ ô.

Chỉ cần chia đôi cạnh ảnh, khối lượng bộ nhớ dành riêng cho Attention **giảm 16 lần**. Vì ma trận này phải được giữ nguyên trong VRAM suốt Forward Pass để phục vụ Backward Pass, đây chính là thủ phạm khiến card 16GB (T4/P100 trên Kaggle) văng lỗi `CUDA Out Of Memory` ở mức 1280 nếu không ép batch size xuống mức tối thiểu.

**2. Đảm bảo thời gian huấn luyện & suy luận nằm trong khuôn khổ đồ án.**
Kaggle chỉ cấp 30 giờ GPU miễn phí mỗi tuần cho cả 3 mô hình. Ở mức 1280, RT-DETR buộc phải chạy batch rất nhỏ nên số bước cập nhật trên mỗi epoch tăng vọt, kéo dài thời gian train tới mức không thể hoàn thành 50 epochs trong hạn mức (ghi chú thực nghiệm trong `train_rtdetr.ipynb`: *"Nếu để 1280 sẽ mất 15 tiếng"*). Hạ xuống 640 cho phép nâng batch lên 8, rút ngắn thời gian train, và quan trọng hơn là giữ được tốc độ suy luận (Inference) đủ nhanh để mô hình xứng đáng với chữ **RT (Real-Time)** trong tên gọi của nó.

**3. Không đánh mất khả năng bắt vật thể nhỏ, vì ta đã có hàng phòng thủ khác.**
Điểm yếu duy nhất của việc hạ độ phân giải là biển báo li ti bị teo nhỏ. Tuy nhiên bài toán này đã được xử lý ở hai tầng khác:
- **Tầng kiến trúc:** Nhiệm vụ "soi vật thể siêu nhỏ" đã được giao cho **YOLOv8s-P2** (`imgsz=1280` + nhánh P2 Stride 4). RT-DETR giữ đúng vai trò của nó là chứng minh sức mạnh **Nhận thức Ngữ cảnh Toàn cục** (giải quyết hiện tượng Đồng xuất hiện E5), chứ không phải đi tranh phần việc của P2 Layer.
- **Tầng suy luận:** Khi lên Web App, cả 3 mô hình đều được bọc trong **SAHI**. SAHI cắt ảnh gốc thành các mảnh nhỏ rồi phóng to trước khi nạp vào mạng, nên biển báo li ti vẫn được zoom cận cảnh bất kể `imgsz` lúc train là bao nhiêu.

> **Tóm lại:** `imgsz=640` cho RT-DETR là lựa chọn bắt buộc để mô hình chạy được trên phần cứng thực tế, và là lựa chọn hợp lý vì nhiệm vụ vật thể nhỏ đã có YOLOv8-P2 + SAHI gánh vác.
