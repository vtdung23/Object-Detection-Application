# KẾ HOẠCH THỰC HIỆN ĐỀ 1: ỨNG DỤNG NHẬN DIỆN BIỂN BÁO GIAO THÔNG (OBJECT DETECTION)

Tài liệu này là bản kế hoạch chi tiết, chuẩn hóa từ A-Z để thực hiện Đề 1, bám sát các yêu cầu từ file PDF gốc của môn học.

**Quyết định cốt lõi:**
1. **Dataset chốt:** Zalo AI Traffic Sign 2020 (Biển báo giao thông đường phố Việt Nam).
2. **Môi trường:** Kaggle Notebooks (để Train AI) và Hugging Face Spaces (để chạy Web App).
3. **Mô hình (Models):** YOLOv8/v11, Faster R-CNN, và DETR.

---

## I. Phân tích Yêu cầu & Đặc thù của Dataset

### 1. Cấu trúc dữ liệu thô (Raw Data) trên Kaggle
Tôi đã nắm cực kỳ rõ cấu trúc của bộ dữ liệu này. Nó được tổ chức theo chuẩn COCO Format chứ không phải chuẩn YOLO:

```text
/kaggle/input/za-traffic-2020/za_traffic_2020/
├── traffic_train/
│   ├── images/                           # Chứa ~4500 tấm ảnh huấn luyện (.png)
│   └── train_traffic_sign_dataset.json   # TẤT CẢ nhãn bị gộp chung vào 1 file này!
└── traffic_public_test/
    └── images/                           # Chứa ảnh test (không có nhãn để tự chấm)
```

**Bên trong file `train_traffic_sign_dataset.json` có gì?**
Nó chứa 3 danh sách (List) khổng lồ:
1. `categories`: Danh sách Tên và ID của 7 loại biển báo giao thông.
2. `images`: Danh sách chiều cao (height), chiều rộng (width) và tên file của 4500 tấm ảnh.
3. `annotations`: Chứa hàng chục ngàn tọa độ biển báo dưới dạng `[x_min, y_min, width, height]` (tọa độ góc trên bên trái và chiều ngang/dọc của hộp). Tọa độ này tính bằng pixel tuyệt đối.

### 2. Thách thức cốt lõi của Dataset
- **Đạt yêu cầu của đề:** Dataset chứa ảnh đường phố Việt Nam, số lượng classes > 5.
- **Thách thức cực đại (Vật thể siêu nhỏ):** Các biển báo trong ảnh rất nhỏ (ví dụ chỉ chiếm 19x18 pixel trong một tấm ảnh độ phân giải 1622x626). Đây là bài toán *Small Object Detection*.
  - **Tác động:** Faster R-CNN (vốn giỏi quét chi tiết) sẽ hoạt động khá ổn. Tuy nhiên, YOLO và DETR nếu nhận ảnh nguyên bản sẽ rớt hiệu suất thê thảm do vật thể bị mất tích trong quá trình trích xuất đặc trưng (Feature Extraction).
  - **Giải pháp:** Bắt buộc phải áp dụng kỹ thuật **Image Tiling** (Cắt 1 ảnh to thành 4-6 ảnh nhỏ) hoặc dùng thư viện **SAHI** (Slicing Aided Hyper Inference).

---

## II. Lộ trình thực hiện chi tiết (Step-by-Step)

Dự án chia làm 2 giai đoạn: **Huấn luyện trên Đám mây (Kaggle)** và **Triển khai Web App (Hugging Face Spaces)**.

### Giai đoạn 1: LÀM VIỆC TRÊN KAGGLE NOTEBOOKS (100% Online)

#### Bước 1: Khởi tạo dữ liệu
- **Dùng trực tiếp Dataset:** Không cần viết code tải dữ liệu. Bạn chỉ cần vào trang dataset trên Kaggle và tạo một Notebook mới (New Notebook). Dữ liệu 6GB đã nằm sẵn trong thư mục `/kaggle/input/` để dùng ngay lập tức.
- **Thống kê (EDA):** Viết code đọc file JSON, vẽ biểu đồ phân bố số lượng của từng loại biển báo (class distribution) để đưa vào báo cáo.

#### Bước 2: Tiền xử lý dữ liệu & Xử lý Vật thể nhỏ (Nhiệm vụ cốt lõi của Đồ án)
- **Tạo Dataset chuẩn format YOLO:** Viết script Python chuyển từ tọa độ JSON gốc `[x_min, y_min, width, height]` sang format YOLO chuẩn hóa `[class_id, x_center, y_center, w, h]`.
- **Chia tập Train/Val:** Tách ngẫu nhiên tập Train thành 80% (train) và 20% (val).
- **Giải quyết bài toán "Vật thể siêu nhỏ":** Áp dụng kỹ thuật Huấn luyện độ phân giải cao (High-Resolution Training) kết hợp với thư viện Slicing Aided Hyper Inference (SAHI) thay cho phương pháp cắt ảnh thủ công. 
- *(Chi tiết về phân tích các chiến thuật chuyên sâu và cách đưa code lên Kaggle ở bước này, vui lòng xem Phần VI ở cuối tài liệu).*

#### Bước 3: Huấn luyện 3 mô hình
- **Model 1: Faster R-CNN (Đại diện Two-stage)**
  - Nạp thẳng file COCO JSON gốc (không cần file .txt).
  - Mô hình này chạy chậm, nặng máy, nhưng độ chính xác dự kiến sẽ cao nhất do khả năng tìm vùng có chứa vật thể nhỏ xuất sắc.
- **Model 2: YOLOv8 hoặc v11 (Đại diện One-stage)**
  - Nạp dữ liệu YOLO `.txt` đã qua xử lý cắt nhỏ ảnh.
  - Tốc độ huấn luyện nhanh, có thể chạy thực tế (real-time) tốt.
- **Model 3: DETR (Đại diện Transformer)**
  - Nạp dữ liệu COCO JSON.
  - Transformer khát dữ liệu, nhưng số lượng 4500+ ảnh của Zalo AI là hoàn toàn đủ để mô hình hội tụ.
  - Cần chỉnh sửa kiến trúc RT-DETR (Real-time DETR) thay vì DETR gốc để huấn luyện nhẹ nhàng hơn.

> ⚠️ **LUẬT SINH TỒN TRÊN KAGGLE:** 
> Mỗi tuần Kaggle cho bạn 30 tiếng dùng GPU miễn phí. Hãy lưu file trọng số (model weights) `best.pt` liên tục vào thư mục `/kaggle/working/` và tải về máy tính để phòng hờ trường hợp hết giờ chạy.

#### Bước 4: Đánh giá và Lập báo cáo
- Chạy 3 mô hình trên tập Test.
- Thu thập các chỉ số: **mAP@50**, **mAP@50-95**, **Inference Time** (Thời gian dự đoán 1 ảnh), và **Model Size** (Dung lượng mô hình).
- Lập bảng so sánh và phân tích sâu sắc sự đánh đổi giữa Tốc độ (của YOLO) và Độ chi tiết (của Faster R-CNN) đối với bài toán nhận diện biển báo kích thước nhỏ.

---

### Giai đoạn 2: LÀM VIỆC TRÊN HUGGING FACE SPACES

Sau khi có kết quả từ Giai đoạn 1, ta chọn ra mô hình có kết quả tổng hợp tốt nhất (Ví dụ YOLOv8) để làm sản phẩm cuối.

#### Bước 5: Triển khai Web App ứng dụng
- **Tải model:** Tải duy nhất file trọng số `best.pt` của mô hình chiến thắng từ Kaggle về máy tính (dung lượng chỉ khoảng vài chục MB).
- **Lập trình giao diện:** Sử dụng thư viện `Streamlit` hoặc `Gradio` viết mã nguồn tạo trang Web và đẩy code lên **Hugging Face Spaces**.
- **Tính năng Web:** 
  1. Người dùng truy cập đường link Public, bấm nút Upload 1 tấm ảnh đường phố.
  2. Load file `best.pt` để dự đoán (Chạy trực tiếp trên server miễn phí của Hugging Face).
  3. Trả về kết quả là tấm ảnh đã được vẽ các Bounding Box xung quanh biển báo và tên biển báo.

---

## III. Các Script / File Code cụ thể cần phải viết

Để hoàn thành Đề 1, danh sách các công việc thực hành (Coding) mà chúng ta cần viết theo thứ tự là:

- [ ] 1. Khởi tạo Kaggle Notebook đính kèm sẵn dataset Zalo AI.
- [ ] 2. Script Data EDA (Phân tích dữ liệu JSON và vẽ biểu đồ).
- [ ] 3. Script Chuyển đổi định dạng: COCO JSON sang YOLO TXT.
- [ ] 4. Script Tiền xử lý: Cắt nhỏ ảnh (Image Tiling) để xử lý biển báo siêu nhỏ.
- [ ] 5. Notebook (File .ipynb) để Train và Đánh giá YOLO.
- [ ] 6. Notebook (File .ipynb) để Train và Đánh giá Faster R-CNN.
- [ ] 7. Notebook (File .ipynb) để Train và Đánh giá DETR.
- [ ] 8. Source code `app.py` (Streamlit) để deploy Web App lên Hugging Face Spaces.

---

## IV. Cấu trúc thư mục chuẩn (Project Structure)

Để quản lý code một cách chuyên nghiệp (tránh tình trạng code vứt lung tung không biết file nào chạy ở đâu), toàn bộ dự án trên máy tính/GitHub của bạn cần được tổ chức theo cấu trúc sau:

```text
Traffic-Sign-Detection-ZaloAI/
│
├── data_preparation/           # Chứa các file kịch bản (Script) xử lý dữ liệu
│   ├── convert_coco_to_yolo.py # Chuyển đổi nhãn COCO gốc sang định dạng YOLO TXT
│   ├── image_tiling.py         # Code cắt nhỏ ảnh (Tiling) cho vật thể nhỏ
│   └── eda_analysis.ipynb      # Phân tích biểu đồ dữ liệu
│
├── notebooks/                  # Các file này SẼ ĐƯỢC UPLOAD LÊN KAGGLE ĐỂ TRAIN
│   ├── train_yolov8.ipynb      # Notebook chứa code tải data, train và test YOLO
│   ├── train_faster_rcnn.ipynb # Notebook chứa code train Faster R-CNN
│   └── train_detr.ipynb        # Notebook chứa code train DETR
│
├── web_app/                    # Các file này CHỈ ĐỂ ĐẨY LÊN HUGGING FACE SPACES
│   ├── app.py                  # Source code chính của giao diện Web
│   ├── utils.py                # Chứa các hàm hỗ trợ vẽ ảnh, xử lý dự đoán
│   ├── requirements.txt        # Danh sách thư viện cần thiết để Hugging Face cài đặt
│   └── weights/                
│       └── best_yolov8.pt      # File trọng số bạn tải từ Kaggle về đặt vào đây
│
├── reports/                    # Thư mục lưu kết quả để viết Báo cáo
│   ├── comparison_table.csv    # Bảng csv so sánh mAP, Speed 3 mô hình
│   └── charts/                 # Lưu các hình ảnh chụp biểu đồ Loss, Confusion Matrix
│
└── ke_hoach_de_1.md            # File kế hoạch bạn đang xem
```

> 💡 **GIẢI ĐÁP QUY TRÌNH KAGGLE & LOCAL:**
> 1. **Kaggle có dùng .ipynb không?** RẤT CHUẨN. Kaggle Notebook bản chất chính là Jupyter Notebook (`.ipynb`). Môi trường nó y hệt Google Colab.
> 2. **Có up toàn bộ cái thư mục to đùng này lên Kaggle không?** KHÔNG. Máy tính của bạn (Local) là nơi lưu giữ toàn bộ cấu trúc này. Khi đến phiên làm việc trên Kaggle, bạn chỉ cần mở trình duyệt, vào Kaggle, bấm nút **Upload Notebook** và chọn các file `.ipynb` nằm trong thư mục `notebooks/` đưa lên đó. 
> 3. Kaggle tự động có sẵn 6GB ảnh ở đường dẫn `/kaggle/input/`. Bạn train xong, file `best.pt` sinh ra ở `/kaggle/working/`, bạn nhấn nút Download tải file đó về, ném vào thư mục `web_app/weights/` là xong!

**KẾT LUẬN:** Đây là một kế hoạch hoàn hảo, vừa đáp ứng mọi yêu cầu của giảng viên, vừa cho thấy chiều sâu nghiên cứu (xử lý được bài toán khó: Small Object Detection). Sẵn sàng bắt tay vào Bước 1 ngay khi bạn muốn!

---

## V. Phân tích chi tiết Task 1: Thống kê Dữ liệu (EDA - Exploratory Data Analysis)

Đây là bước đầu tiên trước khi đưa dữ liệu vào huấn luyện. Đề bài không bắt buộc viết code EDA, nhưng nếu không làm, bạn sẽ mất hoàn toàn cơ sở khoa học để viết báo cáo.

### 1. Đề bài yêu cầu những gì?
- Đề bài ghi rõ: *"Sinh viên phải thu thập đủ dữ liệu để đảm bảo rằng việc huấn luyện và đánh giá mô hình là có ý nghĩa."*
- Giảng viên sẽ chấm điểm phần Biện luận (Justification) cực kỳ gắt gao: Tại sao bạn chọn mô hình này? Dữ liệu của bạn có đặc thù gì? Tại sao mô hình Faster R-CNN lại nhận diện tốt hơn YOLO trên bộ ảnh này? Để trả lời được những câu đó, bạn phải "hiểu" dữ liệu trước bằng số liệu cụ thể.

### 2. Chúng ta cần làm những gì?
Chúng ta sẽ viết code để đọc file `train_traffic_sign_dataset.json` và vẽ ra **2 Biểu đồ cốt lõi** để đưa thẳng vào quyển báo cáo:

- **Biểu đồ 1: Phân bố số lượng của từng loại biển báo (Class Distribution)**
  - **Mục đích:** Đếm xem Biển Cấm có bao nhiêu tấm, Biển Rẽ có bao nhiêu tấm...
  - **Giá trị báo cáo:** Nhìn vào biểu đồ này, ta sẽ biết dữ liệu có bị "mất cân bằng" (Data Imbalance) hay không. (Ví dụ: Biển Cấm có tận 3000 tấm, trong khi Biển Nguy hiểm chỉ có 50 tấm). Phát hiện này giúp ta chèn một câu xuất sắc vào báo cáo: *"Do dữ liệu mất cân bằng, nhóm đã quyết định dùng các hàm Loss function có trọng số để AI không bị thiên vị"*.
  
- **Biểu đồ 2: Phân bố diện tích của Bounding Box (Bbox Size Distribution)**
  - **Mục đích:** Vẽ biểu đồ thể hiện kích thước chiều ngang / chiều dọc của các biển báo.
  - **Giá trị báo cáo:** Bằng biểu đồ này, ta sẽ chứng minh được bằng Toán học rằng: *"Hơn 80% biển báo trong bộ dữ liệu này có kích thước nhỏ hơn 30x30 pixels (chiếm chưa tới 1% bức ảnh gốc)"*. Đây là **luận điểm vàng** để chứng minh bộ dữ liệu này thuộc dạng *Small Object Detection*, từ đó biện luận cho việc vì sao ta phải dùng cắt nhỏ ảnh (Image Tiling) trước khi cho YOLO học!

### 3. Làm bằng cách nào? (Công cụ & Phương pháp)
- **Nơi chạy code:** Khởi tạo một Kaggle Notebook trống, kết nối thẳng với bộ dữ liệu.
- **Thư viện Python sử dụng:** 
  - `json`: Để trích xuất thông tin tọa độ.
  - `pandas`: Dùng để đếm số lượng class nhanh chóng.
  - `matplotlib` & `seaborn`: Dùng để vẽ đồ thị có màu sắc chuyên nghiệp.
- **Output:** Output của bước này không phải là một mô hình AI, mà là **các file ảnh .png của biểu đồ**. Bạn tải các ảnh này về máy tính cá nhân (thư mục `reports/charts/`) và dán vào file Word báo cáo ngay lập tức!

---

## VI. Phân tích chi tiết Task 2: Tiền xử lý & Chiến lược trị Small Object

Đây là giai đoạn quyết định sự thành bại của mô hình, đặc biệt là khi dữ liệu có đặc tính "Vật thể siêu nhỏ" (Small Object Detection). 

### 1. Về chiến thuật Upload Code lên Kaggle
- **Vấn đề:** Không nên upload nguyên cấu trúc cây thư mục (gồm cả script rời) lên Kaggle vì quản lý cực kỳ bất tiện.
- **Cách làm chuẩn:** Mọi thao tác Tiền xử lý (Convert JSON sang YOLO, Augmentation) sẽ được tích hợp luôn vào phần đầu của các file Notebook huấn luyện (ví dụ: `train_yolov8.ipynb`). Nghĩa là, bạn chỉ việc tải duy nhất file Notebook đó lên Kaggle. Khi chạy, nó sẽ tự động đọc data từ `/kaggle/input`, tự động tạo thư mục format YOLO trong `/kaggle/working/`, và tự động Train ngay trong 1 luồng. Cực kỳ gọn gàng!

### 2. Tạo Dataset chuẩn format YOLO
- YOLO không hiểu định dạng COCO JSON mặc định của Zalo AI. Ta phải viết code chuyển từ `[x_min, y_min, width, height]` (tọa độ pixel) sang `[class_id, x_center, y_center, width, height]` (tọa độ chuẩn hóa từ 0-1).
- Quá trình này cũng kèm theo việc tách tập Train gốc thành `train` (80%) và `val` (20%).

### 3. Phân tích Cực sâu 2 Chiến lược trị "Vật thể siêu nhỏ"

Mô hình AI nhận diện vật thể thường bắt buộc phải ép (resize) ảnh về hình vuông nhỏ (như 640x640) để xử lý cho nhanh. Nhưng ảnh gốc của ta là 1622x626, biển báo lại bé xíu (19x18 pixel). Nếu bóp nguyên bức ảnh to đùng đó vào khung 640x640, biển báo sẽ bị bóp vụn thành 1 chấm mờ 7 pixel. AI hoàn toàn bị "Mù".

Để chữa bệnh mù này, thay vì chỉ dùng phương pháp cắt vụn ảnh cơ bản, ta có 2 chiến lược tuyệt đỉnh sau:

#### Chiến lược 1: Huấn luyện Độ phân giải cao (High-Resolution Training)
- **Cơ chế hoạt động:** Thay vì ép AI nhìn ở size mặc định `640x640`, ta bắt YOLO phải nới rộng "cặp mắt" của nó ra, học ở size `1024x1024` hoặc `1280x1280`.
- **Tại sao lại hiệu quả?** Ở độ phân giải 1280, tấm ảnh sẽ ít bị bóp méo (resize) hơn, do đó cái biển báo 19px ban đầu vẫn giữ được hình thù rõ nét thay vì biến thành cái chấm mờ nhạt nhòa. Mô hình sẽ dễ dàng trích xuất được đặc trưng của biển báo (hình tròn, tam giác, màu đỏ, v.v.).
- **Rủi ro và Giải pháp:** Size ảnh càng to, GPU càng phải nạp vào RAM nhiều điểm ảnh (pixel). Nếu bộ nhớ VRAM của GPU không đủ, máy sẽ bị sập (lỗi Out of Memory). Giải pháp là ta sẽ hy sinh tốc độ bằng cách cho nó học từng bức ảnh một (chỉnh `batch_size = 4` hoặc `8`) thay vì nhồi 16 bức ảnh cùng lúc. Card GPU P100 (16GB VRAM) của Kaggle dư sức gánh được việc này!

#### Chiến lược 2: Dùng thư viện SAHI (Slicing Aided Hyper Inference)
- **Bản chất của việc Cắt ảnh (Tiling):** Nếu ảnh quá to không nhét vừa độ phân giải của AI, ta có thể nghĩ đến việc lấy "kéo" cắt tấm ảnh 1622x626 đó thành 4 tấm ảnh nhỏ (ví dụ mỗi tấm 800x600). Khi đó, đem các tấm ảnh nhỏ này vào AI sẽ không bị bóp méo nữa, biển báo sẽ to rõ như cũ.
- **Rắc rối của việc cắt thủ công:** *"Cắt 1 ảnh thành nhiều ảnh thì có bị sao không?"*. **CÓ THỂ RẤT SAO!** Lưỡi kéo cắt vô tình có thể **cắt đôi cái biển báo ở chính giữa màn hình**, làm nửa cái biển báo rơi vào ảnh trái, nửa cái rơi vào ảnh phải. AI học những bức ảnh bị đứt đôi này sẽ "tẩu hỏa nhập ma" và nhận diện sai bét.
- **Quyền năng của SAHI:** SAHI là một thư viện vô cùng thông minh được sinh ra để trị dứt điểm rắc rối trên. 
  1. Nó không cắt vụn ảnh một cách thô bạo. Nó dùng **"Cửa sổ trượt đè lấp" (Overlapping Sliding Window)**. Tức là nó cắt tấm ảnh thứ 1, rồi khi chuẩn bị cắt tấm ảnh thứ 2, nó sẽ lùi lại một chút (trồng lấn khoảng 20% lên phần ranh giới của tấm ảnh 1) rồi mới cắt tiếp. Nhờ sự đè lấp này, nếu biển báo xui xẻo nằm ở ranh giới và bị đứt đôi ở ảnh 1, nó chắc chắn sẽ lọt TRỌN VẸN vào phần đè lấp của ảnh 2!
  2. Lúc dự đoán xong, nếu có một biển báo được nhận diện ở cả 2 ảnh nhỏ (do phần đè lấp sinh ra trùng lặp), SAHI sẽ dùng thuật toán NMS (Non-Maximum Suppression) để xóa đi cái khung thừa, giữ lại 1 khung duy nhất và ghép trả lại chuẩn xác vào tọa độ của bức ảnh to nguyên bản.

*=> Chốt phương án cực mạnh cho Đồ án:* Ta sẽ huấn luyện YOLO bằng **Chiến lược 1 (Độ phân giải 1024 hoặc 1280)** để nó khôn ra. Sau đó, lúc đem đi Test hoặc chạy Web App thực tế, ta bọc YOLO bằng **Chiến lược 2 (SAHI)**. Combo kết hợp này là đỉnh cao của việc xử lý Small Object Detection!

---

## VII. Đặc tả (Specification) chi tiết các Mô hình

### 1. Specification Mô hình 1 - YOLOv8
- **File thực hiện:** `notebooks/train_yolov8.ipynb`
- **Phong cách Code:** Tuân thủ chặt chẽ `AGENTS.md` (Viết code mộc mạc kiểu sinh viên, comment tiếng Việt cho các khối logic).
- **Quy trình chi tiết trong file Notebook (Chạy trên Kaggle):**
  1. **Khối 1 (Tiền xử lý):** Sử dụng hàm `json.load()` đọc file `train_traffic_sign_dataset.json`. Lặp qua từng bức ảnh, tạo thư mục `dataset/train/images`, copy ảnh gốc vào. Chuyển đổi tọa độ bounding box từ dạng Pixel sang dạng chuẩn hóa của YOLO (từ 0 đến 1) và xuất ra file `.txt` tương ứng trong `dataset/train/labels`. (Tách ngẫu nhiên 80% train, 20% val).
  2. **Khối 2 (Cấu hình):** Code tự động tạo file `dataset.yaml` chứa đường dẫn tới folder data và danh sách ID của 7 loại biển báo.
  3. **Khối 3 (Huấn luyện):** Cài đặt thư viện `ultralytics`. Tải trước trọng số của `yolov8s.pt` (bản Small). Dùng lệnh `model.train()` với cấu hình: `epochs=50` (hoặc test thử với epochs=2), `imgsz=1280` (trị vật thể nhỏ), `batch=8` (chống tràn RAM).
- **Lưu ý triển khai:** Khối 1 và Khối 2 xử lý việc tạo folder và json ngay trên Kaggle RAM, tránh việc phải upload hàng vạn file txt từ máy cá nhân lên gây lỗi mạng.

### 2. Specification Mô hình 2 - Faster R-CNN
- **File thực hiện:** `notebooks/train_faster_rcnn.ipynb`
- **Môi trường chạy:** Google Colab (GPU T4 miễn phí).
- **Phong cách Code:** Code bằng PyTorch thuần (`torchvision`), comment tiếng Việt chi tiết cách xây dựng hàm DataLoader và vòng lặp.
- **Quy trình chi tiết trong file Notebook:**
  1. **Khối 1 (Tải dữ liệu):** Sử dụng Kaggle API để tải trực tiếp dataset Zalo AI về Google Colab.
  2. **Khối 2 (Tạo lớp Dataset):** Code class `ZaloTrafficDataset` kế thừa từ `torch.utils.data.Dataset`. Hàm `__getitem__` sẽ đọc thẳng tọa độ từ file COCO JSON gốc (không cần convert sang YOLO format), xử lý bounding box thành cấu trúc tensor dictionary `{boxes, labels}`.
  3. **Khối 3 (Khởi tạo Mô hình):** Import `fasterrcnn_resnet50_fpn`. Cắt bỏ lớp phân loại mặc định (COCO 91 classes) và thay bằng lớp phân loại mới với 8 classes (7 biển báo + 1 nền).
  4. **Khối 4 (Huấn luyện):** Thiết lập `SGD Optimizer`. Viết vòng lặp `for epoch in range(epochs):` thủ công, nạp batch vào GPU, tính toán Loss, gọi `backward()` và cập nhật trọng số. Lưu trọng số `faster_rcnn_best.pth` nếu Loss giảm.

### 3. Specification Mô hình 3 - DETR (RT-DETR)
- **File thực hiện:** `notebooks/train_rtdetr.ipynb`
- **Môi trường chạy:** Kaggle Notebook (có thể chạy sau khi YOLO train xong).
- **Kiến trúc:** Nhóm quyết định sử dụng **RT-DETR** (Real-Time DETR) thay cho DETR truyền thống. Sự thay đổi này mang tính chiến lược vì RT-DETR khắc phục điểm yếu chí mạng của Transformer là "tốc độ chậm", giúp mô hình có thể đạt Real-time như YOLO nhưng mang trong mình sự chính xác của Transformer.
- **Quy trình chi tiết trong file Notebook:**
  1. **Khối 1 (Chuẩn bị):** Tái sử dụng lại toàn bộ cấu trúc folder YOLO (images, labels) đã được tạo ra từ file `train_yolov8.ipynb`. (Bởi vì RT-DETR của Ultralytics hỗ trợ đọc chung format với YOLO).
  2. **Khối 2 (Huấn luyện):** Load mô hình `rtdetr-l.pt` (bản Large). Gọi hàm `model.train()` với `imgsz=1280` và `epochs=50`.
- **Lưu ý:** Việc sử dụng chung thư viện `ultralytics` cho 2 mô hình (YOLO và RT-DETR) là một điểm sáng, giúp đơn giản hóa pipeline tiền xử lý, tránh viết code rườm rà dễ sinh lỗi.
