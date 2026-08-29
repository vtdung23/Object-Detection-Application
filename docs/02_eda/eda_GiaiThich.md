## 5. Giải thích chi tiết các khái niệm EDA trong Object Detection

Để bạn dễ dàng đưa vào báo cáo và trình bày trước hội đồng, dưới đây là giải thích chi tiết cho các thuật ngữ và khái niệm xuất hiện trong phần EDA ở trên:

### 5.1. Object (Vật thể) trong ảnh là gì?
- **Khái niệm**: Trong bài toán Object Detection (Nhận diện vật thể), "Object" không phải là toàn bộ bức ảnh, mà là **một thực thể cụ thể** nằm bên trong bức ảnh đó và được đánh dấu bằng một hộp giới hạn (Bounding Box). 
- **Ví dụ cụ thể**: Trong 1 bức ảnh chụp ngã tư đường, có thể có 3 biển báo giao thông. Khi đó:
  - Số lượng Image (ảnh) = 1
  - Số lượng Object (vật thể/biển báo) = 3. Mỗi object sẽ có tọa độ Bounding Box riêng (x, y, width, height) và nhãn (class) riêng (vd: 1 biển cấm rẽ, 2 biển giới hạn tốc độ).

### 5.2. Co-occurrence Matrix (Ma trận đồng xuất hiện) là gì?
- **Khái niệm**: "Đồng xuất hiện" (Co-occurrence) nghĩa là 2 loại biển báo (class) cùng xuất hiện **chung trong một bức ảnh**. Ma trận đồng xuất hiện là một bảng dạng lưới (grid), trong đó mỗi ô giao nhau giữa hàng A và cột B thể hiện số lượng bức ảnh có chứa **cả biển báo loại A và biển báo loại B**.
- **Ý nghĩa thực tiễn**: Giúp phát hiện các quy luật giao thông. Ví dụ: Biển "Cấm đỗ xe" thường hay xuất hiện cùng biển "Cấm quay đầu" tại các ngã tư hẹp. Nếu mô hình học được mối tương quan ngữ nghĩa (Contextual correlation) này, khi nó nhận diện được biển "Cấm đỗ xe", nó sẽ có xu hướng chú ý hơn để tìm biển "Cấm quay đầu" gần đó.

### 5.3. Class Balance (Độ cân bằng nhãn)
- **Khái niệm**: Đo lường xem số lượng object (biển báo) của mỗi class có xấp xỉ bằng nhau hay không.
- **Ý nghĩa thực tiễn**: Nếu class "Cảnh báo nguy hiểm" chỉ có 50 object, trong khi "Cấm đỗ xe" có tới 5000 object, dữ liệu bị **mất cân bằng (Imbalanced)**. Nếu không xử lý, mô hình học máy sẽ bị "lười" và luôn dự đoán là "Cấm đỗ xe" để dễ đạt độ chính xác cao (Accuracy Paradox), dẫn đến việc bỏ sót các biển báo hiếm nhưng quan trọng.

### 5.4. Images & Object Distribution (Phân bố vật thể trên ảnh)
- **Khái niệm**: Biểu đồ thống kê xem phần lớn các bức ảnh trong dataset chứa bao nhiêu vật thể. Có bao nhiêu ảnh chỉ có 1 biển báo? Bao nhiêu ảnh có 2, 3 hay 10 biển báo cùng lúc?
- **Ý nghĩa thực tiễn**: Giúp ta biết mức độ phức tạp của khung hình. Nếu đa số ảnh chỉ có 1 biển báo, mô hình sẽ dễ hội tụ hơn. Nếu ảnh chứa chằng chịt 10-20 biển báo chồng chéo, ta phải dùng các kỹ thuật Non-Maximum Suppression (NMS) gắt gao hơn để tránh việc mô hình nhận diện trùng lặp 1 biển báo nhiều lần.

### 5.5. Class Sizes & Bounding Box Area (Kích thước vật thể)
- **Khái niệm**: Đo lường diện tích của Bounding Box (Width × Height) so với toàn bộ bức ảnh gốc. 
- **Ý nghĩa thực tiễn**: Được dùng để phân loại bài toán thành Small, Medium hay Large Object Detection. Nếu vật thể quá nhỏ (diện tích < 1%), khi đi qua các tầng tích chập (Convolutional Layers) của YOLO, thông tin pixel của biển báo sẽ bị gộp lại, mờ đi và biến mất hoàn toàn ở các Feature Map cuối cùng. Hiểu được điều này giúp ta quyết định không thu nhỏ ảnh gốc (resize) quá đà khi train.

### 5.6. Spatial Heatmap (Bản đồ nhiệt không gian)
- **Khái niệm**: Tưởng tượng bạn xếp chồng toàn bộ 4500 bức ảnh lên nhau, sau đó chấm một điểm đỏ vào vị trí tâm của mọi biển báo giao thông xuất hiện. Nơi nào có nhiều dấu chấm đỏ chồng lên nhau, nơi đó sẽ "nóng" lên (chuyển sang màu đỏ rực), nơi nào ít sẽ có màu xanh lạnh. Đó là bản đồ nhiệt.
- **Ý nghĩa thực tiễn**: Cho ta biết **vị trí địa lý ưu tiên** của vật thể trên khung hình. Trong dataset này, heatmap sẽ đỏ rực ở nửa trên bên phải của ảnh. Từ đó, ta rút ra kinh nghiệm để tinh chỉnh thuật toán Augmentation: Cấm hệ thống tự động cắt (Crop) bỏ phần trên bên phải của bức ảnh khi sinh dữ liệu huấn luyện, vì như thế là tự cắt đi biển báo.

---

## 6. Giải thích chi tiết Menu Lựa chọn Kỹ thuật Triển khai

Phần này giải thích chi tiết ý nghĩa, lý do lựa chọn và tác dụng thực tế của từng kỹ thuật (EDA và Modeling) được liệt kê trong `eda_phan_tich_va_gop_y.md`. Bạn hãy đọc kỹ để chọn ra "combo" phù hợp nhất cho đồ án của mình.

### 🎯 6.1. Giải thích các kỹ thuật vẽ biểu đồ EDA chuyên sâu

*   **[E1] Tổng quan Phân bố nhãn (Class Balance Dashboard)**:
    *   *So với Dataset Ninja*: Làm giống hệt họ bằng cách lập bảng thống kê đa chiều (Images, Objects, Avg count, Avg area) thay vì chỉ vẽ Bar chart đếm số lượng đơn điệu.
    *   *Giúp gì cho model*: Bảng này cung cấp cái nhìn toàn cảnh, tác động trực tiếp đến việc tinh chỉnh tham số huấn luyện:
        - **Cột "Objects" (Dùng Focal Loss)**: Bình thường hàm loss đánh giá mọi lỗi sai như nhau. Nếu biển Cấm đỗ có 10.000 cái, biển Cấm rẽ chỉ có 100 cái, model sẽ có xu hướng đoán mọi thứ là Cấm đỗ để dễ có điểm cao. Nhìn vào cột Objects bị lệch, ta biết phải bật `Focal Loss`. Hàm này hoạt động theo cơ chế tự động giảm nhẹ hình phạt với class dễ (nhiều data) và "phạt cực kỳ nặng" khi model đoán sai class khó (hiếm data), ép model không được bỏ rơi biển Cấm rẽ.
        - **Cột "Avg count" (Giới hạn tham số `max_det`)**: Các mạng như YOLO có thông số `max_det` (số lượng bounding box tối đa xuất ra trên 1 ảnh, mặc định là 300). Nếu Avg count chỉ báo trung bình 1.5 biển báo/ảnh, ta có thể tự tin hạ `max_det` xuống 30-50. Việc này giúp **tăng trực tiếp chỉ số FPS (Frame Per Second - Tốc độ khung hình/giây)** nhờ giảm tải cực lớn cho thuật toán hậu xử lý **NMS (Non-Maximum Suppression)**. 
          *(Giải thích thêm về cơ chế NMS & FPS)*: Khi YOLO phân tích 1 bức ảnh, nó thường vạch ra hàng trăm bounding box chồng chéo lên nhau quanh 1 vật thể. Thuật toán NMS có nhiệm vụ quét qua toàn bộ các box này, so sánh độ trùng lặp (chỉ số IoU) để giữ lại duy nhất 1 box có điểm tự tin (Confidence score) cao nhất và xóa bỏ các box thừa đi. Quá trình quét và so sánh này rất tốn tài nguyên CPU. Bằng cách ép model chỉ được xuất ra tối đa 50 box (`max_det=50`) ngay từ đầu thay vì 300, NMS sẽ có cực ít dữ liệu phải xử lý. Nhờ vậy, tốc độ phản hồi tổng thể của model (FPS) sẽ nhanh hơn đáng kể, rất quan trọng khi chạy thực tế trên camera hành trình ô tô.
*   **[E2] Phân bố Mật độ vật thể (Object Distribution Heatmap)**:
    *   *So với Dataset Ninja*: Nâng cấp từ biểu đồ cột thông thường thành một lưới Heatmap (Hàng là Class, Cột là Số lượng object 1,2,3...).
    *   *Giúp gì cho model*: Cho thấy mức độ "đông đúc" của từng loại biển báo. Nếu biển báo "Cấm đỗ" thường xuất hiện 3-4 cái trong 1 ảnh (tức là mật độ dày đặc), model sẽ bắt buộc phải tinh chỉnh tham số **IoU (Intersection over Union) threshold** trong hàm NMS.
        *(Giải thích thêm về cách chỉnh IoU Threshold)*: Khi có nhiều biển báo đứng sát nhau (ví dụ cắm chung trên 1 cột điện), các bounding box của chúng sẽ bị đè (chồng chéo) lên nhau. Hàm NMS dùng `IoU threshold` (mặc định thường là 0.45) làm mốc: nếu 2 hộp chồng lên nhau lớn hơn 45%, nó sẽ xóa bớt 1 hộp vì tưởng model dự đoán lặp lại cùng 1 vật. Nhưng vì EDA (biểu đồ E2) báo cho ta biết biển báo thực tế đứng sát nhau rất nhiều, ta phải **tăng chỉ số IoU threshold lên cao (ví dụ 0.65 - 0.7)**. Việc này "ra lệnh" cho model: *"Chỉ xóa hộp khi chúng chồng lên nhau quá 70%, còn nếu chỉ đè 50% thì hãy giữ lại cả hai, vì đó rất có thể là 2 biển báo khác nhau nằm cạnh nhau!"*. Nhờ sự can thiệp này từ EDA, model sẽ không bị xóa nhầm các biển báo hợp lệ.
*   **[E3] Kích thước chi tiết & Tree Map (Class Sizes & Tree Map)**:
    *   *So với Dataset Ninja*: Khôi phục lại trọn vẹn bảng thống kê chi tiết (Min/Max/Avg cho Width, Height, Area) và vẽ biểu đồ Tree Map diện tích.
    *   *Giúp gì cho model*: Chứng minh một cách định lượng đây là bài toán "Small Object Detection" (khi Avg area chỉ loanh quanh 0.1%). Đây là căn cứ khoa học tuyệt đối để kích hoạt các tính năng chuyên trị vật thể nhỏ như **P2 Layer (trong YOLO)** hoặc áp dụng kỹ thuật **SAHI (Slicing Aided Hyper Inference)**.
        *(Giải thích & Tư vấn chi tiết về P2 Layer và SAHI)*:
        - **P2 Layer là gì?** Mặc định, YOLO có 3 đầu ra dự đoán (gọi là P3, P4, P5) để tìm vật thể kích thước Vừa, Lớn, Siêu Lớn. Vì ảnh phải đi qua nhiều tầng tích chập, nó bị thu nhỏ (downsample) nhiều lần, khiến các pixel của biển báo siêu nhỏ bị hòa tan và biến mất. Kích hoạt P2 Layer nghĩa là ta mở thêm 1 đầu ra dự đoán ở tầng nông hơn (khi ảnh chưa bị thu nhỏ quá nhiều), giúp YOLO "nhìn" rõ được các biển báo li ti. 
        - **SAHI là gì?** SAHI không phải là sửa model, mà là kỹ thuật xử lý ảnh đầu vào. Thay vì nhét cả bức ảnh 1622x626 khổng lồ vào model để dự đoán, SAHI sẽ cắt bức ảnh đó thành các ô vuông nhỏ hơn (ví dụ cắt thành các mảng 512x512) và di chuyển lướt qua toàn bộ ảnh. Nhờ cắt nhỏ ảnh, biển báo tự nhiên trở nên "to hơn" một cách tương đối so với cái khung hình 512x512 mới, giúp model nhận diện siêu nét. Sau khi đoán xong các ô nhỏ, SAHI tự động ráp tọa độ lại vào vị trí trên ảnh gốc.
        - **Tư vấn (Nên dùng cái nào?):** Khuyên bạn nên **DÙNG CẢ HAI** nhưng ở 2 thời điểm khác nhau. Khi huấn luyện (Training YOLO), hãy cấu hình bật P2 Layer để model sinh ra trọng số nhạy cảm với vật nhỏ. Khi đem model đi dự đoán thực tế (Testing/Inference), hãy nhúng model đó vào luồng chạy của SAHI. Đây là combo hủy diệt giúp mAP tăng vọt.
        - **Hai mô hình kia (Faster R-CNN, RT-DETR) có cần không?** 
          - *Với P2 Layer*: Không cần. P2 là tên gọi đặc thù của kiến trúc YOLO. Mạng Faster R-CNN bản thân nó đã xài kiến trúc FPN (Feature Pyramid Network) - một cơ chế khai thác đặc trưng đa tầng tự nhiên đã rất mạnh với vật nhỏ rồi. Mạng RT-DETR dùng kiến trúc Transformer (xét mọi pixel đồng thời) nên cũng không có khái niệm P2 Layer.
          - *Với SAHI*: **Vẫn rất CẦN**. SAHI là thuật toán độc lập không phụ thuộc vào model (Model-Agnostic). Tức là bạn bọc YOLO, Faster R-CNN hay RT-DETR vào trong SAHI thì nó đều tự động cắt ảnh ra giùm bạn. Áp dụng chung SAHI cho cả 3 model khi test sẽ tạo ra một môi trường so sánh công bằng nhất cho báo cáo đồ án.
*   **[E4] Bản đồ nhiệt Không gian (Spatial Heatmap)**:
    *   *So với Dataset Ninja*: Tái hiện lại bản đồ nhiệt quét vị trí tâm của biển báo trên toàn bộ 4500 ảnh.
    *   *Giúp gì cho model*: Khám phá ra "vùng mù" và "vùng mật độ cao" (biển báo hay nằm ở lề phải khung hình). Từ đó, ta code các lớp Data Augmentation (như Random Crop/Cutout) một cách thông minh: cấm cắt xén ngẫu nhiên vào góc phải của bức ảnh để không tự hủy dữ liệu huấn luyện.
        *(Giải thích kỹ: Ta sẽ code gì và code thế nào?)*:
        - **Vấn đề của Code mặc định**: Bình thường khi dùng hàm cắt ảnh (`RandomCrop`), máy tính sẽ "nhắm mắt" cắt 1 mảng tọa độ bất kỳ để đem đi train. Nếu nó vô tình cắt đúng vào góc lề phải (nơi chứa 90% biển báo theo như Heatmap đã chứng minh), thì mảng ảnh bị cắt ra có thể chỉ toàn bầu trời hoặc cây cối. NHƯNG, nhãn (label) của bức ảnh gốc vẫn bị hệ thống gán mặc định cho mảng ảnh này. Hậu quả là model bị ép học 1 bức ảnh bầu trời nhưng phải tin đó là "Biển cấm đỗ". Điều này sinh ra dữ liệu rác làm model học sai lệch.
        - **Giải pháp (Ta sẽ code thế nào?)**: Thay vì dùng code Augmentation mặc định của PyTorch, ta sẽ sử dụng thư viện chuyên dụng **`Albumentations`**. Thư viện này hỗ trợ kỹ thuật **BBox-safe Augmentation** (Augmentation an toàn cho hộp giới hạn). 
          - *Cụ thể trong code Python:* Khi định nghĩa lớp cắt ảnh `A.RandomCrop`, ta sẽ gài thêm tham số `BboxParams(min_visibility=0.5)`. 
          - *Cơ chế hoạt động:* Tham số này đóng vai trò như một người gác cổng. Nó cho phép thuật toán cắt ảnh thoải mái, nhưng sau khi cắt, nó phải lấy tọa độ mới đi kiểm tra chéo với tọa độ Bounding box cũ. Nếu biển báo bị lẹm mất quá 50% diện tích (hoặc biến mất hoàn toàn), hệ thống sẽ **hủy kết quả đó** và ép máy tính cắt lại ở một vùng an toàn khác (ví dụ vùng lề trái không có biển báo).
          - *Kết quả:* Data sinh ra luôn sạch sẽ, model học được sự đa dạng bối cảnh (cắt trái, phải, trên, dưới) nhưng tuyệt đối không bao giờ bị mất dữ liệu biển báo.
*   **[E5] Ma trận Đồng xuất hiện (Co-occurrence Matrix)**:
    *   *So với Dataset Ninja*: Tính toán số lần 2 class bất kỳ cùng xuất hiện trong một bức ảnh và vẽ Heatmap Matrix.
    *   *Giúp gì cho model*: Cung cấp Contextual Awareness (Nhận thức ngữ cảnh). Nếu "Biển cảnh báo nguy hiểm" và "Biển hạn chế tốc độ" hay đi liền với nhau, ta có thể dùng kiến thức này để sửa lỗi sai của mô hình khi nó dự đoán các biển báo mờ lân cận.
*   **[E6] Tỷ lệ Khung hình (Aspect Ratio Distribution) - *Phần bổ sung độc quyền***:
    *   *Tại sao phải bổ sung*: Dataset Ninja chỉ cung cấp thông số Width/Height đơn lẻ, nhưng thứ mà mô hình AI thực sự quan tâm là tỷ lệ tỷ đối (Aspect Ratio = Width / Height).
    *   *Giúp gì cho model*: Các kiến trúc mạng Anchor-based (như Faster R-CNN, YOLOv5, YOLOv7) khi bắt đầu nhận diện sẽ tung ra hàng ngàn các "hộp ảo" (Anchor Box) với nhiều hình dáng mặc định (ví dụ: hộp dài như con người tỷ lệ 1:3, hộp dẹt như ô tô tỷ lệ 2:1). 
        *(Giải thích kỹ: Tác dụng của việc nắm rõ Aspect Ratio)*:
        - **Vấn đề của Anchor mặc định**: Nếu ta lười biếng dùng nguyên cấu hình mặc định, model sẽ tung ra các hộp dài và dẹt để tìm biển báo. Nhưng thực tế biển báo giao thông Việt Nam đa số là hình tròn hoặc hình vuông (tỷ lệ chuẩn 1:1). Để ép một cái hộp dài 1:3 bao vừa khít một cái biển báo hình tròn 1:1, model phải tốn hàng chục Epoch (chu kỳ huấn luyện) chật vật học cách "co ngắn" cái hộp lại. Việc này gây lãng phí tài nguyên tính toán cực lớn và làm giảm độ chính xác tổng thể.
        - **Cách giải quyết từ EDA**: Biểu đồ Scatter Plot của E6 sẽ chứng minh bằng toán học rằng 95% biển báo trong dataset Zalo tập trung ở mốc tỷ lệ 1:1. Dựa vào bằng chứng này, ta chạy thuật toán **K-Means Clustering** (Kỹ thuật M3.1) để tự động đúc ra một bộ Anchor Box mới "đo ni đóng giày" riêng cho đồ án này (chỉ gồm các hộp vuông hoặc hơi chữ nhật). 
        - **Kết quả**: Khi train, model tung ra các hộp vuông 1:1, khớp gần như hoàn hảo với biển báo ngay từ những Epoch đầu tiên. Model không phải học cách "bóp méo" hộp nữa, dẫn đến tốc độ hội tụ siêu nhanh và sai số (Loss) giảm chạm đáy. *(Lưu ý: Mạng YOLOv8 là kiến trúc Anchor-free nên không dùng kỹ thuật này, nhưng nếu chạy Faster R-CNN thì đây là kỹ thuật bắt buộc phải làm để ghi điểm với hội đồng).*

### ⚙️ 6.2. Giải thích các kỹ thuật tối ưu Model

**Nhóm M1: Xử lý Mất cân bằng dữ liệu (Imbalanced Data)**
*   **[M1.1] Focal Loss / Class Weights**:
    *   *Lý do*: Các hàm loss thông thường sẽ đánh giá mọi sai sót như nhau. Focal Loss sẽ tự động hạ thấp "tiền phạt" với các class dễ (có số lượng nhiều) và "phạt nặng" hơn khi model đoán sai các class khó/hiếm.
    *   *Tác dụng*: Trực tiếp cải thiện chỉ số mAP cho các nhóm biển báo thiểu số, giúp độ chính xác trung bình tăng lên.
*   **[M1.2] Mosaic & MixUp Augmentation**:
    *   *Lý do*: Thiếu dữ liệu của một số class thì ta phải sinh thêm dữ liệu bằng cách trộn nhiều bức ảnh lại với nhau.
    *   *Tác dụng*: Vừa giúp cân bằng dữ liệu, vừa ép model học được đặc trưng vật thể trong nhiều bối cảnh lộn xộn khác nhau, tăng khả năng tổng quát hóa (generalization).

**Nhóm M2: Xử lý Vật thể siêu nhỏ (Small Objects)**
*   **[M2.1] SAHI (Slicing Aided Hyper Inference)**:
    *   *Lý do*: Với ảnh độ phân giải quá cao (1622x626) mà biển báo chỉ 10x10, khi model resize ảnh về 640x640, biển báo sẽ mờ đi và biến mất hoàn toàn. SAHI giải quyết bằng cách cắt bức ảnh to ra thành nhiều mảnh nhỏ, cho model dự đoán từng mảnh rồi ghép lại.
    *   *Tác dụng*: Cải thiện cực kỳ rõ rệt khả năng nhận diện các biển báo siêu nhỏ ở tận chân trời. Đây là kỹ thuật "chuẩn công nghiệp" sẽ làm hội đồng ấn tượng mạnh.
*   **[M2.2] High-resolution Training / Zoom-in Augmentation**:
    *   *Lý do*: Thay vì dùng SAHI lúc suy luận, ta xử lý ngay lúc train bằng cách không cho model thu nhỏ ảnh, hoặc ép hệ thống tự động "zoom in" vào các góc chứa biển báo.
    *   *Tác dụng*: Giúp các Feature Map cuối cùng trong mạng CNN bảo toàn được thông tin pixel của vật thể nhỏ.
*   **[M2.3] Tinh chỉnh Feature Map (YOLOv8)**:
    *   *Lý do*: Mặc định YOLO dùng 3 nhánh đầu ra (P3, P4, P5) tập trung bắt các vật thể vừa và lớn.
    *   *Tác dụng*: Ta can thiệp vào file cấu hình (yaml) của YOLO để mở thêm nhánh `P2` (nhánh có kích thước lưới lớn nhất nhưng cũng tốn RAM nhất), giúp nó dò tìm các vật cực nhỏ một cách nhạy bén hơn hẳn.

**Nhóm M3: Xử lý Khung hình & Vị trí**
*   **[M3.1] Anchor Box K-Means Clustering**:
    *   *Lý do*: Các mạng Faster R-CNN hay YOLO phiên bản cũ dùng bộ anchor box sinh ra từ dataset chung chung bên ngoài. Ta dùng thuật toán học máy K-Means để gom cụm kích thước của chính bộ dữ liệu Zalo này, sinh ra bộ anchor "đo ni đóng giày".
    *   *Giải thích chi tiết (Anchor nằm ở đâu và K-Means tìm ra nó thế nào?)*:
        - **Anchor nằm ở đâu?**: Nó không nằm trên bức ảnh, mà là các "chiếc lưới ảo" được lập trình chìm bên trong mạng RPN (Region Proposal Network) của Faster R-CNN. Khi mô hình quét qua lưới Feature Map (ví dụ lưới 80x80), tại TỪNG Ô VUÔNG trên lưới, nó sẽ tự động tung ra một chùm các Anchor Box (ví dụ tung 3 hộp: 1 hộp vuông, 1 hộp dọc, 1 hộp dẹt ngang) để "ướm thử" xem có bắt trúng vật thể nào không.
        - **Dùng K-Means tìm ra nó thế nào?**: Đầu tiên, ta rút trích toàn bộ hàng chục ngàn Bounding Box thật trong dataset Zalo ra. Ta vứt bỏ tọa độ X, Y (vì không quan tâm nó nằm ở đâu trên ảnh), chỉ giữ lại kích thước Chiều Rộng (Width) và Chiều Cao (Height). Ta chấm các cặp (Width, Height) này lên một đồ thị 2D. Sau đó, ta chạy thuật toán gom cụm **K-Means Clustering**, yêu cầu nó gom hàng ngàn chấm này thành $K$ tâm cụm (ví dụ $K=5$). K-Means sẽ tự động tính toán khoảng cách và chốt ra 5 hình dáng đại diện phổ biến nhất (ví dụ: hộp vuông 10x10, hộp vuông 20x20, chữ nhật đứng 15x20...). Ta lấy 5 kích thước chuẩn xác này nạp vào code của Faster R-CNN để làm Anchor mặc định thay thế cho Anchor gốc.
    *   *Tác dụng*: Model không phải tốn thời gian học cách "co giãn" hộp dự đoán quá nhiều vì đã có hộp neo chuẩn, dẫn đến quá trình training hội tụ nhanh và loss giảm sâu hơn cực kỳ nhiều.
*   **[M3.2] Safe Spatial Augmentation (Hạn chế crop vùng lề phải)**:
    *   *Lý do*: Biển báo nằm ở lề phải, nếu hệ thống áp dụng cắt ảnh random ngẫu nhiên sẽ có tỷ lệ lớn xóa luôn vùng chứa biển báo.
    *   *Tác dụng*: Bằng cách cấu hình thư viện Albumentations chỉ thực hiện cắt, xoay ở các vùng an toàn (ví dụ: cắt ở góc dưới bên trái), ta đảm bảo tập dữ liệu huấn luyện không bị sinh ra các bức ảnh "rác" (ảnh nhiễu nhưng vẫn bị giữ nhầm nhãn bbox).
