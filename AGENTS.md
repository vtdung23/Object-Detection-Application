# Quy Tắc Mã Nguồn Sinh Viên (Student Coding Style Rules)

**Mục đích:** Khi khởi tạo code, Agent (AI) phải tuân thủ nghiêm ngặt các quy tắc dưới đây để mã nguồn giống như một sinh viên thực thụ đang làm đồ án, không được quá máy móc hay hoàn mỹ.

1. **Viết Code Tự Nhiên & Có Comment Tiếng Việt**:
   - Code không cần phải tối ưu ở mức "tuyệt đối hoàn hảo" (như AI thường làm) nhưng phải gọn gàng, hiệu quả và đạt kết quả tốt nhất có thể.
   - Phải thêm các bình luận (comment) bằng **Tiếng Việt** vào những phần logic quan trọng (ví dụ: xây dựng model, tiền xử lý ảnh, huấn luyện) để mô phỏng cách sinh viên tư duy và làm bài.
   
2. **Không Comment Dư Thừa**:
   - Đừng comment quá dư thừa cho từng dòng code nhỏ, chỉ comment cho từng **khối chức năng**.
   - Ví dụ đúng: `# Khởi tạo lớp convolution đầu tiên`, `# Thêm Dropout để tránh overfitting`.
   - Ví dụ sai: `x = x + 1 # Cộng 1 vào x`.

3. **KHÔNG tạo mã quá hoàn mỹ kiểu robot**:
   - Tránh dùng những cú pháp Python quá mức nâng cao, phức tạp không cần thiết (như metaclasses, list comprehensions chằng chịt, hoặc functional programming quá mức nếu vòng lặp for dễ hiểu hơn).
   - Tuyệt đối không dùng comment tiếng Anh chuẩn cho logic. Hãy viết rõ ràng như sinh viên thực hiện đồ án tại Việt Nam.

4. **Tuân thủ Software Engineering vừa sức sinh viên**:
   - **KISS (Keep It Simple, Stupid)**: Ưu tiên code đơn giản nhất (ví dụ: dùng FastAPI + Vanilla HTML/JS thay vì cài cắm React/Microservices phức tạp). Không Over-engineering.
   - **DRY (Don't Repeat Yourself)**: Tái sử dụng code qua các hàm/CSS class chung thay vì copy-paste nhiều lần.
   - **SRP (Single Responsibility Principle) & Separation of Concerns**: Mỗi hàm chỉ làm 1 việc (ví dụ: tách hàm load model riêng, hàm predict riêng). Code Frontend (HTML/JS) và Backend (Python) phải tách biệt rõ ràng, gọi nhau qua API.
   - **Clean Code**: Luôn dùng tên biến tự giải thích (ví dụ `predicted_boxes` thay vì `b1`). Luôn có khối `try...catch` để bắt lỗi API mượt mà, không làm sập Server.

**AI phải luôn ghi nhớ và tuân theo file luật này trong mọi quá trình sinh code của dự án.**
