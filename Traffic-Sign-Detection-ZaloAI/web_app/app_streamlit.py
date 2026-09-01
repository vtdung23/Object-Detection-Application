import streamlit as st
import cv2
import numpy as np
import io
# Import thẳng hàm dự đoán từ utils (nó sẽ tự load model best.pt)
from utils import predict_traffic_signs

# Tiêu đề và cấu hình trang
st.set_page_config(page_title="Nhận diện Biển báo (Streamlit)", layout="wide")
st.title("Hệ thống Nhận diện Biển báo Giao thông")
st.write("Đồ án sử dụng YOLOv8 và SAHI. (Bản gộp Streamlit siêu nhẹ để chạy Free trên Hugging Face)")

# Từ điển ánh xạ Tiếng Việt (Y hệt bản web kia)
label_map = {
    '0': 'Cấm ngược chiều',
    '1': 'Cấm dừng và đỗ',
    '2': 'Cấm rẽ',
    '3': 'Giới hạn tốc độ',
    '4': 'Cấm còn lại',
    '5': 'Nguy hiểm',
    '6': 'Hiệu lệnh',
    '7': 'Hiệu lệnh',
    'Max Speed': 'Giới hạn tốc độ',
    'Other prohibition signs': 'Cấm còn lại',
    'No entry': 'Cấm ngược chiều',
    'No parking / waiting': 'Cấm dừng và đỗ',
    'No turn': 'Cấm rẽ',
    'Danger': 'Nguy hiểm',
    'Mandatory': 'Hiệu lệnh',
    'Cam Nguoc Chieu': 'Cấm ngược chiều',
    'Cam Dung Va Do': 'Cấm dừng và đỗ',
    'Cam Re': 'Cấm rẽ',
    'Gioi Han Toc Do': 'Giới hạn tốc độ',
    'Cam Con Lai': 'Cấm còn lại',
    'Nguy Hiem': 'Nguy hiểm',
    'Hieu Lenh': 'Hiệu lệnh'
}

# Khu vực upload file
uploaded_file = st.file_uploader("Tải lên ảnh cần phân tích (JPG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Đọc file ảnh dưới dạng mảng byte
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    
    # Giải mã ảnh bằng OpenCV
    img = cv2.imdecode(file_bytes, 1)
    
    # OpenCV mặc định dùng BGR, Streamlit/Trình duyệt dùng RGB nên phải đảo màu
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    st.image(img_rgb, caption='Ảnh gốc bạn tải lên', use_column_width=True)
    
    if st.button("Bắt đầu Nhận diện (SAHI)"):
        with st.spinner("Đang chạy thuật toán SAHI băm ảnh (mất vài giây)..."):
            # Lấy giá trị byte ban đầu để truyền vào hàm predict (giống hệt API)
            image_data = uploaded_file.getvalue()
            predictions = predict_traffic_signs(image_data)
            
            # Lọc rác (giữ >= 50%)
            valid_preds = [p for p in predictions if p['score'] >= 0.5]
            
            if not valid_preds:
                st.warning("Không tìm thấy biển báo nào (hoặc độ tin cậy < 50%).")
            else:
                st.success(f"Phân tích hoàn tất! Tìm thấy {len(valid_preds)} biển báo hợp lệ.")
                
                # Tạo một bản sao ảnh để vẽ Box lên
                img_draw = img_rgb.copy()
                sign_index = 1
                
                st.markdown("### Danh sách chi tiết:")
                
                for pred in valid_preds:
                    bbox = pred['bbox']
                    score = pred['score']
                    raw_label = str(pred['label']).strip()
                    label = label_map.get(raw_label, raw_label)
                    
                    x1, y1, x2, y2 = [int(v) for v in bbox]
                    
                    # 1. Vẽ Khung đỏ (Thickness = 2)
                    cv2.rectangle(img_draw, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    
                    # 2. Đánh số thứ tự để chống đè chữ
                    text = f"[{sign_index}]"
                    text_x = x2 + 5
                    text_y = y1 + 15
                    
                    # OpenCV tính toán kích thước text để vẽ cái nền trắng
                    (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    
                    # Vẽ khối hộp màu trắng
                    cv2.rectangle(img_draw, 
                                  (text_x - 2, text_y - text_height - 2), 
                                  (text_x + text_width + 2, text_y + baseline), 
                                  (255, 255, 255), 
                                  cv2.FILLED)
                    
                    # Tô số màu đỏ lên trên cái khối nền trắng
                    cv2.putText(img_draw, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                    
                    # Ghi kết quả chi tiết ra màn hình
                    st.write(f"- **[{sign_index}] {label}** *(Độ tin cậy: {score:.0%})*")
                    sign_index += 1
                
                st.markdown("### Ảnh Kết quả:")
                st.image(img_draw, caption='Ảnh đã được khoanh vùng bởi YOLOv8 + SAHI', use_column_width=True)
