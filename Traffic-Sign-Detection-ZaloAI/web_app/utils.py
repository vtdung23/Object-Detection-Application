import cv2
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

# Biến toàn cục lưu model để không load lại nhiều lần
_detection_model = None

def load_ai_model(weights_path="weights/best.pt", device="cpu"):
    """
    Load mô hình YOLOv8 thông qua thư viện SAHI.
    Tuân thủ SRP: Hàm này chỉ chịu trách nhiệm nạp mô hình vào RAM.
    """
    global _detection_model
    try:
        _detection_model = AutoDetectionModel.from_pretrained(
            model_type='yolov8',
            model_path=weights_path,
            confidence_threshold=0.25,
            device=device
        )
        print("Đã nạp thành công mô hình AI!")
        return True
    except Exception as e:
        print(f"Lỗi khi load mô hình: {e}")
        return False

def predict_traffic_signs(img_path):
    """
    Thực hiện dự đoán ảnh với SAHI, Soft-NMS và giới hạn max_det=50.
    """
    if _detection_model is None:
        raise ValueError("Mô hình chưa được nạp!")

    # Đọc ảnh gốc bằng OpenCV và chuyển sang RGB
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError("Không thể đọc được file ảnh hợp lệ.")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Thực hiện SAHI: Tối ưu đánh đổi - Cắt ngang 640px, dọc 512px để ép mô hình Zoom ảnh 2.5x
    result = get_sliced_prediction(
        img,
        _detection_model,
        slice_height=512,
        slice_width=640,
        overlap_height_ratio=0.25,
        overlap_width_ratio=0.25,
        postprocess_type="NMS",
        postprocess_match_metric="IOU",
        postprocess_match_threshold=0.6,
    )

    # Lọc lấy danh sách các hộp dự đoán
    predictions = result.object_prediction_list
    
    # Sắp xếp theo độ tin cậy giảm dần và chỉ lấy tối đa 50 biển báo (max_det=50)
    predictions = sorted(predictions, key=lambda x: x.score.value, reverse=True)[:50]

    # Trích xuất dữ liệu sang chuẩn JSON đơn giản để gửi cho Frontend
    output = []
    for obj in predictions:
        bbox = obj.bbox.to_xyxy() # Lấy mảng [x_min, y_min, x_max, y_max]
        output.append({
            "bbox": bbox,
            "score": float(obj.score.value),
            "label": obj.category.name
        })

    return output
