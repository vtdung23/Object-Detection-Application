import os
import tempfile
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
from utils import load_ai_model, predict_traffic_signs

# Khởi tạo App FastAPI
app = FastAPI(title="Traffic Sign Detection API")

# Cấu hình thư mục Frontend
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Nạp model AI ngay khi Server vừa khởi động
MODEL_LOADED = load_ai_model(weights_path="weights/best.pt", device="cpu")

@app.get("/")
async def serve_dashboard(request: Request):
    """
    Render giao diện chính (HTML) của Web App.
    """
    return templates.TemplateResponse(request=request, name="index.html")

@app.post("/predict")
async def api_predict(file: UploadFile = File(...)):
    """
    API Nhận ảnh từ Frontend, xử lý và trả về Bounding Box.
    Tuân thủ Clean Code: Luôn có khối try...catch để bắt lỗi mượt mà.
    """
    if not MODEL_LOADED:
        return JSONResponse(content={"error": "Mô hình AI chưa được nạp do lỗi file weights."}, status_code=500)
    
    try:
        # Lưu file ảnh do người dùng upload ra thư mục tạm (tempfile)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            contents = await file.read()
            temp_file.write(contents)
            temp_file_path = temp_file.name

        # Gọi logic xử lý ảnh từ utils.py
        predicted_boxes = predict_traffic_signs(temp_file_path)

        # Dọn dẹp: Xóa file ảnh nháp để giải phóng ổ cứng
        os.remove(temp_file_path)

        # Trả kết quả thành công cho Frontend
        return JSONResponse(content={"success": True, "predictions": predicted_boxes})

    except Exception as e:
        # Lỗi sẽ được trả về dạng JSON để Frontend hiển thị thông báo đàng hoàng
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    # Chạy Server ở cổng 7860
    uvicorn.run(app, host="0.0.0.0", port=7860)
