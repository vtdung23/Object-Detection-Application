// Lấy các tham chiếu đến DOM HTML
const uploadZone = document.getElementById('uploadZone');
const fileInput = document.getElementById('fileInput');
const loadingOverlay = document.getElementById('loadingOverlay');
const resultZone = document.getElementById('resultZone');
const imageCanvas = document.getElementById('imageCanvas');
const ctx = imageCanvas.getContext('2d');
const btnReset = document.getElementById('btnReset');
const labelsList = document.getElementById('labelsList');

// SRP: Hàm chỉ quản lý sự kiện kéo thả file
uploadZone.addEventListener('click', () => fileInput.click());

uploadZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadZone.style.borderColor = 'var(--primary-color)';
});

uploadZone.addEventListener('dragleave', () => {
    uploadZone.style.borderColor = 'var(--glass-border)';
});

uploadZone.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadZone.style.borderColor = 'var(--glass-border)';
    if (e.dataTransfer.files.length > 0) {
        handleFileUpload(e.dataTransfer.files[0]);
    }
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFileUpload(e.target.files[0]);
    }
});

btnReset.addEventListener('click', () => {
    resultZone.classList.add('hidden');
    uploadZone.classList.remove('hidden');
    fileInput.value = '';
    labelsList.innerHTML = '';
});

// Hàm điều phối chính (Xử lý ảnh -> Gọi API -> Vẽ)
async function handleFileUpload(file) {
    if (!file.type.startsWith('image/')) {
        alert('Vui lòng chọn một file ảnh hợp lệ!');
        return;
    }

    // 1. Hiển thị UI Loading Radar Scan
    loadingOverlay.classList.remove('hidden');

    // 2. Load ảnh vào đối tượng Image của JS để lấy kích thước gốc
    const imgObj = new Image();
    const objectUrl = URL.createObjectURL(file);
    imgObj.src = objectUrl;

    imgObj.onload = async () => {
        // Thiết lập kích thước Canvas khớp với ảnh gốc
        imageCanvas.width = imgObj.width;
        imageCanvas.height = imgObj.height;
        // Vẽ lại ảnh gốc lên Canvas
        ctx.drawImage(imgObj, 0, 0);
        
        // 3. Chuẩn bị gửi file lên Backend FastAPI
        const formData = new FormData();
        formData.append('file', file);

        try {
            // Clean Code: Dùng khối try..catch cho thao tác mạng
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            
            if (response.ok && data.success) {
                // 4. Nếu thành công, vẽ các Bounding Box đè lên ảnh
                drawPredictions(data.predictions);
                // 5. Chuyển đổi trạng thái Giao diện
                loadingOverlay.classList.add('hidden');
                uploadZone.classList.add('hidden');
                resultZone.classList.remove('hidden');
            } else {
                throw new Error(data.error || 'Lỗi không xác định từ Server');
            }
        } catch (error) {
            loadingOverlay.classList.add('hidden');
            alert('Lỗi API: ' + error.message);
        }
    };
}

// Ánh xạ tên nhãn tiếng Anh/ID sang tiếng Việt
const labelMap = {
    '0': 'Cấm ngược chiều',
    '1': 'Cấm dừng và đỗ',
    '2': 'Cấm rẽ',
    '3': 'Giới hạn tốc độ',
    '4': 'Cấm còn lại',
    '5': 'Nguy hiểm',
    '6': 'Hiệu lệnh',
    '7': 'Hiệu lệnh',
    'Cam Nguoc Chieu': 'Cấm ngược chiều',
    'Cam Dung Va Do': 'Cấm dừng và đỗ',
    'Cam Re': 'Cấm rẽ',
    'Gioi Han Toc Do': 'Giới hạn tốc độ',
    'Cam Con Lai': 'Cấm còn lại',
    'Nguy Hiem': 'Nguy hiểm',
    'Hieu Lenh': 'Hiệu lệnh',
    'Max Speed': 'Giới hạn tốc độ',
    'Other prohibition signs': 'Cấm còn lại',
    'No entry': 'Cấm ngược chiều',
    'No parking / waiting': 'Cấm dừng và đỗ',
    'No turn': 'Cấm rẽ',
    'Danger': 'Nguy hiểm',
    'Mandatory': 'Hiệu lệnh'
};

// DRY: Hàm chuyên biệt để vẽ Box, dùng chung cho mọi đối tượng
function drawPredictions(predictions) {
    labelsList.innerHTML = ''; // Xóa nhãn cũ
    
    // SRP: Lọc rác, chỉ giữ lại các biển báo có độ tin cậy >= 50%
    const validPredictions = predictions.filter(pred => pred.score >= 0.5);

    if (validPredictions.length === 0) {
        labelsList.innerHTML = '<span class="badge">Không tìm thấy biển báo nào (Score > 50%)</span>';
        return;
    }

    // Thiết lập style nét vẽ
    const strokeWidth = Math.max(2, imageCanvas.width / 600);
    ctx.lineWidth = strokeWidth; // Viền mỏng
    ctx.strokeStyle = '#ff0000'; // Đỏ (Red)
    ctx.font = `bold ${Math.max(16, imageCanvas.width / 70)}px Arial`;
    
    // Lưu các nhãn duy nhất để hiển thị bên dưới
    const uniqueLabels = new Set();
    let signIndex = 1;

    validPredictions.forEach(pred => {
        const [x1, y1, x2, y2] = pred.bbox;
        const width = x2 - x1;
        const height = y2 - y1;
        
        // Lấy tên tiếng Việt từ labelMap (chuẩn hóa tên gốc để so khớp chính xác)
        const rawLabel = String(pred.label).trim();
        const label = labelMap[rawLabel] || rawLabel;
        const score = (pred.score * 100).toFixed(0) + '%';
        
        // Ghép số thứ tự vào tên nhãn để hiển thị bên dưới ảnh
        const badgeText = `[${signIndex}] ${label} (${score})`;
        uniqueLabels.add(badgeText);

        // Vẽ hộp Bounding Box (chỉ vẽ viền đỏ, không đổ nền)
        ctx.beginPath();
        ctx.rect(x1, y1, width, height);
        ctx.stroke();

        // Đánh số thứ tự lên ảnh thay vì ghi tên dài dòng để chống đè chữ
        const text = `[${signIndex}]`;
        
        // Vị trí chữ: Nằm bên PHẢI của hộp Bounding Box
        const textX = x2 + 5;
        const textY = y1 + 15; // Căn hơi thụt xuống ngang với mép trên của hộp
        
        // Dùng API đo lường chính xác của Canvas để bao khít nền
        const metrics = ctx.measureText(text);
        const textHeight = metrics.actualBoundingBoxAscent + metrics.actualBoundingBoxDescent;
        
        ctx.fillStyle = '#ffffff';
        // Padding nhẹ 2px để chữ không bị chạm viền
        ctx.fillRect(textX - 2, textY - metrics.actualBoundingBoxAscent - 2, metrics.width + 4, textHeight + 4);
        
        // Tô chữ số màu Đỏ thuần túy đè lên nền trắng
        ctx.fillStyle = '#ff0000';
        ctx.fillText(text, textX, textY);
        
        // Phục hồi lại nét vẽ viền Box đỏ cho vòng lặp tiếp theo
        ctx.strokeStyle = '#ff0000';
        ctx.lineWidth = strokeWidth;
        
        signIndex++;
        });

    // Render danh sách Badge
    uniqueLabels.forEach(label => {
        const span = document.createElement('span');
        span.className = 'badge';
        span.textContent = label;
        labelsList.appendChild(span);
    });
}
