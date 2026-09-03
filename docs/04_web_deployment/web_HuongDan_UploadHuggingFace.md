# Hướng dẫn Upload Trọng số lên Hugging Face Hub

> **Mục đích:** đưa file trọng số `best.pt` của YOLOv8s-P2 lên Hugging Face Hub để
> Streamlit App tải về lúc chạy, thay vì nhét file nặng vào repo GitHub.

---

## 0. Vì sao phải qua Hugging Face mà không để file thẳng trong repo?

Có ba lý do, xếp theo mức độ quan trọng:

1. **GitHub chặn file trên 100 MB.** File YOLOv8s-P2 chỉ 20.8 MB nên về lý thuyết
   vẫn lọt, nhưng nếu sau này bạn đổi sang RT-DETR-L (251.5 MB) thì tắc ngay. Làm
   đúng ngay từ đầu thì sau đổi mô hình chỉ cần upload lại, không phải sửa code.
2. **Repo Git phình to không kiểm soát.** Git lưu lại **mọi phiên bản** của file
   nhị phân. Mỗi lần train lại và commit đè `best.pt` là repo lại nặng thêm 20 MB
   vĩnh viễn, kể cả khi bạn xoá file đi ở commit sau.
3. **Tách bạch code và trọng số.** Đây là quy ước chuẩn trong ngành: repo Git giữ
   mã nguồn, model registry (Hugging Face Hub) giữ trọng số. Trong báo cáo đồ án,
   đây là một điểm cộng về mặt quy trình MLOps.

File `app/app.py` đã ghim sẵn tên file là **`best.pt`**, nên khi upload bắt buộc
phải đặt đúng tên đó trong repo Hub.

---

## 1. Chuẩn bị

### 1.1. Tài khoản Hugging Face

Đăng ký miễn phí tại https://huggingface.co/join. Username bạn chọn ở bước này
chính là phần đứng trước dấu `/` trong repo id, ví dụ `vtdungfitus/ten-repo`.

### 1.2. Xác định file `best.pt` đang nằm ở đâu

Sau khi train xong trên Kaggle, bạn tải về gói `yolov8_v3_results.zip`. Giải nén
ra, file cần tìm nằm ở:

```
runs/detect/yolov8s_p2_v3/weights/best.pt
```

Nhắc lại phân biệt hai file trong thư mục `weights/`:

| File | Nội dung | Dùng để |
|------|----------|---------|
| `best.pt` | Trọng số tại epoch có mAP tốt nhất | **Upload cái này** |
| `last.pt` | Trọng số tại epoch cuối cùng | Chỉ dùng khi cần train tiếp |

Mở PowerShell tại thư mục chứa file và kiểm tra dung lượng để chắc chắn không
nhầm file:

```powershell
Get-Item best.pt | Select-Object Name, @{N='MB';E={[math]::Round($_.Length/1MB,1)}}
```

Kết quả phải ra khoảng **20.8 MB**. Nếu ra 6 MB thì bạn đang cầm nhầm file
`yolov8n.pt` gốc của Ultralytics chứ không phải model đã train.

### 1.3. Cài công cụ dòng lệnh `hf`

Kiểm tra xem đã có sẵn chưa:

```powershell
hf version
```

Nếu ra dòng `version=1.xx.x` là đã có, bỏ qua bước cài. Nếu báo lỗi
`is not recognized as the name of a cmdlet`, cài qua pip:

```powershell
python -m pip install -U huggingface_hub
```

> **Lưu ý cho máy đã từng gặp lỗi PATH với `kaggle`:** nếu sau khi cài mà gõ `hf`
> vẫn báo không tìm thấy lệnh, dùng dạng gọi qua module thay thế — mọi lệnh trong
> tài liệu này chỉ cần đổi `hf` thành `python -m huggingface_hub.cli.hf`. Ví dụ:
> `python -m huggingface_hub.cli.hf auth login`.

---

## 2. Đăng nhập

### 2.1. Tạo Access Token

Token của Hugging Face **phải có quyền ghi**, token mặc định chỉ có quyền đọc và
sẽ bị từ chối khi upload.

1. Vào https://huggingface.co/settings/tokens
2. Bấm **New token** (hoặc **Create new token**)
3. Đặt tên bất kỳ, ví dụ `do-an-bien-bao`
4. Ở phần loại token, chọn **Write** — đây là bước hay bị bỏ sót nhất
5. Bấm **Create token**, rồi **copy chuỗi token** hiện ra

> **Token chỉ hiện đúng một lần.** Đóng trang mà chưa copy thì phải tạo token mới,
> không có cách xem lại. Chuỗi này có dạng `hf_xxxxxxxxxxxxxxxxxxxx`.

### 2.2. Đăng nhập bằng token

```powershell
hf auth login
```

Terminal sẽ hỏi:

```
Enter your token (input will not be visible):
```

Dán token vào rồi Enter. **Màn hình sẽ không hiện ra ký tự nào** khi bạn dán —
đây là hành vi cố ý để token không lộ ra, không phải lỗi treo máy. Cứ dán và Enter.

Câu hỏi tiếp theo:

```
Add token as git credential? (Y/n)
```

Gõ `n` cũng được, ta không dùng git để đẩy file.

### 2.3. Kiểm tra đã đăng nhập đúng chưa

```powershell
hf auth whoami
```

Phải in ra đúng username của bạn. Nếu ra `Not logged in` thì token dán bị thiếu
ký tự, làm lại mục 2.2.

---

## 3. Tạo Model Repository

```powershell
hf repos create traffic-sign-yolov8s-p2 --type model
```

Giải thích tham số:

- `traffic-sign-yolov8s-p2` — tên repo. Đặt gì cũng được, nhưng nhớ để đối chiếu
  với code ở Phần 5.
- `--type model` — **bắt buộc**. Thiếu tham số này, Hub sẽ tạo nhầm thành dataset
  repo và `hf_hub_download` trong app sẽ báo lỗi 404 không tìm thấy.

Lệnh chạy xong sẽ in ra đường dẫn đầy đủ, dạng:

```
https://huggingface.co/vtdungfitus/traffic-sign-yolov8s-p2
```

Phần `vtdungfitus/traffic-sign-yolov8s-p2` chính là **repo id** — ghi nhớ để dùng
ở Phần 5.

> **Để repo ở chế độ Public.** Đây là mặc định. Nếu bạn thêm `--private`, app trên
> Streamlit sẽ không tải được model trừ khi khai báo thêm token vào Secrets. Với
> đồ án môn học thì để public là đơn giản và đủ dùng.

---

## 4. Upload file `best.pt`

Mở PowerShell tại thư mục chứa file `best.pt`, rồi chạy:

```powershell
hf upload vtdungfitus/traffic-sign-yolov8s-p2 best.pt best.pt --type model
```

Cấu trúc lệnh có ba phần dễ nhầm, đọc kỹ thứ tự:

```
hf upload  <repo-id>  <đường-dẫn-file-trên-máy>  <tên-file-trên-Hub>
```

Tham số thứ hai là file trên máy bạn, tham số thứ ba là tên file sẽ nằm trên Hub.
Ở đây cả hai đều là `best.pt` nên nhìn hơi khó phân biệt. Nếu file trên máy nằm ở
chỗ khác, ví dụ đường dẫn đầy đủ, thì lệnh sẽ là:

```powershell
hf upload vtdungfitus/traffic-sign-yolov8s-p2 "D:\tai-ve\runs\detect\yolov8s_p2_v3\weights\best.pt" best.pt --type model
```

**Tham số thứ ba luôn phải là `best.pt`**, vì hằng số `HF_TEN_FILE` trong
`app/app.py` đang ghim đúng tên này.

Upload xong, lệnh in ra link tới file. Mở link đó bằng trình duyệt, vào tab
**Files and versions**, phải thấy `best.pt` với dung lượng khoảng 20.8 MB.

---

## 5. Trỏ App tới đúng repo

Mở `app/app.py`, tìm dòng:

```python
HF_REPO_ID_MAC_DINH = "vtdungfitus/traffic-sign-yolov8s-p2"
```

Sửa thành repo id thật của bạn nếu khác. Chỉ cần đúng dòng này là app chạy được.

### Cách thay repo mà không phải sửa code

App đọc repo id theo thứ tự ưu tiên: **Streamlit Secrets → biến môi trường
`HF_REPO_ID` → hằng số trong code**. Nhờ vậy bạn có thể đổi repo lúc chạy thử ở
máy mà không đụng vào file nguồn:

```powershell
$env:HF_REPO_ID = "tai-khoan-khac/repo-khac"
streamlit run app/app.py
```

Còn khi deploy lên Streamlit Community Cloud, vào **Settings → Secrets** của app
và thêm:

```toml
HF_REPO_ID = "vtdungfitus/traffic-sign-yolov8s-p2"
```

---

## 6. Kiểm tra trước khi deploy

Chạy thử ngay tại máy để chắc chắn app tải được model từ Hub:

```powershell
streamlit run app/app.py
```

Vào tab **Demo**, upload một ảnh bất kỳ. Lần chạy đầu tiên sẽ mất một lúc vì app
đang tải 20.8 MB trọng số từ Hub về cache. Các lần sau dùng lại file đã cache nên
nhanh hơn hẳn.

Nếu muốn kiểm tra riêng khâu tải model mà không cần mở giao diện:

```powershell
python -c "from huggingface_hub import hf_hub_download; print(hf_hub_download(repo_id='vtdungfitus/traffic-sign-yolov8s-p2', filename='best.pt'))"
```

Lệnh này in ra đường dẫn file trong cache là coi như khâu Hub đã thông.

---

## 7. Xử lý lỗi thường gặp

| Thông báo lỗi | Nguyên nhân | Cách sửa |
|---|---|---|
| `401 Unauthorized` khi upload | Token chỉ có quyền Read | Tạo token mới với quyền **Write** (mục 2.1), rồi `hf auth login` lại |
| `RepositoryNotFoundError` / `404` | Sai repo id, hoặc lỡ tạo repo dạng dataset | Kiểm tra `hf repos create` có kèm `--type model` chưa |
| `Repository Not Found` dù repo tồn tại | Repo đang để private | Vào Settings của repo trên Hub, đổi sang Public |
| `EntryNotFoundError: best.pt` | Đã upload nhưng đặt tên khác | Vào tab Files trên Hub xem tên thật, hoặc sửa `HF_TEN_FILE` trong `app.py` |
| Gõ `hf` báo không tìm thấy lệnh | `hf` chưa nằm trong PATH | Dùng `python -m huggingface_hub.cli.hf ...` thay thế |
| Dán token mà màn hình trống trơn | Không phải lỗi | Terminal cố tình ẩn token, cứ Enter bình thường |

---

## 8. Khi train lại và muốn cập nhật trọng số

Chạy lại đúng lệnh upload ở Phần 4. Hub sẽ tạo một commit mới đè lên file cũ, và
điều đáng nói là **phiên bản cũ vẫn được giữ nguyên trong lịch sử** — vào tab
Files and versions là xem lại được. App luôn tải bản mới nhất nên không cần sửa
gì thêm.

Nên kèm ghi chú cho lần cập nhật để sau này còn lần ra:

```powershell
hf upload vtdungfitus/traffic-sign-yolov8s-p2 best.pt best.pt --type model --commit-message "Train lai 100 epoch, mAP@50-95 tang len 45.2%"
```

---

## Tóm tắt toàn bộ quy trình

```powershell
# 1. Đăng nhập (dán token quyền Write)
hf auth login
hf auth whoami

# 2. Tạo model repo
hf repos create traffic-sign-yolov8s-p2 --type model

# 3. Upload trọng số
hf upload <username>/traffic-sign-yolov8s-p2 best.pt best.pt --type model

# 4. Sửa HF_REPO_ID_MAC_DINH trong app/app.py cho khớp, rồi chạy thử
streamlit run app/app.py
```
