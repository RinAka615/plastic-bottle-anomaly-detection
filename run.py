import cv2 as cv
import numpy as np
from openvino import Core
import os

# ==========================================
# 1. CẤU HÌNH ĐƯỜNG DẪN
# ==========================================
MODEL_PATH = r"D:\HCMUTE\IAC\Game\Bandweight\model.xml"
TEST_DIR = "datasets/test"          # Thư mục chứa ảnh gốc
OUTPUT_DIR = "ket_qua_hinh_anh"     # Thư mục sẽ chứa ảnh bản đồ nhiệt xuất ra

# ==========================================
# 2. KHỞI TẠO MÔ HÌNH OPENVINO
# ==========================================
print("Đang tải mô hình OpenVINO...")
ie = Core()
model = ie.read_model(model=MODEL_PATH)
compiled_model = ie.compile_model(model=model, device_name="CPU")

# Tìm đúng cổng heatmap
heatmap_layer = None
for out in compiled_model.outputs:
    if any("anomaly_map" in name for name in out.names):
        heatmap_layer = out
        break
    if out.partial_shape.rank.is_static:
        if out.partial_shape.rank.get_length() >= 3:
            heatmap_layer = out
            break
if heatmap_layer is None:
    heatmap_layer = compiled_model.outputs[-1]

# Tọa độ cắt ảnh chuẩn
(x, y, w, h) = (494, 116, 347, 596)

# ==========================================
# 3. QUÉT DATASET VÀ LƯU ẢNH NGẦM
# ==========================================
print(f"\nBắt đầu quét và tạo ảnh bản đồ nhiệt vào thư mục: {OUTPUT_DIR}")
categories = ["Normal", "Unnormal"]

for category in categories:
    folder_path = os.path.join(TEST_DIR, category)
    if not os.path.exists(folder_path):
        print(f"[Cảnh báo] Không tìm thấy thư mục: {folder_path}")
        continue

    # Tạo thư mục con tương ứng ở đầu ra (ví dụ: ket_qua_hinh_anh/Normal)
    save_folder = os.path.join(OUTPUT_DIR, category)
    os.makedirs(save_folder, exist_ok=True)

    files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"-> Đang xử lý và xuất {len(files)} ảnh cho nhóm '{category}'...")

    for file in files:
        img_path = os.path.join(folder_path, file)
        
        # Đọc ảnh
        frame = cv.imread(img_path)
        if frame is None:
            continue
        
        # Cắt ảnh
        if frame.shape[0] > h and frame.shape[1] > w:
            cropped_frame = frame[y : y+h, x : x+w].copy() 
        else:
            cropped_frame = frame.copy()
        
        # Tiền xử lý cho AI
        input_img = cv.resize(cropped_frame, (256, 256))
        rgb_img = cv.cvtColor(input_img, cv.COLOR_BGR2RGB)
        input_tensor = np.expand_dims(rgb_img.transpose(2, 0, 1), axis=0).astype(np.float32) / 255.0
        
        # Dự đoán
        results = compiled_model([input_tensor])
        
        # Lấy bản đồ nhiệt
        anomaly_map = results[heatmap_layer]
        anomaly_map = np.squeeze(anomaly_map)
        max_score = float(np.max(anomaly_map))
        
        # --- VẼ BẢN ĐỒ NHIỆT ---
        a_map_norm = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-8)
        a_map_norm = (a_map_norm * 255).astype(np.uint8)
        a_map_norm = np.ascontiguousarray(a_map_norm)
        
        heatmap_colored = cv.applyColorMap(a_map_norm, cv.COLORMAP_JET)
        overlay = cv.addWeighted(input_img, 0.5, heatmap_colored, 0.5, 0)
        
        # Phóng to ảnh ra 512x512 cho dễ nhìn
        final_img = cv.resize(overlay, (512, 512))
        
        # Ghi điểm số trực tiếp lên góc bức ảnh xuất ra để dễ đánh giá
        cv.putText(final_img, f"Score: {max_score:.3f}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Lưu ảnh vào thư mục đầu ra
        save_path = os.path.join(save_folder, f"heatmap_{file}")
        cv.imwrite(save_path, final_img)

print("\n Hoàn thành! Toàn bộ ảnh đã được vẽ bản đồ nhiệt và lưu thành công.")