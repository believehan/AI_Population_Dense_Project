import streamlit as st
import subprocess
import cv2
import numpy as np
from ultralytics import YOLO, RTDETR

st.set_page_config(layout="wide")
st.title("YOLO 실시간 CCTV ROI 탐지 (4개 Zone)")

# --- 컬럼 설정: 전체 화면, ROI 4개, 상태 ---
col_full, col_rois, col_status = st.columns([4,4,2])
frame_ph = col_full.empty()
roi_ph = col_rois.empty()
status_ph = col_status.empty()

# --- 모델 로드 (사람만 탐지) ---
zones = ["Zone-1","Zone-2","Zone-3","Zone-4"]
models = {zone: YOLO("yolo11l.pt") for zone in zones}

# --- ROI 좌표 (고정값) ---
zone_coords = {
    "Zone-1": (866, 326, 990, 439),
    "Zone-2": (1053, 242, 1184, 361),
    "Zone-3": (5, 658, 374, 1007),
    "Zone-4": (1065, 380, 1297, 583)
}

# --- 비디오 열기 ---
url = "https://www.youtube.com/live/rnXIjl_Rzy4?feature=shared"

result = subprocess.run(["streamlink","--stream-url",url,"best"], capture_output=True, text=True)
stream_url = result.stdout.strip()
cap = cv2.VideoCapture(stream_url)

frame_skip = 3
frame_count = 0

# --- 상태 텍스트 정의 ---
def get_status_text_and_color(count):
    if count >= 5: return "Danger", (0,0,255)
    elif count == 4: return "Warning", (0,165,255)
    elif count == 3: return "Caution", (0,255,255)
    elif count == 2: return "Normal", (0,255,0)
    else: return "Safe", (255,255,255)

# --- ROI 처리 ---
def process_block(model, block, size=(256,256)):
    if block.size==0:
        return np.zeros((size[1], size[0],3), dtype=np.uint8), 0
    block_resized = cv2.resize(block, (640,640))
    results = model(block_resized, classes=[0], conf=0.25)  # 사람만 탐지
    img = results[0].plot()
    if img.dtype != np.uint8: img = (255*np.clip(img,0,1)).astype(np.uint8)
    if img.shape[2]==4: img = img[:,:,:3]
    img = cv2.resize(img, size)
    count = len(results[0].boxes)
    return img, count

# --- 상태 표시 및 색상 오버레이 ---
def draw_status(img, zone, count):
    status, color = get_status_text_and_color(count)
    if status=="Danger":
        overlay = img.copy()
        overlay[:] = (0,0,255)
        img = cv2.addWeighted(overlay, 0.3, img, 0.7, 0)
    cv2.putText(img, f"{zone}: {count} ({status})", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    return img, status

# --- 메인 루프 ---
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame_count += 1
    if frame_count % frame_skip != 0:
        continue

    # --- 전체 화면 표시 + ROI 박스 ---
    frame_with_boxes = frame.copy()
    for zone, (x1,y1,x2,y2) in zone_coords.items():
        cv2.rectangle(frame_with_boxes, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(frame_with_boxes, zone, (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    frame_ph.image(frame_with_boxes, channels="BGR", caption="전체 화면 (ROI 표시)")

    roi_imgs = []
    danger, warning = [], []

    # --- 각 ROI 탐지 ---
    for zone in zones:
        x1, y1, x2, y2 = zone_coords[zone]
        block = frame[y1:y2, x1:x2]
        img, count = process_block(models[zone], block)
        img, status = draw_status(img, zone, count)
        roi_imgs.append(img)
        if status=="Danger": danger.append(zone)
        elif status=="Warning": warning.append(zone)

    # --- ROI 4개를 세로로 합치기 ---
    roi_column = np.vstack(roi_imgs)
    roi_ph.image(roi_column, channels="BGR", caption="4개 Zone 탐지 결과 (세로)")

    # --- 상태 메시지 ---
    md = "## 상태 알림\n"
    if danger:
        md += "### 🚨 Danger\n- " + "\n- ".join(danger) + "\n"
    if warning:
        md += "### ⚠️ Warning\n- " + "\n- ".join(warning) + "\n"
    if not danger and not warning:
        md += "### ✅ 경보 없음"
    status_ph.markdown(md)

cap.release()
cv2.destroyAllWindows()
