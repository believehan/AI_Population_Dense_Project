from flask import Flask, render_template, Response, jsonify
import cv2
import os
import threading
import torch
import numpy as np
from ultralytics import YOLO
import copy; # roi변경을 위해 필요한 모튤

app = Flask(__name__)

zone_coords = {
    # "1": [(16,19), (2578,15), (2564,1632), (13,1640) ]
    "1": [(2782,807), (3080,783), (3140,911), (2829,924)],
    "2": [(1550,878), (1791,876), (1836,951), (1572,955) ],
    "3": [(1593,1033),(1894,1020),(1929,1103),(1643,1107)],
    "4": [(869,1249),(1140,1161),(1337,1247),(1036,1389)]
}

# # --- YOLO 모델 (사람 탐지 전용) ---
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# print(f"🔍 Using device: {device}")
# model = YOLO("/workspace/wdtc/project/flask2/epoch40_4.engine").to(device)

# --- YOLO 모델 (사람 탐지 전용) ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"🔍 Using device: {device}")
# .engine은 .to(device) 불가 → predict에서 device 지정
DEVICE_ARG = 0 if device == 'cuda' else 'cpu'
# model = YOLO("/workspace/wdtc/project/flask2/yolov8n.pt", task="detect")
# model = YOLO("/workspace/wdtc/project/code/yolo11s.engine", task="detect")
model = YOLO("epoch40.pt", task="detect")

def infer(img, **kwargs):
    # 내보낸 모델은 predict만 지원하고, 여기서 device를 넘겨야 함
    return model.predict(img, device=DEVICE_ARG, **kwargs)

# --- 영상 경로 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# video_path = os.path.join(BASE_DIR, "static", "videos", "E05_020.mp4")
video_path = os.path.join(BASE_DIR, "static", "videos", "sample3.mp4")

# --- 전역 변수: 구역별 객체 탐지 결과 ---
zone_detections = {1: 0, 2: 0, 3: 0, 4: 0}
detection_lock = threading.Lock()

# ─────────────────────────────────────────────────────────────
# (ROI 좌표 변경을 위한 변수
# ─────────────────────────────────────────────────────────────
_zones_now  = copy.deepcopy(zone_coords)   # 현재 ROI 좌표 (수동 설정으로 갱신)
_frame_wh   = [None, None]               # [W, H] 첫 프레임 읽을 때 기록
#─────────────────────────────────────────────────────────────

# --- 경보 임계값 ---
WARNING_THRESHOLD = 5
DANGER_THRESHOLD = 7

def count_objects_in_zone(roi_frame):
    """ROI 내부 사람 수 정확히 카운트"""
    # results = model(roi_frame, classes=[0], conf=0.25)
    results = infer(roi_frame, classes=[0], conf=0.25)
    if not results[0].boxes or len(results[0].boxes) == 0:
        return 0
    return len(results[0].boxes)  # ROI 안 사람 수 그대로 반환

def generate_frames(camera_id):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"영상 파일을 열 수 없습니다: {video_path}")
    
    frame_skip = 5  # 프레임 설정
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        # MAIN 화면
        if str(camera_id) == 'main':
            img = frame.copy()
            for zone_id_str, coords in zone_coords.items():
                zone_id = int(zone_id_str)
                xs = [p[0] for p in coords]
                ys = [p[1] for p in coords]
                x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
                
                pts = np.array(coords, np.int32)
                pts = pts.reshape((-1,1,2))  # OpenCV 다각형 형식

                # 녹색(0,255,0), 두께 2로 폴리곤 그리기
                cv2.polylines(frame, [pts], isClosed=True, color=(0,255,0), thickness=2)
                
                roi_frame = frame[y1:y2, x1:x2]

                with detection_lock:
                    zone_detections[zone_id] = count_objects_in_zone(roi_frame)

                # roi_results = model(roi_frame, classes=[0], conf=0.25)
                roi_results = infer(roi_frame, classes=[0], conf=0.25)
                try:
                    roi_img = roi_results[0].plot()
                except:
                    roi_img = roi_frame
                img[y1:y2, x1:x2] = roi_img

        # Zone만 스트리밍
        else:
            camera_num = str(camera_id)
            if camera_num in zone_coords:
                coords = zone_coords[camera_num]
                xs = [p[0] for p in coords]
                ys = [p[1] for p in coords]
                x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
              
                # 1) 전체 사각형 크롭
                cropped_frame = frame[y1:y2, x1:x2].copy()

                # 2) 다각형 좌표를 cropped_frame 기준으로 변환
                poly_pts = np.array([(x - x1, y - y1) for x, y in coords], np.int32)
                poly_pts = poly_pts.reshape((-1, 1, 2))

                # 3) 크롭 이미지 크기와 같은 크기의 검정 마스크 생성
                mask = np.zeros(cropped_frame.shape[:2], dtype=np.uint8)

                # 4) 다각형 영역만 흰색으로 채우기
                cv2.fillPoly(mask, [poly_pts], 255)

                # 5) 마스크 적용해서 다각형 영역만 추출
                masked_img = cv2.bitwise_and(cropped_frame, cropped_frame, mask=mask)

                with detection_lock:
                    zone_detections[int(camera_num)] = count_objects_in_zone(masked_img)

                # results = model(masked_img, classes=[0], conf=0.25)
                results = infer(masked_img, classes=[0], conf=0.25)
                try:
                    img = results[0].plot()
                except:
                    img = masked_img
            else:
                img = frame.copy()

        _, buffer = cv2.imencode('.jpg', img)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

# ------------------ Flask Routes ------------------
@app.route('/')
def index():
    return render_template('index.html', zone_coords=zone_coords)

@app.route('/camera/<camera_id>')
def camera_feed(camera_id):
    print(f"카메라 요청: {camera_id} (타입: {type(camera_id)})")
    return Response(generate_frames(camera_id),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/zone_detections')
def get_zone_detections():
    with detection_lock:
        return jsonify({
            'zones': zone_detections,
            'total_detections': sum(zone_detections.values()),
            'danger_zones': [zone_id for zone_id, count in zone_detections.items() if count >= DANGER_THRESHOLD],
            'thresholds': {
                'warning': WARNING_THRESHOLD,
                'danger': DANGER_THRESHOLD,
            }
        })

@app.route('/api/detection_stats')
def detection_stats():
    with detection_lock:
        return jsonify({
            'total_detections': sum(zone_detections.values()),
            'active_cameras': 5,  # MAIN + 4개 카메라
            'zone_detections': zone_detections,
            'thresholds': {
                'warning': WARNING_THRESHOLD,
                'danger': DANGER_THRESHOLD,
            }
        })

@app.route('/api/alerts')
def get_alerts():
    with detection_lock:
        alerts = []
        for zone_id, count in zone_detections.items():
            if count >= DANGER_THRESHOLD:
                alerts.append(f"🚨 ZONE {zone_id}: {count}명 탐지 - 위험 수준!")
            elif count >= WARNING_THRESHOLD:
                alerts.append(f"⚡ ZONE {zone_id}: {count}명 탐지 - 주의 필요")
        
        if not alerts:
            alerts.append("✅ 모든 구역 정상 상태")
            
        return jsonify({'alerts': alerts})
    
    
# ─────────────────────────────────────────────────────────────
# 6) 좌표 찍기 모듈(블루프린트) 등록
#     - /roi : 좌표 찍는 전용 페이지 (팝업/새탭으로 열어 사용)
#     - /api/frame_size, /api/roi, /api/roi/<zid>, /api/roi/reset
#       는 블루프린트 내부에서 제공
# ─────────────────────────────────────────────────────────────
from roi_picker import create_roi_blueprint  # 별도 파일(roi_picker.py)에 있음

def _get_frame_size():
    return (_frame_wh[0], _frame_wh[1])

def _get_all_zones():
    with detection_lock:
        return dict(_zones_now)

def _set_zone_bbox(zid, bbox):
    with detection_lock:
        if zid in _zones_now:
            _zones_now[zid] = tuple(int(v) for v in bbox)

def _reset_all():
    with detection_lock:
        for k, v in zone_coords.items():
            _zones_now[k] = v

roi_bp = create_roi_blueprint(
    get_frame_size=_get_frame_size,
    get_all_zones=_get_all_zones,
    set_zone_bbox=_set_zone_bbox,
    reset_all=_reset_all
)
app.register_blueprint(roi_bp)


# ------------------ 앱 실행 ------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)