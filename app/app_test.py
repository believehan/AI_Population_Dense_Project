from flask import Flask, render_template, Response, request, jsonify, abort
import cv2, os, math, threading
import numpy as np
from ultralytics import YOLO

app = Flask(__name__)

# =========================
# 0) 설정
# =========================
# 이벤트 소스: "people" (기본) | "custom"
EVENT_MODE = "people"   # ← 특별 이벤트로 바꿀 땐 "custom"

# 초기 ROI(1개만 사용) — 현재는 크기 고정, 위치만 이동
ZONE_INIT = (866, 326, 990, 439)

MODEL_WEIGHTS = "yolo11s.pt"
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
VIDEO_PATH = os.path.join(BASE_DIR, "static", "videos", "sample1.mp4")

model = YOLO(MODEL_WEIGHTS)

# 커스텀 이벤트 포인트 공유(스레드 세이프)
_EVENT_POINTS = []
_EVENT_LOCK = threading.Lock()

# =========================
# 1) 공통 유틸
# =========================
def clamp_roi(x1, y1, x2, y2, W, H):
    w = x2 - x1; h = y2 - y1
    x1 = max(0, min(W - 1, x1)); y1 = max(0, min(H - 1, y1))
    x2 = x1 + w; y2 = y1 + h
    if x2 > W - 1: x1 -= (x2 - (W - 1)); x2 = W - 1
    if y2 > H - 1: y1 -= (y2 - (H - 1)); y2 = H - 1
    x1 = max(0, x1); y1 = max(0, y1)
    return int(x1), int(y1), int(x2), int(y2)

def draw_zone(img, rect, state=None, conf=None):
    x1,y1,x2,y2 = rect
    cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,255), 2)
    label = "ZONE"
    if state is not None and conf is not None:
        label += f"  {state}  conf:{conf:.3f}"
    cv2.putText(img, label, (x1, max(0, y1-8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
    return img

# =========================
# 2) 이벤트 소스 (모듈화 포인트)
# =========================
def event_points_people(results):
    """Ultralytics Results → 사람(0)의 '발끝' 픽셀"""
    pts = []
    if results is None or results.boxes is None or results.boxes.xyxy is None:
        return pts
    xyxy = results.boxes.xyxy
    cls  = results.boxes.cls
    try:
        xyxy = xyxy.cpu().numpy()
        cls  = (cls.cpu().numpy().astype(int) if cls is not None else np.zeros(len(xyxy), dtype=int))
    except Exception:
        xyxy = xyxy.numpy()
        cls  = (cls.numpy().astype(int) if cls is not None else np.zeros(len(xyxy), dtype=int))
    for i, (x1, y1, x2, y2) in enumerate(xyxy):
        if cls[i] != 0:  # person만
            continue
        xf = int(round((x1 + x2) * 0.5))
        yf = int(round(y2))
        pts.append((xf, yf))
    return pts

def event_points_custom():
    """외부에서 POST로 들어온 커스텀 이벤트 포인트 사용"""
    with _EVENT_LOCK:
        return list(_EVENT_POINTS)  # (x,y) 리스트

# =========================
# 3) SingleZoneFollower (전역→국소, 떨림 억제)
# =========================
class SingleZoneFollower:
    """
    - ACQ(전역): 크게 탐색 → TRK(국소): 데드밴드/EMA/클램프로 부드럽게
    - 입력: 임의의 '이벤트 포인트' 리스트 [(x,y), ...]
    """
    def __init__(self, init_rect, frame_size,
                 scale=0.5, decay=0.92, point_radius=6, blur_ksize=7,
                 warmup_frames=90, R_global_ratio=0.5, R_local=80,
                 stride_global=8, stride_local=3,
                 enter_p=0.65, exit_ratio=0.7, ema_alpha=0.2,
                 deadband_px=8, max_step_local=8, pos_ema_local=0.85):
        self.W, self.H = frame_size
        self.rect = init_rect
        self.prev_c = ((init_rect[0]+init_rect[2])//2, (init_rect[1]+init_rect[3])//2)

        # 상태
        self.state = "ACQ"   # ACQ | TRK
        self.conf_ema = 0.0
        self.frame_idx = 0

        # 히트맵
        self.scale = float(scale)
        self.sW = max(8, int(round(self.W * self.scale)))
        self.sH = max(8, int(round(self.H * self.scale)))
        self.heat = np.zeros((self.sH, self.sW), np.float32)
        self.decay = float(decay)
        self.point_radius = int(point_radius)
        self.blur_ksize = int(blur_ksize)

        # 탐색/전이 파라미터
        self.warmup = int(warmup_frames)
        self.Rg = int(min(self.W, self.H) * float(R_global_ratio))
        self.Rl = int(R_local)
        self.sg = int(stride_global)
        self.sl = int(stride_local)
        self.enter_p = float(enter_p)
        self.exit_ratio = float(exit_ratio)
        self.ema_alpha = float(ema_alpha)

        # 떨림 억제
        self.deadband = int(deadband_px)
        self.max_step = int(max_step_local)
        self.pos_ema = float(pos_ema_local)

        # 적응 임계
        self.last_scores = []

    # ---- 내부 ----
    def _to_scaled(self, x, y):
        return int(round(x * self.scale)), int(round(y * self.scale))

    def _integral(self, m):  # 적분영상
        return cv2.integral(m)

    def _sum_win(self, ii, x, y, w, h):
        x2, y2 = x+w, y+h
        return ii[y2, x2] - ii[y, x2] - ii[y2, x] + ii[y, x]

    # ---- 핵심 ----
    def update_heat(self, points):
        self.heat *= self.decay
        if not points: return
        canvas = np.zeros_like(self.heat)
        pr = max(2, int(round(self.point_radius * self.scale)))
        for (x,y) in points:
            xs, ys = self._to_scaled(x,y)
            if 0 <= xs < self.sW and 0 <= ys < self.sH:
                cv2.circle(canvas, (xs,ys), pr, 1.0, -1)
        if self.blur_ksize >= 3:
            k = self.blur_ksize | 1
            canvas = cv2.GaussianBlur(canvas, (k,k), 0)
        self.heat += canvas

    def _search(self, rect, mode):
        x1,y1,x2,y2 = rect; w = x2-x1; h = y2-y1
        if mode == "ACQ" or self.frame_idx < self.warmup:
            xmin, ymin = 0, 0
            xmax, ymax = self.W - w, self.H - h
            step = self.sg
        else:
            cx, cy = (x1+x2)//2, (y1+y2)//2
            R = self.Rl
            xmin = max(0, cx - R - w//2); ymin = max(0, cy - R - h//2)
            xmax = min(self.W - w, cx + R - w//2); ymax = min(self.H - h, cy + R - h//2)
            step = self.sl

        xs_min, ys_min = self._to_scaled(xmin, ymin)
        xs_max, ys_max = self._to_scaled(xmax, ymax)
        sw, sh = max(1,int(round(w*self.scale))), max(1,int(round(h*self.scale)))

        ii = self._integral(self.heat)
        best, best_sum = None, -1.0
        for ys in range(ys_min, ys_max+1, max(1,int(round(step*self.scale)))):
            for xs in range(xs_min, xs_max+1, max(1,int(round(step*self.scale)))):
                if xs+sw >= self.sW or ys+sh >= self.sH:
                    continue
                s = self._sum_win(ii, xs, ys, sw, sh)
                if s > best_sum:
                    best_sum = s; best = (xs, ys)

        if best is None:
            return rect, 0.0, 0.0

        bx, by = best
        nx1 = int(round(bx / self.scale)); ny1 = int(round(by / self.scale))
        nx2 = nx1 + w; ny2 = ny1 + h
        area = float(sw*sh)
        conf = (best_sum / max(area, 1.0))
        return (nx1,ny1,nx2,ny2), best_sum, conf

    def step(self, points):
        self.frame_idx += 1
        self.update_heat(points)

        # 적응 임계
        if self.last_scores:
            arr = np.array(self.last_scores, dtype=np.float32)
            tau_enter = float(np.quantile(arr, self.enter_p))
            tau_exit  = tau_enter * self.exit_ratio
        else:
            tau_enter = tau_exit = 0.0

        # 탐색
        prop_rect, score, conf = self._search(self.rect, self.state)
        cx, cy = (prop_rect[0]+prop_rect[2])//2, (prop_rect[1]+prop_rect[3])//2
        px, py = self.prev_c
        spd = math.hypot(cx-px, cy-py)
        self.conf_ema = (1-self.ema_alpha)*self.conf_ema + self.ema_alpha*conf

        # 상태 전이
        if self.state == "ACQ":
            if (score >= tau_enter) and (spd <= 6.0) and (self.frame_idx >= self.warmup):
                self.state = "TRK"
        else:
            if (score < tau_exit):
                self.state = "ACQ"

        # TRK 떨림 억제
        if self.state == "TRK":
            if spd < self.deadband:
                cx, cy = px, py
            else:
                dx, dy = cx - px, cy - py
                mag = math.hypot(dx, dy)
                if mag > self.max_step and mag > 0:
                    sc = self.max_step / mag
                    cx = int(round(px + dx*sc)); cy = int(round(py + dy*sc))
                cx = int(round(self.pos_ema*px + (1-self.pos_ema)*cx))
                cy = int(round(self.pos_ema*py + (1-self.pos_ema)*cy))
            # 크기 고정 복원
            w = self.rect[2]-self.rect[0]; h = self.rect[3]-self.rect[1]
            prop_rect = (cx - w//2, cy - h//2, cx + (w - w//2), cy + (h - h//2))

        # 경계 클리핑 & 커밋
        x1,y1,x2,y2 = clamp_roi(*prop_rect, self.W, self.H)
        self.rect = (x1,y1,x2,y2)
        self.prev_c = ((x1+x2)//2, (y1+y2)//2)

        # 스코어 버퍼
        self.last_scores.append(score)
        if len(self.last_scores) > 120:
            self.last_scores = self.last_scores[-120:]

        return self.rect, self.state, float(self.conf_ema)

# =========================
# 4) 스트리밍
# =========================
def generate_frames(camera_id):
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"영상 파일을 열 수 없습니다: {VIDEO_PATH}")

    follower = None
    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0); continue

        H, W = frame.shape[:2]
        if follower is None:
            follower = SingleZoneFollower(ZONE_INIT, (W, H))

        # --- 이벤트 포인트 취득 ---
        pts = []
        r = None
        if EVENT_MODE == "people":
            try:
                res = model(frame, classes=[0], conf=0.25)  # 공식 predict API
                r = res[0]
                pts = event_points_people(r)
            except Exception:
                pts = []
        else:  # "custom"
            pts = event_points_custom()

        # --- ROI 업데이트 ---
        rect, state, conf = follower.step(pts)

        # --- 출력 ---
        is_main = (camera_id == 'main' or str(camera_id) == 'main')
        if is_main:
            img = r.plot() if r is not None else frame.copy()
            img = draw_zone(img, rect, state, conf)
        else:
            # 하나뿐이지만 /camera/1 로 크롭을 열 수 있게 유지
            try:
                cam_num = int(camera_id)
                if cam_num != 1: abort(404)
            except Exception:
                abort(400)
            x1,y1,x2,y2 = rect
            roi = frame[y1:y2, x1:x2]
            if EVENT_MODE == "people":
                try:
                    rr = model(roi, classes=[0], conf=0.25)[0]
                    img = rr.plot()
                except Exception:
                    img = roi.copy()
            else:
                img = roi.copy()

        ok, buf = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if ok:
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')

    cap.release()

# =========================
# 5) 라우트
# =========================
@app.route('/')
def index2():
    return render_template('index2.html')

@app.route('/camera/<camera_id>')
def camera_feed(camera_id):
    print(f"카메라 요청: {camera_id}")
    return Response(generate_frames(camera_id),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# 커스텀 이벤트 포인트 주입 (EVENT_MODE="custom" 일 때 사용)
@app.route('/api/event_points', methods=['POST'])
def push_event_points():
    data = request.get_json(silent=True) or {}
    pts = data.get("points", [])
    ok = []
    for p in pts:
        if isinstance(p, (list, tuple)) and len(p) == 2:
            try:
                x = int(p[0]); y = int(p[1])
                ok.append((x, y))
            except Exception:
                pass
    with _EVENT_LOCK:
        _EVENT_POINTS.clear()
        _EVENT_POINTS.extend(ok)
    return jsonify({"received": len(ok)})

if __name__ == '__main__':
    # 리로더 비활성: 스트리밍 안정화
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
