# -*- coding: utf-8 -*-
# YouTube(streamlink) + YOLO11s + "사람 키" 자가보정 격자(≈10 m²) + ROI + Global NMS + Auto-fit
# 사용: ROI 다각형 클릭(3점+) → Enter. 초기 수 초 "Calibrating..." 후 격자 표시.

import subprocess, shutil, math, time
import cv2
import numpy as np
from ultralytics import YOLO

# ========= 설정 =========
# YOUTUBE_URL = "https://www.youtube.com/live/EaRgJQ--2eE"
YOUTUBE_URL = "https://www.youtube.com/watch?v=u4UZ4UvZXrg"
MODEL   = "yolo11s.pt"
DEVICE  = "cpu"        # GPU면 "cuda:0"
IMGSZ   = 640
CONF    = 0.35
ROWS, COLS_TILES = 2, 4
USE_HALF = False

PERSON_HEIGHT_M = 1.7
TARGET_CELL_AREA_M2 = 10.0
CELL_SIDE_M = float(math.sqrt(TARGET_CELL_AREA_M2))
MIN_CELL_PX = 12
ALERT_THRESHOLD = 5
FILL_ALPHA = 0.45
DRAW_COUNTS = True

# 표시 유틸
DISPLAY_MAX_W = 1280
DISPLAY_MAX_H = 720
FONT = cv2.FONT_HERSHEY_SIMPLEX
def resize_to_fit(img, max_w=DISPLAY_MAX_W, max_h=DISPLAY_MAX_H):
    h, w = img.shape[:2]
    s = min(max_w / w, max_h / h, 1.0)
    return cv2.resize(img, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA) if s < 1.0 else img
def show_fit(win, img):
    cv2.namedWindow(win, cv2.WINDOW_NORMAL); cv2.imshow(win, resize_to_fit(img))
def compute_fit_scale(img_w, img_h, max_w=DISPLAY_MAX_W, max_h=DISPLAY_MAX_H):
    return min(max_w / img_w, max_h / img_h, 1.0)

# ========= 스트림 열기 =========
def resolve_stream(url: str) -> str:
    exe = (shutil.which("streamlink")
           or r"C:\Users\user\anaconda3\envs\py39\Scripts\streamlink.exe")
    if not exe:
        raise RuntimeError("streamlink 실행파일을 찾지 못했습니다.")
    out = subprocess.run([exe, "--stream-url", url, "best"],
                         capture_output=True, text=True, check=True)
    s = out.stdout.strip()
    if not s:
        raise RuntimeError("streamlink로 스트림 URL 얻기 실패")
    return s

# ========= 견고 회귀(허버) =========
def huber_weights(resid, delta):
    a = np.abs(resid)
    w = np.ones_like(a)
    mask = a > delta
    w[mask] = (delta / a[mask])
    return w

def robust_fit_h_eq_s_y_plus_c(y_vals, h_vals, iters=3):
    y = np.asarray(y_vals, np.float64)
    h = np.asarray(h_vals, np.float64)
    if len(y) < 8:
        return None
    A = np.vstack([y, np.ones_like(y)]).T
    s, c = np.linalg.lstsq(A, h, rcond=None)[0]
    for _ in range(iters):
        resid = h - (s*y + c)
        mad = np.median(np.abs(resid - np.median(resid))) + 1e-9
        delta = 1.5 * mad
        w = huber_weights(resid, delta)
        Aw = (A.T * w).T
        hw = h * w
        s, c = np.linalg.lstsq(Aw, hw, rcond=None)[0]
    if not np.isfinite(s) or not np.isfinite(c) or s <= 0:
        return None
    return float(s), float(c)

# ========= 캘리브레이터 =========
class HeightCalibrator:
    """h ≈ s*y + c.  지평선 y_h = -c/s,  px_per_meter(y) = (s / PERSON_HEIGHT_M) * (y - y_h)"""
    def __init__(self, img_h):
        self.samples_y = []; self.samples_h = []
        self.min_samples = 120; self.max_samples = 1200
        self.ready = False; self.s = None; self.c = None; self.y_h = None
        self.ema_alpha = 0.1; self.img_h = img_h

    def add(self, y_foot, h_px):
        if h_px <= 0: return
        self.samples_y.append(float(y_foot)); self.samples_h.append(float(h_px))
        if len(self.samples_y) > self.max_samples:
            self.samples_y = self.samples_y[-self.max_samples:]
            self.samples_h = self.samples_h[-self.max_samples:]

    def try_fit(self):
        if len(self.samples_y) < self.min_samples:
            self.ready = False; return False
        y = np.array(self.samples_y, np.float64)
        h = np.array(self.samples_h, np.float64)
        K = len(y); idx = np.argsort(h); lo = int(0.10*K); hi = int(0.90*K)
        sel = idx[lo:hi]
        fit = robust_fit_h_eq_s_y_plus_c(y[sel], h[sel], iters=3)
        if fit is None:
            self.ready = False; return False
        s_new, c_new = fit
        y_h_new = -c_new / (s_new + 1e-12)
        y_h_new = float(np.clip(y_h_new, -0.2*self.img_h, 0.6*self.img_h))  # 보호
        if self.s is None:
            self.s, self.c, self.y_h = s_new, c_new, y_h_new
        else:
            a = self.ema_alpha
            self.s = (1-a)*self.s + a*s_new
            self.c = (1-a)*self.c + a*c_new
            self.y_h = (1-a)*self.y_h + a*y_h_new
        self.ready = True; return True

    def px_per_meter(self, y):
        if not self.ready: return None
        return (self.s / PERSON_HEIGHT_M) * max(0.0, (y - self.y_h))

# ========= ROI =========
def click_roi(first):
    roi = []; SCALE = compute_fit_scale(first.shape[1], first.shape[0])
    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            roi.append([int(round(x / SCALE)), int(round(y / SCALE))])
    cv2.namedWindow("Draw ROI polygon", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Draw ROI polygon", on_mouse)
    while True:
        disp = first.copy()
        for i, (px, py) in enumerate(roi):
            cv2.circle(disp, (px, py), 4, (0,200,255), -1)
            if i > 0: cv2.line(disp, tuple(roi[i-1]), (px, py), (0,200,255), 2)
        if len(roi) >= 3: cv2.line(disp, tuple(roi[-1]), tuple(roi[0]), (0,200,255), 1)
        cv2.putText(disp, "ROI click(3+) → Enter (r=reset)", (20,20), FONT, 0.7, (255,255,255), 2)
        show_fit("Draw ROI polygon", disp)
        k = cv2.waitKey(30) & 0xFF
        if k == 27: cv2.destroyAllWindows(); raise SystemExit
        if k == ord('r'): roi = []
        if k == 13 and len(roi) >= 3: break
    cv2.destroyWindow("Draw ROI polygon")
    return np.array(roi, dtype=np.int32)

def polygon_mask(img_h, img_w, poly):
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    cv2.fillPoly(mask, [poly.astype(np.int32)], 255)
    return mask

# ========= D) 오목 ROI 스팬 안정화 =========
def _horizontal_spans_in_poly(poly, y, eps=1e-9):
    """y=const와 다각형 교차의 모든 내부 구간을 [(xL,xR), ...]로 반환"""
    pts = poly.reshape(-1, 2).astype(np.float64)
    xs = []
    N = len(pts)
    for i in range(N):
        x1, y1 = pts[i]; x2, y2 = pts[(i+1) % N]
        if (y1 <= y < y2) or (y2 <= y < y1):  # half-open: 중복 카운트 방지
            t = (y - y1) / (y2 - y1 + eps)
            xs.append(x1 + t * (x2 - x1))
    if len(xs) < 2: return []
    xs.sort()
    # 중복 제거
    dedup = [xs[0]]
    for xv in xs[1:]:
        if abs(xv - dedup[-1]) > 1e-6: dedup.append(xv)
    xs = dedup
    if len(xs) % 2 == 1: xs = xs[:-1]
    return [(float(xs[i]), float(xs[i+1])) for i in range(0, len(xs), 2)]

def _interval_iou(a, b):
    L1, R1 = a; L2, R2 = b
    inter = max(0.0, min(R1, R2) - max(L1, L2))
    if inter <= 0: return 0.0
    len1 = max(0.0, R1 - L1); len2 = max(0.0, R2 - L2)
    return inter / (len1 + len2 - inter + 1e-9)

def horizontal_span_in_poly_stable(poly, y, prev_span=None, mode="overlap"):
    spans = _horizontal_spans_in_poly(poly, y)
    if not spans: return None
    if mode == "longest" or prev_span is None:
        return max(spans, key=lambda s: s[1] - s[0])
    return max(spans, key=lambda s: _interval_iou(s, prev_span))

# ========= 그리드 =========
class PerspectiveGrid:
    def __init__(self, poly_px, img_h, img_w, calibrator: HeightCalibrator):
        self.poly = poly_px.astype(np.int32)
        self.h = img_h; self.w = img_w
        self.calib = calibrator
        self.built = False
        self.strips = []
        self.min_y = int(np.min(self.poly[:,1])); self.max_y = int(np.max(self.poly[:,1]))

    def build(self):
        if not self.calib.ready:
            self.built = False; self.strips = []; return False

        EDGE_EMA = 0.4      # 경계 약한 평활
        MAX_EDGE_JUMP = 40  # 경계 점프 제한(px)
        SHRINK_MAX = 0.92   # 한 줄 올라갈 때 가로폭 축소 하한

        prev_span_top = None
        prev_span_bot = None
        prev_Wt = prev_Wb = None
        strips = []

        y_bot = min(self.h-2, self.max_y - 1)
        y_top_limit = max(self.min_y + 1, int(self.calib.y_h + 4))
        STEP = 4

        def smooth_edge(curL, curR, prev):
            if prev is None: return curL, curR
            pL, pR = prev
            L = EDGE_EMA*curL + (1-EDGE_EMA)*pL
            R = EDGE_EMA*curR + (1-EDGE_EMA)*pR
            L = max(pL - MAX_EDGE_JUMP, min(pL + MAX_EDGE_JUMP, L))
            R = max(pR - MAX_EDGE_JUMP, min(pR + MAX_EDGE_JUMP, R))
            return L, R

        while True:
            span_b = horizontal_span_in_poly_stable(self.poly, y_bot, prev_span_bot, mode="overlap")
            if span_b is None:
                y_bot -= STEP
                if y_bot <= y_top_limit: break
                continue
            xLb, xRb = span_b
            if xRb - xLb < MIN_CELL_PX*2:
                y_bot -= STEP
                if y_bot <= y_top_limit: break
                continue

            pvm_bot = self.calib.px_per_meter(y_bot)
            if pvm_bot is None or pvm_bot <= 0: break
            Hpx = max(MIN_CELL_PX, pvm_bot * CELL_SIDE_M)

            # 가로폭 = 세로폭(정사각형 느낌). 과도 축소 방지.
            Wt = Wb = Hpx
            if prev_Wt is not None:
                Wt = max(Wt, prev_Wt * SHRINK_MAX)
                Wb = max(Wb, prev_Wb * SHRINK_MAX)

            y_top = int(round(y_bot - Hpx))
            if y_top <= y_top_limit: break

            span_t = horizontal_span_in_poly_stable(self.poly, y_top, prev_span_top, mode="overlap")
            if span_t is None:
                y_bot -= STEP
                if y_bot <= y_top_limit: break
                continue
            xLt, xRt = span_t

            if xRt - xLt < MIN_CELL_PX*2:
                y_bot -= STEP
                if y_bot <= y_top_limit: break
                continue

            # 경계 부드럽게
            xLt, xRt = smooth_edge(xLt, xRt, prev_span_top)
            xLb, xRb = smooth_edge(xLb, xRb, prev_span_bot)

            # 칼럼 수
            ncols_top = int((xRt - xLt) // Wt)
            ncols_bot = int((xRb - xLb) // Wb)
            ncols = max(0, min(ncols_top, ncols_bot))
            if ncols < 1:
                y_bot = y_top
                prev_span_top = span_t; prev_span_bot = span_b
                prev_Wt, prev_Wb = Wt, Wb
                continue

            # 각 셀(사다리꼴)
            polys = []
            for j in range(ncols):
                xt0 = xLt + j*Wt; xt1 = xLt + (j+1)*Wt
                xb0 = xLb + j*Wb; xb1 = xLb + (j+1)*Wb
                poly = np.array([[xt0, y_top],
                                 [xt1, y_top],
                                 [xb1, y_bot],
                                 [xb0, y_bot]], dtype=np.int32)
                polys.append(poly)

            strips.append(dict(
                y_top=int(y_top), y_bot=int(y_bot),
                xL_top=float(xLt), xR_top=float(xRt),
                xL_bot=float(xLb), xR_bot=float(xRb),
                W_top=float(Wt), W_bot=float(Wb),
                ncols=int(ncols), polys=polys
            ))

            # 다음 스트립을 위해 갱신
            y_bot = y_top
            prev_span_top = (xLt, xRt); prev_span_bot = (xLb, xRb)
            prev_Wt, prev_Wb = Wt, Wb
            if y_bot <= y_top_limit: break

        self.strips = strips
        self.built = len(self.strips) > 0
        return self.built

    def locate_cell(self, x, y):
        for i, S in enumerate(self.strips):
            if not (S["y_top"] < y <= S["y_bot"]): continue
            jt = (x - S["xL_top"]) / (S["W_top"] + 1e-9)
            jb = (x - S["xL_bot"]) / (S["W_bot"] + 1e-9)
            j = int(math.floor(0.5*(jt + jb)))
            if 0 <= j < S["ncols"]:
                if (x < min(S["xL_top"], S["xL_bot"])) or (x > max(S["xR_top"], S["xR_bot"])):
                    return (None, None)
                return (i, j)
        return (None, None)

# ========= 타일 추론 + 글로벌 NMS =========
def split_tiles(frame, rows=ROWS, cols=COLS_TILES):
    h, w = frame.shape[:2]
    hs = [int(round(r*h/rows)) for r in range(rows+1)]
    ws = [int(round(c*w/cols)) for c in range(cols+1)]
    crops, offs = [], []
    for r in range(rows):
        for c in range(cols):
            y1,y2 = hs[r], hs[r+1]; x1,x2 = ws[c], ws[c+1]
            crops.append(frame[y1:y2, x1:x2]); offs.append((x1, y1))
    return crops, offs

def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a[:4]; bx1, by1, bx2, by2 = b[:4]
    ix1=max(ax1,bx1); iy1=max(ay1,by1); ix2=min(ax2,bx2); iy2=min(ay2,by2)
    iw=max(0.0,ix2-ix1); ih=max(0.0,iy2-iy1); inter=iw*ih
    if inter<=0: return 0.0
    area_a=max(0.0,ax2-ax1)*max(0.0,ay2-ay1)
    area_b=max(0.0,bx2-bx1)*max(0.0,by2-by1)
    return inter/(area_a+area_b-inter+1e-9)

def nms_global(boxes, iou_th=0.55):
    boxes=sorted(boxes, key=lambda x:x[4], reverse=True)
    keep=[]
    while boxes:
        cur=boxes.pop(0); keep.append(cur)
        boxes=[b for b in boxes if iou_xyxy(cur,b) < iou_th]
    return keep

# ========= 메인 =========
def main():
    stream_url = resolve_stream(YOUTUBE_URL)
    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened(): raise RuntimeError("VideoCapture 열기 실패")

    ok, first = cap.read()
    if not ok: raise RuntimeError("첫 프레임 읽기 실패")
    fr_h, fr_w = first.shape[:2]

    roi_poly = click_roi(first)
    roi_mask = polygon_mask(fr_h, fr_w, roi_poly)

    calib = HeightCalibrator(img_h=fr_h)
    grid  = PerspectiveGrid(roi_poly, fr_h, fr_w, calib)

    model = YOLO(MODEL)

    last_grid_build = 0
    REBUILD_SEC = 2.0
    FREEZE_AFTER_SEC = 20.0
    first_built_ts = None

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok: break

        # 타일 추론(사람만)
        crops, offs = split_tiles(frame, ROWS, COLS_TILES)
        results = model(crops, conf=CONF, imgsz=IMGSZ, device=DEVICE, half=USE_HALF, classes=[0])

        # 타일→원본 좌표 변환 + 글로벌 NMS
        all_boxes=[]
        for res,(ox,oy) in zip(results, offs):
            boxes=res.boxes
            if boxes is None or len(boxes)==0: continue
            xyxy = boxes.xyxy.cpu().numpy(); confs=boxes.conf.cpu().numpy()
            for (x1,y1,x2,y2),cf in zip(xyxy,confs):
                all_boxes.append([x1+ox, y1+oy, x2+ox, y2+oy, float(cf)])
        kept=nms_global(all_boxes, iou_th=0.55)

        # 캘리브레이션 샘플(전신+ROI 내부 발끝)
        for x1,y1,x2,y2,cf in kept:
            if cf < 0.5: continue
            w = x2-x1; h = y2-y1
            if h <= 0 or w <= 0: continue
            aspect = h / (w + 1e-9)
            if aspect < 1.6 or h < 40: continue
            footx = 0.5*(x1+x2); footy = y2
            if roi_mask[int(min(max(footy,0),fr_h-1)), int(min(max(footx,0),fr_w-1))] == 0:
                continue
            calib.add(footy, h)

        # 캘리브레이션 & 그리드 재빌드/프리즈
        calib.try_fit()
        now = time.time()
        if calib.ready:
            need_rebuild = (not grid.built) or (now - last_grid_build > REBUILD_SEC)
            if need_rebuild:
                grid.build()
                last_grid_build = now
                if first_built_ts is None and grid.built:
                    first_built_ts = now
        if first_built_ts is not None and (now - first_built_ts) > FREEZE_AFTER_SEC:
            REBUILD_SEC = 1e9  # 사실상 재빌드 금지

        # 카운트
        counts = [np.zeros(S["ncols"], dtype=np.int32) for S in grid.strips] if grid.built else []
        for x1,y1,x2,y2,cf in kept:
            footx = 0.5*(x1+x2); footy = y2
            if roi_mask[int(min(max(footy,0),fr_h-1)), int(min(max(footx,0),fr_w-1))] == 0:
                continue
            if grid.built:
                si, sj = grid.locate_cell(footx, footy)
                if si is not None: counts[si][sj] += 1

        # 오버레이
        overlay = frame.copy()
        if grid.built:
            for i, S in enumerate(grid.strips):
                for j in range(S["ncols"]):
                    poly = S["polys"][j]
                    n = int(counts[i][j])
                    color = (180,180,180) if n==0 else ((0,0,255) if n>=ALERT_THRESHOLD else (0,180,0))
                    # 너무 작은 셀은 표시만 생략
                    edges = [
                        np.linalg.norm(poly[1]-poly[0]),
                        np.linalg.norm(poly[2]-poly[1]),
                        np.linalg.norm(poly[3]-poly[2]),
                        np.linalg.norm(poly[0]-poly[3]),
                    ]
                    if min(edges) >= MIN_CELL_PX:
                        cv2.fillPoly(overlay, [poly], color)
                        cv2.polylines(overlay, [poly], True, (50,50,50), 1)
                        if DRAW_COUNTS and n>0:
                            cx = int(np.mean(poly[:,0])); cy = int(np.mean(poly[:,1]))
                            cv2.putText(overlay, str(n), (cx-6, cy+6), FONT, 0.6, (0,0,0), 2)
                            cv2.putText(overlay, str(n), (cx-6, cy+6), FONT, 0.6, (255,255,255), 1)

        overlay_masked = cv2.bitwise_and(overlay, overlay, mask=polygon_mask(fr_h, fr_w, roi_poly))
        out = cv2.addWeighted(overlay_masked, FILL_ALPHA, frame, 1.0 - FILL_ALPHA, 0)

        # ROI 윤곽/디버그
        cv2.polylines(out, [roi_poly], True, (0,200,255), 2)
        if calib.ready:
            yh = int(round(max(0, min(fr_h-1, calib.y_h))))
            cv2.line(out, (0, yh), (fr_w-1, yh), (255,255,0), 1)
            cv2.putText(out, f"s={calib.s:.3f}  y_h={calib.y_h:.1f}", (12,52), FONT, 0.7, (0,0,0), 3)
            cv2.putText(out, f"s={calib.s:.3f}  y_h={calib.y_h:.1f}", (12,52), FONT, 0.7, (255,255,255), 1)
        else:
            cv2.putText(out, f"Calibrating... {len(calib.samples_y)}/{calib.min_samples}",
                        (12,52), FONT, 0.7, (0,0,0), 3)
            cv2.putText(out, f"Calibrating... {len(calib.samples_y)}/{calib.min_samples}",
                        (12,52), FONT, 0.7, (255,255,255), 1)

        cv2.putText(out, f"~{TARGET_CELL_AREA_M2:.0f} m^2 cells (person-based), Alert >= {ALERT_THRESHOLD}",
                    (12,28), FONT, 0.7, (0,0,0), 3)
        cv2.putText(out, f"~{TARGET_CELL_AREA_M2:.0f} m^2 cells (person-based), Alert >= {ALERT_THRESHOLD}",
                    (12,28), FONT, 0.7, (255,255,255), 1)

        show_fit("Person-calibrated Grid Crowd Count", out)
        if cv2.waitKey(1) & 0xFF in (27, ord('q')): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
