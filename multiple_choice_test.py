# -*- coding: utf-8 -*-
# YouTube(streamlink) + YOLO11s
# 5지선다(사람크기 프리셋) + "처음 1회" 고정 격자 + 감마(원근) 분할 그리드
# 키: [1..5]=프리셋, +/-=크기 미세조정, [ / ]=원근(γ), R=다시측정, Q/ESC=종료

import subprocess, shutil, time, math
import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO

# ========= 기본 설정 =========
# YOUTUBE_URL = "https://www.youtube.com/live/rnXIjl_Rzy4?feature=shared"
YOUTUBE_URL = "https://www.youtube.com/live/EaRgJQ--2eE"
# YOUTUBE_URL = "https://www.youtube.com/watch?v=u4UZ4UvZXrg"
MODEL   = "yolo11s.pt"
DEVICE  = "cpu"       # GPU면 "cuda:0"
IMGSZ   = 640
CONF    = 0.35
USE_HALF= False
ROWS_TILES, COLS_TILES = 2, 4

# 프리셋(아주작음~아주큼)
PRESET_NAMES   = ["아주작음", "작음", "보통", "큼", "아주큼"]
PRESET_FACTORS = [0.75, 0.90, 1.00, 1.15, 1.35]  # 셀 기준크기 배율

# 사람 샘플 수집
MID_BAND = (0.40, 0.60)        # 프레임 높이 기준
MIN_H_PX_FOR_SAMPLE = 36
ASPECT_MIN = 1.35
MIN_SAMPLES_FOR_BUILD = 25
MEASURE_TIMEOUT_SEC   = 3.0    # 이 시간 지나도 샘플 부족하면 기본값으로 1회 생성

# 그리드/표시
BASE_K_FROM_PERSON_H = 0.90    # 셀 기준 한 변(px) ≈ k * (중간밴드 사람 높이 px)
MIN_CELL_PX = 12
ALERT_THRESHOLD = 5
FILL_ALPHA = 0.45
DRAW_COUNTS = True

# 원근(감마) 분할
N_ROWS_INIT   = 14            # 행 개수
GAMMA_INIT    = 1.35          # 1.0=균등, ↑클수록 아래(가까이) 크게/위 작게
GAMMA_STEP    = 0.05
SCALE_STEP    = 1.05

# 표시
DISPLAY_MAX_W = 1280
DISPLAY_MAX_H = 720
FONT = cv2.FONT_HERSHEY_SIMPLEX

def resize_to_fit(img, max_w=DISPLAY_MAX_W, max_h=DISPLAY_MAX_H):
    h, w = img.shape[:2]
    s = min(max_w / w, max_h / h, 1.0)
    return cv2.resize(img, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA) if s < 1.0 else img

def show_fit(name, img):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, resize_to_fit(img))

def compute_fit_scale(w, h, max_w=DISPLAY_MAX_W, max_h=DISPLAY_MAX_H):
    return min(max_w / w, max_h / h, 1.0)

# ========= streamlink =========
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

# ========= ROI 입력 =========
def click_roi(first):
    roi = []
    SCALE = compute_fit_scale(first.shape[1], first.shape[0])
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
        if len(roi) >= 3:
            cv2.line(disp, tuple(roi[-1]), tuple(roi[0]), (0,200,255), 1)
        cv2.putText(disp, "ROI click(3+) → Enter (r=reset)", (20,20), FONT, 0.7, (255,255,255), 2)
        show_fit("Draw ROI polygon", disp)
        k = cv2.waitKey(30) & 0xFF
        if k == 27: cv2.destroyAllWindows(); raise SystemExit
        if k == ord('r'): roi = []
        if k == 13 and len(roi) >= 3: break
    cv2.destroyWindow("Draw ROI polygon")
    return np.array(roi, dtype=np.int32)

def polygon_mask(h, w, poly):
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [poly.astype(np.int32)], 255)
    return mask

# ========= ROI 수평교차 =========
def horizontal_span_in_poly(poly, y):
    """y=const와 다각형 교차의 [x_left, x_right] 반환. 없으면 None"""
    pts = poly.reshape(-1, 2).astype(np.float64)
    xs = []
    N = len(pts)
    for i in range(N):
        x1, y1 = pts[i]; x2, y2 = pts[(i+1) % N]
        if (y1 <= y and y2 > y) or (y2 <= y and y1 > y):
            t = (y - y1) / (y2 - y1 + 1e-12)
            xs.append(x1 + t * (x2 - x1))
    if len(xs) < 2: return None
    xs.sort()
    return float(xs[0]), float(xs[-1])

# ========= 감마 원근 그리드 =========
class GammaGrid:
    """
    행을 감마(γ)로 비선형 분할해 아래(가까움) 행을 두껍게, 위(먼 곳) 행을 얇게.
    각 행의 가로 셀폭은 '행 높이'에 비례 → 가까울수록 셀 크게.
    """
    def __init__(self, poly_px, img_h, img_w, n_rows=N_ROWS_INIT):
        self.poly = poly_px.astype(np.int32)
        self.h, self.w = img_h, img_w
        self.n_rows = int(max(6, n_rows))
        self.strips = []   # [{y_top,y_bot,xL_top,xR_top,xL_bot,xR_bot,W_top,W_bot,ncols,polys}]
        self.built = False

    def _edges_gamma(self, y_top, y_bot, gamma):
        edges=[]
        for k in range(self.n_rows+1):
            u = k / self.n_rows
            v = u ** gamma
            yk = y_top + (y_bot - y_top) * v
            edges.append(int(round(yk)))
        return edges

    def build(self, base_side_px: float, gamma: float):
        y_min = int(np.min(self.poly[:,1])); y_max = int(np.max(self.poly[:,1]))
        y_top = max(0, y_min+2); y_bot = min(self.h-2, y_max-1)
        if y_bot - y_top < 6*self.n_rows:
            self.strips=[]; self.built=False; return False

        edges = self._edges_gamma(y_top, y_bot, gamma)
        avgH = (y_bot - y_top) / self.n_rows
        scaleS = max(0.2, float(base_side_px) / max(1.0, avgH))  # W_row = scaleS * H_row

        strips=[]
        for r in range(self.n_rows):
            yt, yb = edges[r], edges[r+1]
            span_t = horizontal_span_in_poly(self.poly, yt)
            span_b = horizontal_span_in_poly(self.poly, yb)
            if (span_t is None) or (span_b is None): 
                continue
            xLt, xRt = span_t; xLb, xRb = span_b

            H = max(MIN_CELL_PX, yb - yt)
            Wt = Wb = max(MIN_CELL_PX, scaleS * H)

            ncols_t = int((xRt - xLt) // Wt)
            ncols_b = int((xRb - xLb) // Wb)
            ncols = max(0, min(ncols_t, ncols_b))
            if ncols < 1: 
                continue

            polys=[]
            for j in range(ncols):
                xt0 = xLt + j*Wt; xt1 = xLt + (j+1)*Wt
                xb0 = xLb + j*Wb; xb1 = xLb + (j+1)*Wb
                poly = np.array([[xt0, yt],[xt1, yt],[xb1, yb],[xb0, yb]], dtype=np.int32)
                polys.append(poly)

            strips.append(dict(
                y_top=int(yt), y_bot=int(yb),
                xL_top=float(xLt), xR_top=float(xRt),
                xL_bot=float(xLb), xR_bot=float(xRb),
                W_top=float(Wt), W_bot=float(Wb),
                ncols=len(polys), polys=polys
            ))

        self.strips = strips
        self.built = len(self.strips) > 0
        return self.built

    def locate_cell(self, x, y):
        for i, S in enumerate(self.strips):
            if not (S["y_top"] < y <= S["y_bot"]):
                continue
            jt = (x - S["xL_top"]) / (S["W_top"] + 1e-9)
            jb = (x - S["xL_bot"]) / (S["W_bot"] + 1e-9)
            j  = int(math.floor(0.5*(jt + jb)))
            if 0 <= j < S["ncols"]:
                if (x < min(S["xL_top"], S["xL_bot"])) or (x > max(S["xR_top"], S["xR_bot"])):
                    return (None, None)
                return (i, j)
        return (None, None)

# ========= 타일 추론/NMS =========
def split_tiles(frame, rows=ROWS_TILES, cols=COLS_TILES):
    h, w = frame.shape[:2]
    hs = [int(round(r*h/rows)) for r in range(rows+1)]
    ws = [int(round(c*w/cols)) for c in range(cols+1)]
    crops, offs = [], []
    for r in range(rows):
        for c in range(cols):
            y1,y2 = hs[r], hs[r+1]; x1,x2 = ws[c], ws[c+1]
            crops.append(frame[y1:y2, x1:x2]); offs.append((x1,y1))
    return crops, offs

def iou_xyxy(a,b):
    ax1,ay1,ax2,ay2=a[:4]; bx1,by1,bx2,by2=b[:4]
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
    # 1) 스트림
    stream_url = resolve_stream(YOUTUBE_URL)
    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        raise RuntimeError("VideoCapture 열기 실패")

    ok, first = cap.read()
    if not ok: raise RuntimeError("첫 프레임 읽기 실패")
    fr_h, fr_w = first.shape[:2]

    # 2) ROI
    roi_poly = click_roi(first)
    roi_mask = polygon_mask(fr_h, fr_w, roi_poly)

    # 3) YOLO
    model = YOLO(MODEL)

    # 4) 상태
    cur_preset  = 2            # 보통
    fine_factor = 1.0
    gamma_depth = GAMMA_INIT
    n_rows      = N_ROWS_INIT

    # 사람 높이 버퍼(중간밴드)
    hbuf = deque(maxlen=300)
    measure_start = time.time()

    # 그리드(한 번만 생성 후 고정)
    grid = GammaGrid(roi_poly, fr_h, fr_w, n_rows=n_rows)
    grid_built = False
    grid_dirty = True   # True일 때만 build

    # 5) 루프
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok: break

        # ----- 검출 -----
        crops, offs = split_tiles(frame)
        results = model(crops, conf=CONF, imgsz=IMGSZ, device=DEVICE, half=USE_HALF, classes=[0])

        # 박스 모으고 NMS
        all_boxes=[]
        for res,(ox,oy) in zip(results, offs):
            boxes=res.boxes
            if boxes is None or len(boxes)==0: continue
            xyxy = boxes.xyxy.cpu().numpy(); confs=boxes.conf.cpu().numpy()
            for (x1,y1,x2,y2),cf in zip(xyxy,confs):
                all_boxes.append([x1+ox, y1+oy, x2+ox, y2+oy, float(cf)])
        kept = nms_global(all_boxes, iou_th=0.55)

        # ----- 중간밴드 샘플 -----
        y_low  = int(MID_BAND[0] * fr_h)
        y_high = int(MID_BAND[1] * fr_h)
        for x1,y1,x2,y2,cf in kept:
            if cf < 0.45: continue
            h = y2-y1; w = x2-x1
            if h < MIN_H_PX_FOR_SAMPLE or w <= 0: continue
            if (h/(w+1e-9)) < ASPECT_MIN: continue
            footx = 0.5*(x1+x2); footy = y2
            if y_low <= footy <= y_high and roi_mask[int(min(max(footy,0),fr_h-1)), int(min(max(footx,0),fr_w-1))] != 0:
                hbuf.append(h)

        # ----- 격자 1회 생성(또는 수동으로만 재생성) -----
        if grid_dirty:
            if (len(hbuf) >= MIN_SAMPLES_FOR_BUILD) or (time.time() - measure_start > MEASURE_TIMEOUT_SEC):
                med_h = float(np.median(hbuf)) if len(hbuf) > 0 else 60.0
                base_side = BASE_K_FROM_PERSON_H * med_h * PRESET_FACTORS[cur_preset] * fine_factor
                base_side = max(MIN_CELL_PX, min(160.0, base_side))
                grid.build(base_side, gamma_depth)
                grid_built = grid.built
                grid_dirty = False

        # ----- 카운트 -----
        counts = []
        if grid_built:
            counts = [np.zeros(S["ncols"], dtype=np.int32) for S in grid.strips]
            for x1,y1,x2,y2,cf in kept:
                footx = 0.5*(x1+x2); footy = y2
                if roi_mask[int(min(max(footy,0),fr_h-1)), int(min(max(footx,0),fr_w-1))] == 0:
                    continue
                si, sj = grid.locate_cell(footx, footy)
                if si is not None:
                    counts[si][sj] += 1

        # ----- 렌더링 -----
        overlay = frame.copy()
        if grid_built:
            for i, S in enumerate(grid.strips):
                for j in range(S["ncols"]):
                    poly = S["polys"][j]
                    n = int(counts[i][j])
                    if n >= ALERT_THRESHOLD: color=(0,0,255)
                    elif n>0:               color=(0,180,0)
                    else:                   color=(180,180,180)
                    cv2.fillPoly(overlay, [poly], color)
                    cv2.polylines(overlay, [poly], True, (50,50,50), 1)
                    if DRAW_COUNTS and n>0:
                        cx=int(np.mean(poly[:,0])); cy=int(np.mean(poly[:,1]))
                        cv2.putText(overlay, str(n), (cx-6, cy+6), FONT, 0.6, (0,0,0), 2)
                        cv2.putText(overlay, str(n), (cx-6, cy+6), FONT, 0.6, (255,255,255), 1)

        overlay_masked = cv2.bitwise_and(overlay, overlay, mask=roi_mask)
        out = cv2.addWeighted(overlay_masked, FILL_ALPHA, frame, 1.0-FILL_ALPHA, 0)
        cv2.polylines(out, [roi_poly], True, (0,200,255), 2)

        # 텍스트
        title = f"~10 m^2 cells (heuristic freeze), Alert >= {ALERT_THRESHOLD}"
        cv2.putText(out, title, (12,28), FONT, 0.7, (0,0,0), 3)
        cv2.putText(out, title, (12,28), FONT, 0.7, (255,255,255), 1)

        if len(hbuf)>0:
            med_txt = f"med_h={np.median(hbuf):.1f}px"
        else:
            med_txt = "med_h=--"
        state = f"Preset: {PRESET_NAMES[cur_preset]}  gamma={gamma_depth:.2f}  {med_txt}"
        cv2.putText(out, state, (12,52), FONT, 0.7, (0,0,0), 3)
        cv2.putText(out, state, (12,52), FONT, 0.7, (255,255,255), 1)

        help1 = "Manual: [1..5]=preset  +/-=size  [ / ]=gamma  R=remeasure"
        cv2.putText(out, help1, (12,76), FONT, 0.7, (0,0,0), 3)
        cv2.putText(out, help1, (12,76), FONT, 0.7, (255,255,255), 1)

        if not grid_built:
            cv2.putText(out, "Measuring people size...", (12,100), FONT, 0.7, (0,0,0), 3)
            cv2.putText(out, "Measuring people size...", (12,100), FONT, 0.7, (255,255,255), 1)

        show_fit("Person-calibrated Grid Crowd Count", out)

        # ----- 키 입력 -----
        k = cv2.waitKey(1) & 0xFF
        if k in (27, ord('q'), ord('Q')): break

        if k in (ord('1'),ord('2'),ord('3'),ord('4'),ord('5')):
            cur_preset = int(chr(k)) - 1
            grid_dirty = True  # 다음 프레임에 재빌드

        elif k in (ord('+'), ord('=')):
            fine_factor *= SCALE_STEP
            grid_dirty = True

        elif k in (ord('-'), ord('_')):
            fine_factor /= SCALE_STEP
            grid_dirty = True

        elif k == ord('['):
            gamma_depth = max(0.80, gamma_depth - GAMMA_STEP)
            grid_dirty = True

        elif k == ord(']'):
            gamma_depth = min(2.50, gamma_depth + GAMMA_STEP)
            grid_dirty = True

        elif k in (ord('r'), ord('R')):
            # 다시 측정: 샘플 버퍼 비우고 타이머 리셋 → 새 파라미터로 1회 재빌드
            hbuf.clear()
            measure_start = time.time()
            grid_dirty = True

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
