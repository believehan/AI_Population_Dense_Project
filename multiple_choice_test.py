# -*- coding: utf-8 -*-
# YOLO11s + 5지선다(사람크기) 1회 측정 → 고정 + 정사각형 그리드(원근 축소 옵션)
# - 샘플 대역: 화면 높이 20~40%
# - 프리셋 스케일: 아주작음일수록 셀이 더 큼(1>2>3>4>5)
# - ROI 밖은 전혀 그리지 않음(오버레이도 ROI로 클립)
# - P: 원근(행별 5% 축소) 토글

import subprocess, shutil, time, cv2, math
import numpy as np
from collections import deque
from ultralytics import YOLO

# ========= 기본 설정 =========
YOUTUBE_URL = "https://www.youtube.com/live/rnXIjl_Rzy4?feature=shared"
# YOUTUBE_URL = "https://www.youtube.com/live/EaRgJQ--2eE"
# YOUTUBE_URL = "https://www.youtube.com/watch?v=u4UZ4UvZXrg"
MODEL   = "epoch30.pt"
DEVICE  = "cpu"
IMGSZ   = 640
CONF    = 0.35
USE_HALF= False
ROWS_TILES, COLS_TILES = 2, 4

# --- 자동 측정 관련 ---
AUTO_MIN_SAMPLES = 12          # 자동확정에 필요한 최소 샘플 수(기존 25 → 12)
AUTO_TIMEOUT_SEC = 25.0        # 이 시간 지나면 강제 확정
# 샘플 필터 완화(사람이 적은 장면 대비)
MIN_H_PX_FOR_SAMPLE = 32       # 36 → 32
ASPECT_MIN = 1.25              # 1.35 → 1.25

# --- 사람 위치 참조점 설정 ---
USE_HEAD_POINT = True       # True=머리(y1), False=발끝(y2)
ROI_TEST_WITH_REF = True    # ROI 포함 여부도 같은 점으로 검사(True 권장)


# 사람-크기 5지선다 (Very big 일수록 더 큰 셀)
PRESET_NAMES   = ["Very big", "big", "Normal", "small", "Very small"] # 사람 크기에 따라 분류
PRESET_FACTORS = [1.95, 1.75, 1.45, 1.20, 1.00]
BASE_K_FROM_PERSON_H = 0.90
MID_BAND = (0.20, 0.40)      # 20~40%

# 셀/표시
MIN_CELL_PX = 12
MAX_CELL_PX = 120
ALERT_THRESHOLD = 5
FILL_ALPHA = 0.45
DRAW_COUNTS = True

# 자동/고정
AUTO_RECALC_SEC = 1.5
FREEZE_AFTER_FIRST_BUILD = True

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

def ref_point(x1, y1, x2, y2, use_head=True):
    """사람 박스의 기준점 반환 (머리 or 발끝)"""
    cx = 0.5 * (x1 + x2)
    cy = y1 if use_head else y2
    return cx, cy


# ========= streamlink =========
def resolve_stream(url: str) -> str:
    exe = (shutil.which("streamlink")
           or r"C:\Users\user\anaconda3\envs\py39\Scripts\streamlink.exe")
    if not exe: raise RuntimeError("streamlink 실행파일을 찾지 못했습니다.")
    out = subprocess.run([exe, "--stream-url", url, "best"],
                         capture_output=True, text=True, check=True)
    s = out.stdout.strip()
    if not s: raise RuntimeError("streamlink로 스트림 URL 얻기 실패")
    return s

# ========= ROI =========
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
        if len(roi) >= 3: cv2.line(disp, tuple(roi[-1]), tuple(roi[0]), (0,200,255), 1)
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

# ========= 타일 추론/NMS =========
def split_tiles(frame, rows=2, cols=4):
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

# ========= 다각형 교차(셀 표시 여부) =========
def _pt_in_poly(pt, poly):
    return cv2.pointPolygonTest(poly, (float(pt[0]), float(pt[1])), False) >= 0
def _seg_intersect(p1,p2,q1,q2):
    def ccw(a,b,c): return (c[1]-a[1])*(b[0]-a[0]) - (b[1]-a[1])*(c[0]-a[0])
    A=ccw(p1,p2,q1); B=ccw(p1,p2,q2); C=ccw(q1,q2,p1); D=ccw(q1,q2,p2)
    if A==B==C==D==0:
        if max(min(p1[0],p2[0]), min(q1[0],q2[0])) <= min(max(p1[0],p2[0]), max(q1[0],q2[0])) and \
           max(min(p1[1],p2[1]), min(q1[1],q2[1])) <= min(max(p1[1],p2[1]), max(q1[1],q2[1])):
            return True
        return False
    return (A*B<=0) and (C*D<=0)
def poly_intersects(poly_a, poly_b):
    for p in poly_a:
        if _pt_in_poly(p, poly_b): return True
    for p in poly_b:
        if _pt_in_poly(p, poly_a): return True
    A=poly_a; B=poly_b
    for i in range(len(A)):
        a1 = tuple(A[i]); a2 = tuple(A[(i+1)%len(A)])
        for j in range(len(B)):
            b1 = tuple(B[j]); b2 = tuple(B[(j+1)%len(B)])
            if _seg_intersect(a1,a2,b1,b2): return True
    return False

# ========= 정사각형 그리드 (원근 옵션) =========
class VarSquareGrid:
    """
    ROI 바운딩박스 안을 '정사각형'으로 채우되,
    아래에서 위로 갈수록 한 행씩 (1 - shrink)^r 만큼 축소시킴.
    - cells: list of dict {poly(4x2 int32), row, col}
    """
    def __init__(self, roi_poly, img_h, img_w):
        self.poly = roi_poly.astype(np.int32)
        self.h, self.w = img_h, img_w
        x,y,w,h = cv2.boundingRect(self.poly)
        self.bbox = (x,y,w,h)
        self.cells = []
        self.base_side = None
        self.shrink = 0.05  # 기본 5%씩 축소
        self.perspective_on = True

    def build(self, base_side_px: float):
        self.base_side = int(max(MIN_CELL_PX, min(MAX_CELL_PX, round(base_side_px))))
        self.cells = []
        x, y, w, h = self.bbox
        # 아래 → 위로 행 생성
        y_bot = y + h
        row = 0
        while True:
            side = self.base_side
            if self.perspective_on:
                side = max(MIN_CELL_PX, int(round(self.base_side * ((1.0 - self.shrink) ** row))))
            y_top = y_bot - side
            if y_top < y:  # 넘어가면 마지막 줄만 남기고 종료
                break
            # 열 생성 (정사각형)
            x0 = x  # 좌측부터
            col = 0
            while x0 < x + w:
                poly = np.array([[x0, y_top],
                                 [x0+side, y_top],
                                 [x0+side, y_bot],
                                 [x0, y_bot]], dtype=np.int32)
                self.cells.append(dict(poly=poly, row=row, col=col, side=side))
                x0 += side
                col += 1
            y_bot = y_top
            row  += 1
        return len(self.cells) > 0

    def locate(self, x, y):
        # 간단: 셀 리스트를 순회(셀 수가 많지 않으므로 OK)
        for idx, c in enumerate(self.cells):
            p = c["poly"]
            if (x >= p[0,0]) and (x <= p[1,0]) and (y > p[0,1]) and (y <= p[2,1]):
                return idx
        return None

# ========= 메인 =========
def main():
    global USE_HEAD_POINT   # ← 이 줄 추가 (머리/발끝 토글 변수는 전역 정의였으니)
    stream_url = resolve_stream(YOUTUBE_URL)
    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened(): raise RuntimeError("VideoCapture 열기 실패")

    ok, first = cap.read()
    if not ok: raise RuntimeError("첫 프레임 읽기 실패")
    fr_h, fr_w = first.shape[:2]

    roi_poly = click_roi(first)
    roi_mask = polygon_mask(fr_h, fr_w, roi_poly)

    model = YOLO(MODEL)

    grid = VarSquareGrid(roi_poly, fr_h, fr_w)
    cur_preset = 2
    fine_factor = 1.0
    gamma = 1.0

    # --- NEW: ROI 확정 즉시 보이는 임시 격자 ---
    # fallback_h_px = max(MIN_H_PX_FOR_SAMPLE, int(0.065 * fr_h))  # 프레임 높이의 ~6.5%를 임시 사람키로 가정
    # init_side = max(MIN_CELL_PX, min(MAX_CELL_PX,
    #             int(BASE_K_FROM_PERSON_H * fallback_h_px * PRESET_FACTORS[cur_preset] * fine_factor)))
    # grid.build(init_side)          # 미리 한 번 그려주기 (자동 측정 끝나면 덮어씀)
    measure_started = time.time()   # <-- NEW: 자동 측정 타이머 시작

    hbuf = deque(maxlen=300)
    last_auto = 0.0
    grid_frozen = False


    while cap.isOpened():
        ok, frame = cap.read()
        if not ok: break

        # 감마 옵션
        if gamma != 1.0:
            f = np.clip(((frame/255.0) ** (1.0/gamma))*255.0, 0, 255).astype(np.uint8)
            frame = f

        crops, offs = split_tiles(frame, ROWS_TILES, COLS_TILES)
        results = model(crops, conf=CONF, imgsz=IMGSZ, device=DEVICE, half=USE_HALF, classes=[0])

        all_boxes=[]
        for res,(ox,oy) in zip(results, offs):
            boxes=res.boxes
            if boxes is None or len(boxes)==0: continue
            xyxy = boxes.xyxy.cpu().numpy(); confs=boxes.conf.cpu().numpy()
            for (x1,y1,x2,y2),cf in zip(xyxy,confs):
                all_boxes.append([x1+ox, y1+oy, x2+ox, y2+oy, float(cf)])
        kept = nms_global(all_boxes, iou_th=0.55)

        # # A) 샘플 수집(20~40% + 부족하면 40~65%) 발끝
        # y_low, y_high = int(MID_BAND[0]*fr_h), int(MID_BAND[1]*fr_h)
        # y_low2, y_high2 = int(0.40*fr_h), int(0.65*fr_h)  # 보조 밴드

        # if not grid_frozen:
        #     for x1,y1,x2,y2,cf in kept:
        #         if cf < 0.35:   # 0.45 → 0.35 (완화)
        #             continue
        #         h = y2-y1; w = x2-x1
        #         if h < MIN_H_PX_FOR_SAMPLE or w <= 0:
        #             continue
        #         if (h/(w+1e-9)) < ASPECT_MIN:
        #             continue
        #         footx = 0.5*(x1+x2); footy = y2
        #         if roi_mask[int(min(max(footy,0),fr_h-1)), int(min(max(footx,0),fr_w-1))] == 0:
        #             continue

        #         # 기본 밴드 또는 보조 밴드에서 수집
        #         if y_low <= footy <= y_high or y_low2 <= footy <= y_high2:
        #             hbuf.append(h)
        
        # A) 샘플 수집(20~40% + 부족하면 40~65%, ROI 내부) 머리
        y_low, y_high   = int(MID_BAND[0]*fr_h), int(MID_BAND[1]*fr_h)
        y_low2, y_high2 = int(0.40*fr_h), int(0.65*fr_h)

        if not grid_frozen:
            for x1,y1,x2,y2,cf in kept:
                if cf < 0.35: 
                    continue
                h = y2 - y1; w = x2 - x1
                if h < MIN_H_PX_FOR_SAMPLE or w <= 0:
                    continue
                if (h/(w+1e-9)) < ASPECT_MIN:
                    continue

                rx, ry = ref_point(x1,y1,x2,y2, use_head=USE_HEAD_POINT)
                # ROI 포함 검사
                testx, testy = (rx, ry) if ROI_TEST_WITH_REF else ref_point(x1,y1,x2,y2, use_head=False)  # False=발끝
                if roi_mask[int(min(max(testy,0),fr_h-1)), int(min(max(testx,0),fr_w-1))] == 0:
                    continue

                # 밴드 조건
                if y_low <= ry <= y_high or y_low2 <= ry <= y_high2:
                    hbuf.append(h)


        # B) 자동 프리셋(처음만)
        now = time.time()
        enough_samples = (len(hbuf) >= AUTO_MIN_SAMPLES)
        timeout_passed = ((now - measure_started) > AUTO_TIMEOUT_SEC and len(hbuf) >= max(5, AUTO_MIN_SAMPLES//2))

        if (not grid_frozen) and (enough_samples or timeout_passed):
            if len(hbuf) > 0:
                med_h = float(np.median(hbuf))
            else:
                med_h = max(MIN_H_PX_FOR_SAMPLE, int(0.065 * fr_h))  # 완전 무샘플이면 안전한 대체값
            base_side = max(MIN_CELL_PX, min(MAX_CELL_PX,
                        BASE_K_FROM_PERSON_H * med_h * PRESET_FACTORS[cur_preset] * fine_factor))
            grid.build(base_side)
            grid_frozen = True


        # C) 수동/재측정/토글
        k = cv2.waitKey(1) & 0xFF

        if k in (27, ord('q')): break
        if k in (ord('1'),ord('2'),ord('3'),ord('4'),ord('5')):
            cur_preset = int(chr(k)) - 1
            med_h = float(np.median(hbuf)) if len(hbuf)>0 else 60.0
            base_side = max(MIN_CELL_PX,
                            min(MAX_CELL_PX, BASE_K_FROM_PERSON_H * med_h * PRESET_FACTORS[cur_preset] * fine_factor))
            grid.build(base_side); grid_frozen = True
        elif k in (ord('+'), ord('=')):
            fine_factor *= 1.05
            med_h = float(np.median(hbuf)) if len(hbuf)>0 else 60.0
            base_side = max(MIN_CELL_PX,
                            min(MAX_CELL_PX, BASE_K_FROM_PERSON_H * med_h * PRESET_FACTORS[cur_preset] * fine_factor))
            grid.build(base_side); grid_frozen = True
        elif k in (ord('-'), ord('_')):
            fine_factor /= 1.05
            med_h = float(np.median(hbuf)) if len(hbuf)>0 else 60.0
            base_side = max(MIN_CELL_PX,
                            min(MAX_CELL_PX, BASE_K_FROM_PERSON_H * med_h * PRESET_FACTORS[cur_preset] * fine_factor))
            grid.build(base_side); grid_frozen = True
            
        elif k in (ord('h'), ord('H')):
            USE_HEAD_POINT = True     # 머리 기준
        elif k in (ord('f'), ord('F')):
            USE_HEAD_POINT = False    # 발끝 기준
            
        elif k in (ord('r'), ord('R')):
            hbuf.clear(); last_auto = 0.0; grid_frozen = False
        elif k in (ord('p'), ord('P')):
            grid.perspective_on = not grid.perspective_on
            # 재빌드
            med_h = float(np.median(hbuf)) if len(hbuf)>0 else 60.0
            base_side = max(MIN_CELL_PX,
                            min(MAX_CELL_PX, BASE_K_FROM_PERSON_H * med_h * PRESET_FACTORS[cur_preset] * fine_factor))
            grid.build(base_side)

        # # D) 카운트(발끝이 ROI 안일 때만)
        # counts = {}
        # if grid.base_side is not None:
        #     for x1,y1,x2,y2,cf in kept:
        #         footx = 0.5*(x1+x2); footy = y2
        #         if roi_mask[int(min(max(footy,0),fr_h-1)), int(min(max(footx,0),fr_w-1))] == 0: continue
        #         idx = grid.locate(footx, footy)
        #         if idx is None: continue
        #         counts[idx] = counts.get(idx, 0) + 1
        
        # D) 카운트(참조점(머리)이 ROI 안일 때만)
        counts = {}
        if grid.base_side is not None:
            for x1,y1,x2,y2,cf in kept:
                rx, ry = ref_point(x1,y1,x2,y2, use_head=USE_HEAD_POINT)
                testx, testy = (rx, ry) if ROI_TEST_WITH_REF else ref_point(x1,y1,x2,y2, use_head=False)
                if roi_mask[int(min(max(testy,0),fr_h-1)), int(min(max(testx,0),fr_w-1))] == 0:
                    continue
                idx = grid.locate(rx, ry)
                if idx is None: 
                    continue
                counts[idx] = counts.get(idx, 0) + 1


        # E) 렌더링(ROI와 전혀 겹치지 않는 셀은 숨김) + ROI 마스크로 클립
        overlay = frame.copy()
        if grid.base_side is not None:
            for i, c in enumerate(grid.cells):
                poly = c["poly"]
                if not poly_intersects(poly, roi_poly):  # 완전 바깥이면 스킵
                    continue
                n = counts.get(i, 0)
                if n >= ALERT_THRESHOLD: color=(0,0,255)
                elif n>0:               color=(0,180,0)
                else:                   color=(180,180,180)
                if c["side"] >= MIN_CELL_PX:
                    cv2.fillPoly(overlay, [poly], color)
                    cv2.polylines(overlay, [poly], True, (50,50,50), 1)

        # ▶ ROI로 클립 후 블렌딩(바깥은 원본 유지)
        overlay_masked = cv2.bitwise_and(overlay, overlay, mask=roi_mask)
        out = cv2.addWeighted(overlay_masked, FILL_ALPHA, frame, 1.0-FILL_ALPHA, 0)
        if not grid_frozen:
            need = AUTO_MIN_SAMPLES
            left = max(0, int(AUTO_TIMEOUT_SEC - (time.time() - measure_started)))
            cv2.putText(out, f"Measuring... samples {len(hbuf)}/{need}  timeout {left}s  [A]=accept now",
                        (12,100), FONT, 0.7, (0,0,0), 3)
            cv2.putText(out, f"Measuring... samples {len(hbuf)}/{need}  timeout {left}s  [A]=accept now",
                        (12,100), FONT, 0.7, (255,255,255), 1)
        
        cv2.polylines(out, [roi_poly], True, (0,200,255), 2)
        mode_txt = "heuristic freeze" if grid_frozen else "measuring..."
        cv2.putText(out, f"~10 m^2 cells ({mode_txt}), Alert >= {ALERT_THRESHOLD}",
                    (12,28), FONT, 0.7, (0,0,0), 3)
        cv2.putText(out, f"~10 m^2 cells ({mode_txt}), Alert >= {ALERT_THRESHOLD}",
                    (12,28), FONT, 0.7, (255,255,255), 1)

        med_txt = f"med_h={np.median(hbuf):.1f}px" if len(hbuf)>0 else "med_h=--"
        p_txt = "ON" if grid.perspective_on else "OFF"
        cv2.putText(out, f"Preset: {PRESET_NAMES[cur_preset]}  x{PRESET_FACTORS[cur_preset]*fine_factor:.2f}  {med_txt}  perspective={p_txt}",
                    (12,52), FONT, 0.7, (0,0,0), 3)
        cv2.putText(out, f"Preset: {PRESET_NAMES[cur_preset]}  x{PRESET_FACTORS[cur_preset]*fine_factor:.2f}  {med_txt}  perspective={p_txt}",
                    (12,52), FONT, 0.7, (255,255,255), 1)
        cv2.putText(out, "Manual: [1..5]=preset  +/-=size   R=remeasure   P=perspective on/off",
                    (12,76), FONT, 0.7, (0,0,0), 3)
        cv2.putText(out, "Manual: [1..5]=preset  +/-=size   R=remeasure   P=perspective on/off",
                    (12,76), FONT, 0.7, (255,255,255), 1)
        
        # 머리로 디텍팅하기 와 발끝으로 디텍팅하기 표시
        ref_txt = "HEAD" if USE_HEAD_POINT else "FOOT"
        p_txt = "ON" if grid.perspective_on else "OFF"
        cv2.putText(out, f"Preset: {PRESET_NAMES[cur_preset]}  x{PRESET_FACTORS[cur_preset]*fine_factor:.2f}  {med_txt}  perspective={p_txt}  ref={ref_txt}  [H/F]",
                    (12,52), FONT, 0.7, (0,0,0), 3)
        cv2.putText(out, f"Preset: {PRESET_NAMES[cur_preset]}  x{PRESET_FACTORS[cur_preset]*fine_factor:.2f}  {med_txt}  perspective={p_txt}  ref={ref_txt}  [H/F]",
                    (12,52), FONT, 0.7, (255,255,255), 1)


        show_fit("Person-calibrated Grid Crowd Count", out)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()