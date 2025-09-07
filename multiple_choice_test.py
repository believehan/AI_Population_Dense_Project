# -*- coding: utf-8 -*-
# YouTube(streamlink) + YOLO11s + 5지선다(사람크기) 자동-선택 + 직사각형 ROI 그리드 카운트
# - 원근/기울기 고려 X (화면축 정렬 네모)
# - 중간 높이(40~60%) 사람의 bbox 높이 분포를 보고 5개 프리셋 중 하나 자동 선택
# - 필요시 키보드 1~5로 수동 선택(1=아주작음, 3=보통, 5=아주큼)

import subprocess, shutil, time, math
import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO

# ========= 기본 설정 =========
# YOUTUBE_URL = "https://www.youtube.com/live/rnXIjl_Rzy4?feature=shared"
# YOUTUBE_URL = "https://www.youtube.com/live/EaRgJQ--2eE"
YOUTUBE_URL = "https://www.youtube.com/watch?v=u4UZ4UvZXrg"
MODEL   = "yolo11s.pt"
DEVICE  = "cpu"       # GPU면 "cuda:0"
IMGSZ   = 640
CONF    = 0.35
USE_HALF= False
ROWS_TILES, COLS_TILES = 2, 4

# 사람-크기 5지선다(자동/수동 공통으로 쓰는 스케일 팩터)
PRESET_NAMES   = ["아주작음", "작음", "보통", "큼", "아주큼"]
PRESET_FACTORS = [0.75, 0.90, 1.00, 1.15, 1.35]   # 셀 픽셀 크기에 곱
AUTO_RECALC_SEC = 1.5                              # 자동 재선택 주기
MID_BAND = (0.40, 0.60)                            # 프레임 높이 기준 중간 밴드
MIN_H_PX_FOR_SAMPLE = 36                           # 샘플에 쓸 최소 사람높이(px)
ASPECT_MIN = 1.35                                  # 전신(높/너) 하한

# 셀/표시
TARGET_CELL_AREA_M2 = 10.0
BASE_K_FROM_PERSON_H = 0.90   # 셀 한 변(px) ≈ BASE_K * (중간밴드 사람 높이 px)
MIN_CELL_PX = 12
MAX_CELL_PX = 120
ALERT_THRESHOLD = 5
FILL_ALPHA = 0.45
DRAW_COUNTS = True

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

# ========= 직사각형 그리드(화면축 정렬) =========
class RectGrid:
    """
    - 화면축 정렬 네모 셀
    - anchor는 ROI 바운딩박스의 좌상단을 'cell_side' 그리드에 스냅
    - 각 셀의 중심이 ROI 안쪽이면 채택
    """
    def __init__(self, roi_poly, img_h, img_w):
        self.poly = roi_poly.astype(np.int32)
        self.h, self.w = img_h, img_w
        x,y,w,h = cv2.boundingRect(self.poly)
        self.bbox = (x,y,w,h)
        self.cells = {}  # (i,j) -> poly(int32)
        self.side = None
        self.anchor = None  # (x0,y0)

    def _inside(self, x, y):
        # 중심점이 ROI 내부인지
        return cv2.pointPolygonTest(self.poly, (float(x), float(y)), measureDist=False) >= 0

    def build(self, cell_side_px: float):
        side = int(max(MIN_CELL_PX, min(MAX_CELL_PX, round(cell_side_px))))
        if side < MIN_CELL_PX: 
            self.cells = {}; self.side=None; return False
        self.side = side
        x, y, w, h = self.bbox
        # 그리드 앵커를 셀 크기에 스냅
        x0 = (x // side) * side
        y0 = (y // side) * side
        self.anchor = (x0, y0)
        self.cells = {}
        # 셀 생성
        for yy in range(y0, y + h + side, side):
            for xx in range(x0, x + w + side, side):
                cx = xx + side*0.5; cy = yy + side*0.5
                if not self._inside(cx, cy): 
                    continue
                poly = np.array([[xx,yy],[xx+side,yy],[xx+side,yy+side],[xx,yy+side]], dtype=np.int32)
                self.cells[(yy//side, xx//side)] = poly
        return len(self.cells) > 0

    def locate(self, x, y):
        if self.side is None or self.anchor is None: 
            return None
        x0,y0 = self.anchor
        i = (int(y) // self.side)
        j = (int(x) // self.side)
        key = (i,j)
        return key if key in self.cells else None

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

    # 4) 그리드 & 자동 프리셋 상태
    grid = RectGrid(roi_poly, fr_h, fr_w)
    cur_preset = 2  # 0..4 (보통)
    fine_factor = 1.0

    # 사람 높이 버퍼(중간밴드)
    hbuf = deque(maxlen=300)
    last_auto = 0.0

    # 5) 루프
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok: break

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

        # === 5-A) 중간밴드 샘플 수집 ===
        y_low  = int(MID_BAND[0] * fr_h)
        y_high = int(MID_BAND[1] * fr_h)
        for x1,y1,x2,y2,cf in kept:
            if cf < 0.45: continue
            h = y2-y1; w = x2-x1
            if h < MIN_H_PX_FOR_SAMPLE or w <= 0: continue
            if (h/(w+1e-9)) < ASPECT_MIN: continue
            footx = 0.5*(x1+x2); footy = y2
            # ROI 내부 + 중간밴드
            if y_low <= footy <= y_high and roi_mask[int(min(max(footy,0),fr_h-1)), int(min(max(footx,0),fr_w-1))] != 0:
                hbuf.append(h)

        # === 5-B) 자동 프리셋 선택(주기적으로) ===
        now = time.time()
        if (now - last_auto > AUTO_RECALC_SEC) and len(hbuf) >= 25:
            med_h = float(np.median(hbuf))
            # 프레임 높이에 대한 비율로 5지선다
            r = med_h / fr_h
            # 임계(대략값): 아주작음 <0.035 <작음 <0.055 <보통 <0.075 <큼 <0.095 <= 아주큼
            if   r < 0.035: cur_preset = 0
            elif r < 0.055: cur_preset = 1
            elif r < 0.075: cur_preset = 2
            elif r < 0.095: cur_preset = 3
            else:           cur_preset = 4
            last_auto = now

            # 그리드 재빌드
            cell_side = max(MIN_CELL_PX,
                            min(MAX_CELL_PX, BASE_K_FROM_PERSON_H * med_h * PRESET_FACTORS[cur_preset] * fine_factor))
            grid.build(cell_side)

        # === 5-C) 수동 프리셋(1~5), 미세조정(+/-) ===
        k = cv2.waitKey(1) & 0xFF
        if k in (27, ord('q')): break
        if k in (ord('1'),ord('2'),ord('3'),ord('4'),ord('5')):
            cur_preset = int(chr(k)) - 1
            # 수동 시에도 최신 중앙값이 있으면 그리드 즉시 재빌드
            med_h = float(np.median(hbuf)) if len(hbuf)>0 else 60.0
            cell_side = max(MIN_CELL_PX,
                            min(MAX_CELL_PX, BASE_K_FROM_PERSON_H * med_h * PRESET_FACTORS[cur_preset] * fine_factor))
            grid.build(cell_side)
        elif k in (ord('+'), ord('=')):
            fine_factor *= 1.05
            med_h = float(np.median(hbuf)) if len(hbuf)>0 else 60.0
            cell_side = max(MIN_CELL_PX,
                            min(MAX_CELL_PX, BASE_K_FROM_PERSON_H * med_h * PRESET_FACTORS[cur_preset] * fine_factor))
            grid.build(cell_side)
        elif k in (ord('-'), ord('_')):
            fine_factor /= 1.05
            med_h = float(np.median(hbuf)) if len(hbuf)>0 else 60.0
            cell_side = max(MIN_CELL_PX,
                            min(MAX_CELL_PX, BASE_K_FROM_PERSON_H * med_h * PRESET_FACTORS[cur_preset] * fine_factor))
            grid.build(cell_side)

        # === 5-D) 셀 카운트 ===
        counts = {}
        if grid.side is not None:
            for x1,y1,x2,y2,cf in kept:
                footx = 0.5*(x1+x2); footy = y2
                # ROI 내에서만 카운트
                if roi_mask[int(min(max(footy,0),fr_h-1)), int(min(max(footx,0),fr_w-1))] == 0:
                    continue
                key = grid.locate(footx, footy)
                if key is None: continue
                counts[key] = counts.get(key, 0) + 1

        # === 5-E) 렌더링 ===
        overlay = frame.copy()

        # 셀 렌더
        if grid.side is not None:
            for key, poly in grid.cells.items():
                n = counts.get(key, 0)
                if n >= ALERT_THRESHOLD: color=(0,0,255)
                elif n>0:               color=(0,180,0)
                else:                   color=(180,180,180)
                # 너무 작은 셀은 생략
                if grid.side >= MIN_CELL_PX:
                    cv2.fillPoly(overlay, [poly], color)
                    cv2.polylines(overlay, [poly], True, (50,50,50), 1)
                    if DRAW_COUNTS and n>0:
                        cx=int(np.mean(poly[:,0])); cy=int(np.mean(poly[:,1]))
                        cv2.putText(overlay, str(n), (cx-6, cy+6), FONT, 0.6, (0,0,0), 2)
                        cv2.putText(overlay, str(n), (cx-6, cy+6), FONT, 0.6, (255,255,255), 1)

        # ROI 마스크 + 블렌딩
        overlay_masked = cv2.bitwise_and(overlay, overlay, mask=roi_mask)
        out = cv2.addWeighted(overlay_masked, FILL_ALPHA, frame, 1.0-FILL_ALPHA, 0)

        # ROI 윤곽
        cv2.polylines(out, [roi_poly], True, (0,200,255), 2)

        # 텍스트
        cv2.putText(out, f"~{TARGET_CELL_AREA_M2:.0f} m^2 cells (heuristic), Alert >= {ALERT_THRESHOLD}",
                    (12,28), FONT, 0.7, (0,0,0), 3)
        cv2.putText(out, f"~{TARGET_CELL_AREA_M2:.0f} m^2 cells (heuristic), Alert >= {ALERT_THRESHOLD}",
                    (12,28), FONT, 0.7, (255,255,255), 1)

        # 프리셋/상태
        med_txt = f"med_h={np.median(hbuf):.1f}px" if len(hbuf)>0 else "med_h=--"
        cv2.putText(out, f"Preset: {PRESET_NAMES[cur_preset]}  x{PRESET_FACTORS[cur_preset]*fine_factor:.2f}  {med_txt}",
                    (12,52), FONT, 0.7, (0,0,0), 3)
        cv2.putText(out, f"Preset: {PRESET_NAMES[cur_preset]}  x{PRESET_FACTORS[cur_preset]*fine_factor:.2f}  {med_txt}",
                    (12,52), FONT, 0.7, (255,255,255), 1)
        cv2.putText(out, "Manual: [1..5] preset, [+]/[-] fine", (12,76), FONT, 0.7, (0,0,0), 3)
        cv2.putText(out, "Manual: [1..5] preset, [+]/[-] fine", (12,76), FONT, 0.7, (255,255,255), 1)

        show_fit("Person-calibrated Grid Crowd Count", out)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
