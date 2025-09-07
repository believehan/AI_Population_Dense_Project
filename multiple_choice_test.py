# -*- coding: utf-8 -*-
# YOLO11s + 5지선다(사람크기) 1회 측정 → 고정 + 정사각형 그리드(ROI 경계에 잘려도 보정 안 함)
# - 샘플 대역: 화면 높이의 20%~40%
# - [R] 재측정(다시 1회 측정 후 고정), [1..5] 수동 프리셋, [+]/[-] 미세조정, [G]/[H] 감마

import subprocess, shutil, time, cv2, math
import numpy as np
from collections import deque
from ultralytics import YOLO

# ========= 기본 설정 =========
YOUTUBE_URL = "https://www.youtube.com/live/rnXIjl_Rzy4?feature=shared"
MODEL   = "yolo11s.pt"
DEVICE  = "cpu"
IMGSZ   = 640
CONF    = 0.35
USE_HALF= False
ROWS_TILES, COLS_TILES = 2, 4

# 사람-크기 5지선다
PRESET_NAMES   = ["아주작음", "작음", "보통", "큼", "아주큼"]
PRESET_FACTORS = [0.75, 0.90, 1.00, 1.15, 1.35]    # 셀 한 변(px)에 곱해지는 계수
BASE_K_FROM_PERSON_H = 0.90                       # 셀 한 변 ≈ k * (중간밴드 사람높이)
MID_BAND = (0.20, 0.40)                           # ⬅️ 20~40%로 변경
MIN_H_PX_FOR_SAMPLE = 36
ASPECT_MIN = 1.35

# 셀/표시
MIN_CELL_PX = 12
MAX_CELL_PX = 120
ALERT_THRESHOLD = 5
FILL_ALPHA = 0.45
DRAW_COUNTS = True
CLIP_TO_ROI = False       # ⬅️ 정사각형 유지 위해 ROI로 클리핑하지 않음(원하면 True로)

# 자동/고정 로직
AUTO_RECALC_SEC = 1.5     # 자동 선택 주기(‘처음 측정’ 단계에서만 사용)
FREEZE_AFTER_FIRST_BUILD = True  # ⬅️ 최초 빌드 후 자동 고정

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
import subprocess, shutil
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

# ========= 정사각형 그리드 =========
class RectGrid:
    """
    ROI의 바운딩 박스 전체를 셀 크기 격자로 채움(ROI 바깥도 포함)
    - counting은 '발끝이 ROI 안일 때만' 증가
    - 셀 자체는 항상 정사각형(ROI 경계에 의해 잘라내지 않음)
    """
    def __init__(self, roi_poly, img_h, img_w):
        self.poly = roi_poly.astype(np.int32)
        self.h, self.w = img_h, img_w
        x,y,w,h = cv2.boundingRect(self.poly)
        self.bbox = (x,y,w,h)
        self.cells = {}   # (i,j) -> poly(int32)
        self.side = None
        self.anchor = None  # (x0,y0)

    def build(self, cell_side_px: float):
        side = int(max(MIN_CELL_PX, min(MAX_CELL_PX, round(cell_side_px))))
        if side < MIN_CELL_PX:
            self.cells = {}; self.side=None; return False
        self.side = side
        x, y, w, h = self.bbox
        # 셀 앵커를 그리드에 스냅
        x0 = (x // side) * side
        y0 = (y // side) * side
        self.anchor = (x0, y0)
        self.cells = {}
        # 바운딩박스 전체를 정사각형으로 채움(ROI 안/밖 무관)
        for yy in range(y0, y + h + side, side):
            for xx in range(x0, x + w + side, side):
                poly = np.array([[xx,yy],[xx+side,yy],[xx+side,yy+side],[xx,yy+side]], dtype=np.int32)
                self.cells[(yy//side, xx//side)] = poly
        return len(self.cells) > 0

    def locate(self, x, y):
        if self.side is None or self.anchor is None: return None
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

    # 4) 그리드 & 상태
    grid = RectGrid(roi_poly, fr_h, fr_w)
    cur_preset = 2
    fine_factor = 1.0
    gamma = 1.35   # 히스토그램 밝기 보정용(선택)

    # 사람 높이 버퍼(중간밴드)
    hbuf = deque(maxlen=300)
    last_auto = 0.0
    grid_frozen = False  # ⬅️ 최초엔 False → 한 번 빌드되면 True

    # 5) 루프
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok: break

        # (선택) 간단 감마 보정
        if gamma != 1.0:
            f = np.clip(((frame/255.0) ** (1.0/gamma))*255.0, 0, 255).astype(np.uint8)
            frame = f

        # 타일 추론
        crops, offs = split_tiles(frame, ROWS_TILES, COLS_TILES)
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

        # === A) 중간밴드 샘플 수집(20~40%) ===
        y_low  = int(MID_BAND[0] * fr_h)
        y_high = int(MID_BAND[1] * fr_h)
        if not grid_frozen:
            for x1,y1,x2,y2,cf in kept:
                if cf < 0.45: continue
                h = y2-y1; w = x2-x1
                if h < MIN_H_PX_FOR_SAMPLE or w <= 0: continue
                if (h/(w+1e-9)) < ASPECT_MIN: continue
                footx = 0.5*(x1+x2); footy = y2
                if y_low <= footy <= y_high and roi_mask[int(min(max(footy,0),fr_h-1)), int(min(max(footx,0),fr_w-1))] != 0:
                    hbuf.append(h)

        # === B) 자동 프리셋 선택(‘처음 측정’ 단계에서만) ===
        now = time.time()
        if (not grid_frozen) and (now - last_auto > AUTO_RECALC_SEC) and len(hbuf) >= 25:
            med_h = float(np.median(hbuf))
            r = med_h / fr_h
            if   r < 0.035: cur_preset = 0
            elif r < 0.055: cur_preset = 1
            elif r < 0.075: cur_preset = 2
            elif r < 0.095: cur_preset = 3
            else:           cur_preset = 4

            cell_side = max(MIN_CELL_PX,
                            min(MAX_CELL_PX, BASE_K_FROM_PERSON_H * med_h * PRESET_FACTORS[cur_preset] * fine_factor))
            if grid.build(cell_side) and FREEZE_AFTER_FIRST_BUILD:
                grid_frozen = True         # ⬅️ 최초 빌드 후 고정
            last_auto = now

        # === C) 수동 조정/재측정 ===
        k = cv2.waitKey(1) & 0xFF
        if k in (27, ord('q')): break
        if k in (ord('1'),ord('2'),ord('3'),ord('4'),ord('5')):
            cur_preset = int(chr(k)) - 1
            med_h = float(np.median(hbuf)) if len(hbuf)>0 else 60.0
            cell_side = max(MIN_CELL_PX,
                            min(MAX_CELL_PX, BASE_K_FROM_PERSON_H * med_h * PRESET_FACTORS[cur_preset] * fine_factor))
            grid.build(cell_side); grid_frozen = True
        elif k in (ord('+'), ord('=')):
            fine_factor *= 1.05
            med_h = float(np.median(hbuf)) if len(hbuf)>0 else 60.0
            cell_side = max(MIN_CELL_PX,
                            min(MAX_CELL_PX, BASE_K_FROM_PERSON_H * med_h * PRESET_FACTORS[cur_preset] * fine_factor))
            grid.build(cell_side); grid_frozen = True
        elif k in (ord('-'), ord('_')):
            fine_factor /= 1.05
            med_h = float(np.median(hbuf)) if len(hbuf)>0 else 60.0
            cell_side = max(MIN_CELL_PX,
                            min(MAX_CELL_PX, BASE_K_FROM_PERSON_H * med_h * PRESET_FACTORS[cur_preset] * fine_factor))
            grid.build(cell_side); grid_frozen = True
        elif k in (ord('r'), ord('R')):      # 재측정(초기 상태로 복구)
            hbuf.clear()
            last_auto = 0.0
            grid_frozen = False
        elif k in (ord('g'), ord('G')):      # 감마 ↓
            gamma = max(0.6, gamma/1.05)
        elif k in (ord('h'), ord('H')):      # 감마 ↑
            gamma = min(2.0, gamma*1.05)

        # === D) 셀 카운트(발끝이 ROI 안일 때만) ===
        counts = {}
        if grid.side is not None:
            for x1,y1,x2,y2,cf in kept:
                footx = 0.5*(x1+x2); footy = y2
                if roi_mask[int(min(max(footy,0),fr_h-1)), int(min(max(footx,0),fr_w-1))] == 0:
                    continue
                key = grid.locate(footx, footy)
                if key is None: continue
                counts[key] = counts.get(key, 0) + 1

        # === E) 렌더링 ===
        overlay = frame.copy()

        # 정사각형 셀(ROI 클립 안 함: CLIP_TO_ROI=False)
        if grid.side is not None:
            for key, poly in grid.cells.items():
                n = counts.get(key, 0)
                if n >= ALERT_THRESHOLD: color=(0,0,255)
                elif n>0:               color=(0,180,0)
                else:                   color=(180,180,180)
                if grid.side >= MIN_CELL_PX:
                    cv2.fillPoly(overlay, [poly], color)
                    cv2.polylines(overlay, [poly], True, (50,50,50), 1)

        if CLIP_TO_ROI:
            overlay = cv2.bitwise_and(overlay, overlay, mask=roi_mask)
        out = cv2.addWeighted(overlay, FILL_ALPHA, frame, 1.0-FILL_ALPHA, 0)

        cv2.polylines(out, [roi_poly], True, (0,200,255), 2)

        # 텍스트
        mode_txt = "heuristic freeze" if grid_frozen else "measuring..."
        cv2.putText(out, f"~10 m^2 cells ({mode_txt}), Alert >= {ALERT_THRESHOLD}",
                    (12,28), FONT, 0.7, (0,0,0), 3)
        cv2.putText(out, f"~10 m^2 cells ({mode_txt}), Alert >= {ALERT_THRESHOLD}",
                    (12,28), FONT, 0.7, (255,255,255), 1)

        med_txt = f"med_h={np.median(hbuf):.1f}px" if len(hbuf)>0 else "med_h=--"
        cv2.putText(out, f"Preset: {PRESET_NAMES[cur_preset]}  gamma={gamma:.2f}  {med_txt}",
                    (12,52), FONT, 0.7, (0,0,0), 3)
        cv2.putText(out, f"Preset: {PRESET_NAMES[cur_preset]}  gamma={gamma:.2f}  {med_txt}",
                    (12,52), FONT, 0.7, (255,255,255), 1)
        cv2.putText(out, "Manual: [1..5]=preset  +/-=size   [G]/[H]=gamma   R=remeasure",
                    (12,76), FONT, 0.7, (0,0,0), 3)
        cv2.putText(out, "Manual: [1..5]=preset  +/-=size   [G]/[H]=gamma   R=remeasure",
                    (12,76), FONT, 0.7, (255,255,255), 1)

        show_fit("Person-calibrated Grid Crowd Count", out)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
