# -*- coding: utf-8 -*-
# YouTube(streamlink) + YOLO11s + Homography + ROI(10 m² grid)
# + Global NMS (tile dedup) + Threshold Red Overlay + Auto-fit display
#
# 요구:
#   pip install ultralytics opencv-python streamlink
#
# 사용:
# 1) "Pick ground points (>=4)" 창에서 gnd_pts 순서대로 지면 4점 클릭 → Enter
# 2) "Draw ROI polygon" 창에서 바닥 ROI 다각형 클릭(3점 이상) → Enter
# 3) 메인 창에서 셀(≈10 m²) 카운트 및 기준 초과(빨강) 확인

import subprocess, shutil
import cv2
import numpy as np
from ultralytics import YOLO

# ========= 0) 기본 설정 =========
YOUTUBE_URL = "https://www.youtube.com/live/rnXIjl_Rzy4?feature=shared"  # 라이브 URL 교체
MODEL   = "yolo11s.pt"
CONF    = 0.30
IMGSZ   = 512          # 작게=빠름, 크게=정확↑
DEVICE  = "cpu"        # GPU 있으면 "cuda:0"
USE_HALF= False        # FP16은 GPU에서만
ROWS, COLS = 2, 4      # 8-타일(2x4)
TARGET_CELL_AREA_M2 = 10.0   # 셀 면적 목표(≈10 m²)
ALERT_THRESHOLD = 5          # 셀별 인원수 기준(≥ 빨강)
FILL_ALPHA = 0.45            # 오버레이 투명도(0~1)
DRAW_COUNTS = True           # 셀 숫자 표시

def gnd2pix_safe(X, Y, Hinv, frame_w, frame_h, eps=1e-7, margin=5):
    """지면→픽셀 역투영의 안전 버전: 수치폭주/프레임밖을 None 처리"""
    q = np.array([X, Y, 1.0], np.float64)
    a, b, c = Hinv @ q
    if abs(c) < eps:  # 지평선 근처
        return None
    x = a / c
    y = b / c
    if not np.isfinite(x) or not np.isfinite(y):
        return None
    # 프레임 조금 밖(margin)까지 허용, 그 외는 None
    if x < -margin or x > frame_w + margin or y < -margin or y > frame_h + margin:
        return None
    return (int(round(x)), int(round(y)))

def order_poly_ccw(pts):
    """2D 점들을 중심각 기준으로 CCW 정렬"""
    pts = np.asarray(pts, dtype=np.float32)
    cx, cy = pts.mean(axis=0)
    ang = np.arctan2(pts[:,1] - cy, pts[:,0] - cx)
    idx = np.argsort(ang)
    return pts[idx].astype(np.int32)

def is_convex_poly(pts):
    """간단한 볼록성 검사(교차곱 부호 일관성)"""
    p = np.asarray(pts, dtype=np.float32)
    n = len(p)
    sgn = 0
    for i in range(n):
        a = p[(i+1)%n] - p[i]
        b = p[(i+2)%n] - p[(i+1)%n]
        z = a[0]*b[1] - a[1]*b[0]
        if z == 0: 
            continue
        if sgn == 0:
            sgn = 1 if z > 0 else -1
        else:
            if (z > 0 and sgn < 0) or (z < 0 and sgn > 0):
                return False
    return True


# ----- 화면 표시 유틸 (자동 리사이즈) -----
DISPLAY_MAX_W = 1280   # 최대 가로
DISPLAY_MAX_H = 720    # 최대 세로
def resize_to_fit(img, max_w=DISPLAY_MAX_W, max_h=DISPLAY_MAX_H):
    h, w = img.shape[:2]
    s = min(max_w / w, max_h / h, 1.0)  # 확대는 안 함
    return cv2.resize(img, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA) if s < 1.0 else img

def show_fit(win_name, img):
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    view = resize_to_fit(img)
    cv2.imshow(win_name, view)

def compute_fit_scale(img_w, img_h, max_w=DISPLAY_MAX_W, max_h=DISPLAY_MAX_H):
    """imshow할 때 사용한 비율을 재현(클릭 좌표 역변환에 사용)"""
    return min(max_w / img_w, max_h / img_h, 1.0)

FONT = cv2.FONT_HERSHEY_SIMPLEX

# ========= 1) 유튜브 라이브 열기 (streamlink) =========
def resolve_stream(url: str) -> str:
    exe = (shutil.which("streamlink")
           or r"C:\Users\user\anaconda3\envs\py39\Scripts\streamlink.exe")  # 환경에 맞게 수정
    if not exe:
        raise RuntimeError("streamlink 실행파일을 찾을 수 없습니다.")
    out = subprocess.run([exe, "--stream-url", url, "best"],
                         capture_output=True, text=True, check=True)
    s = out.stdout.strip()
    if not s:
        raise RuntimeError("streamlink로 스트림 URL 얻기 실패")
    return s

stream_url = resolve_stream(YOUTUBE_URL)
cap = cv2.VideoCapture(stream_url)
if not cap.isOpened():
    raise RuntimeError("VideoCapture 열기 실패")

# ========= 2) 첫 프레임 =========
ok, first = cap.read()
if not ok:
    raise RuntimeError("첫 프레임 읽기 실패")

# ========= 3) 호모그래피: 지면 4점(픽셀) ↔ 지면 좌표(미터) =========
# gnd_pts는 '미터' 단위로 4점(예: 8m x 6m 사각형)
gnd_pts = np.array([
    [0.0, 0.0],
    [8.0, 0.0],
    [8.0, 6.0],
    [0.0, 6.0],
], dtype=np.float32)

# ----- 선택 제약 옵션 -----
ENFORCE_CONVEX          = True   # 볼록(자기 교차 없음) 강제
ENFORCE_TRAPEZOID_LIKE  = False  # 사다리꼴 유사(위/아래 변 거의 평행) 강제 (기본 꺼짐 권장)
PARALLEL_TOL_DEG        = 18.0   # 평행 허용 오차(도)

# ----- 도우미 함수들 -----
def order_quad_TL_TR_BR_BL(pts4):
    """임의 4점을 TL,TR,BR,BL 순서로 재배열"""
    pts = np.array(pts4, dtype=np.float32)
    # 1) y로 위/아래 두 점씩 분리
    idx = np.argsort(pts[:,1])
    top2  = pts[idx[:2]]; bot2 = pts[idx[2:]]
    # 2) 각 쌍을 x로 좌/우 정렬
    TL, TR = top2[np.argsort(top2[:,0])]
    BL, BR = bot2[np.argsort(bot2[:,0])]
    ordered = np.array([TL, TR, BR, BL], dtype=np.float32)
    return ordered

def is_convex_quad(ordered):
    """TL,TR,BR,BL 순 입력의 볼록성 검사(부호 일관 교차곱)"""
    p = ordered
    v = [p[(i+1)%4] - p[i] for i in range(4)]
    z = []
    for i in range(4):
        # 2D cross of consecutive edges
        a = v[i]; b = v[(i+1)%4]
        z.append(a[0]*b[1] - a[1]*b[0])
    # 모두 같은 부호(0은 허용)여야 함
    pos = sum(zz > 0 for zz in z); neg = sum(zz < 0 for zz in z)
    return not (pos > 0 and neg > 0)

def angle_deg(v):
    return np.degrees(np.arctan2(v[1], v[0] + 1e-12))

def wrap180(a):
    a = (a + 180.0) % 360.0 - 180.0
    return a

def is_trapezoid_like(ordered, tol_deg=PARALLEL_TOL_DEG):
    """위/아래 변이 거의 평행이면 True (사다리꼴 유사)"""
    TL, TR, BR, BL = ordered
    v_top    = TR - TL
    v_bottom = BR - BL
    d = abs(wrap180(angle_deg(v_top) - angle_deg(v_bottom)))
    return d <= tol_deg

# ----- 클릭(리사이즈 보정 포함) -----
img_pts = []
first_show = first.copy()
H_SCALE = compute_fit_scale(first_show.shape[1], first_show.shape[0])  # 표시 스케일

def on_mouse_h(event, x, y, flags, param):
    global img_pts, first_show
    if event == cv2.EVENT_LBUTTONDOWN:
        # 클릭 좌표를 원본 좌표로 역변환
        ox = int(round(x / H_SCALE))
        oy = int(round(y / H_SCALE))
        img_pts.append([ox, oy])

        # 화면에 갱신
        first_show = first.copy()
        for px, py in img_pts:
            cv2.circle(first_show, (px, py), 6, (0,255,255), -1)
        cv2.putText(first_show, f"H pts: {len(img_pts)} (Enter=OK, r=reset)",
                    (20, 40), FONT, 0.7, (0,255,255), 2)

cv2.namedWindow("Pick ground points (4 only)", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Pick ground points (4 only)", on_mouse_h)

while True:
    disp = cv2.resize(first_show,
                      (int(first_show.shape[1]*H_SCALE), int(first_show.shape[0]*H_SCALE)),
                      interpolation=cv2.INTER_AREA)
    cv2.putText(disp, "지면 기준점 정확히 4개 클릭 → Enter (사변형만 허용)",
                (20, 20), FONT, 0.6, (255,255,255), 2)
    if ENFORCE_TRAPEZOID_LIKE:
        cv2.putText(disp, f"* 사다리꼴 유사 강제: 위/아래 변 평행 허용오차 {PARALLEL_TOL_DEG:.0f}°",
                    (20, 46), FONT, 0.55, (0,255,255), 2)
    cv2.imshow("Pick ground points (4 only)", disp)

    k = cv2.waitKey(30) & 0xFF
    if k == 27:
        cv2.destroyAllWindows(); cap.release(); raise SystemExit
    if k == ord('r'):
        img_pts = []; first_show = first.copy()

    if k == 13 and len(img_pts) >= 4:
        # 정확히 4개만 사용
        pts4 = np.array(img_pts[:4], dtype=np.float32)

        # 1) TL,TR,BR,BL로 자동 정렬
        ordered = order_quad_TL_TR_BR_BL(pts4)

        # 2) 유효성 검사
        ok_shape = True
        msg = ""
        if ENFORCE_CONVEX and not is_convex_quad(ordered):
            ok_shape = False
            msg = "사각형이 볼록이 아닙니다. 다시 찍어주세요."
        if ok_shape and ENFORCE_TRAPEZOID_LIKE and not is_trapezoid_like(ordered):
            ok_shape = False
            msg = "사다리꼴 형태가 아닙니다(위/아래 변 평행 아님). 다시 찍어주세요."

        # 3) 통과하면 확정, 아니면 경고 후 리셋
        if ok_shape:
            img_pts = ordered
            break
        else:
            warn = first.copy()
            cv2.putText(warn, msg, (20, 60), FONT, 0.9, (0,0,255), 3)
            cv2.putText(warn, "R키로 초기화 후 다시 4점 클릭", (20, 96), FONT, 0.8, (0,0,255), 2)
            wdisp = cv2.resize(warn,
                               (int(warn.shape[1]*H_SCALE), int(warn.shape[0]*H_SCALE)),
                               interpolation=cv2.INTER_AREA)
            cv2.imshow("Pick ground points (4 only)", wdisp)
            cv2.waitKey(1000)  # 1초 메시지 표시
            img_pts = []; first_show = first.copy()

cv2.destroyWindow("Pick ground points (4 only)")

# 최종 img_pts는 TL,TR,BR,BL 순서
img_pts = np.array(img_pts, dtype=np.float32)

# 호모그래피 계산
H, _ = cv2.findHomography(img_pts, gnd_pts, method=0)
if H is None:
    raise RuntimeError("호모그래피 계산 실패(점 대응을 확인하세요)")
Hinv = np.linalg.inv(H)

def pix2gnd(x, y):
    """픽셀(x,y) -> 지면(X,Y, m)"""
    p = np.array([x, y, 1.0], np.float64)
    a,b,c = H @ p
    return (a/c, b/c)

def gnd2pix(X, Y):
    """지면(X,Y, m) -> 픽셀(x,y)"""
    q = np.array([X, Y, 1.0], np.float64)
    a,b,c = Hinv @ q
    return (int(round(a/c)), int(round(b/c)))

# ========= 4) ROI 다각형(픽셀) → 지면 다각형 (클릭 보정 포함) =========
roi_img = []
roi_show = first.copy()
ROI_SCALE = compute_fit_scale(roi_show.shape[1], roi_show.shape[0])  # 표시 스케일

def on_mouse_roi(event, x, y, flags, param):
    global roi_img, roi_show
    if event == cv2.EVENT_LBUTTONDOWN:
        # 클릭 좌표를 원본 좌표로 역변환
        ox = int(round(x / ROI_SCALE))
        oy = int(round(y / ROI_SCALE))
        roi_img.append([ox, oy])

        roi_show = first.copy()
        for i, (px,py) in enumerate(roi_img):
            cv2.circle(roi_show, (px,py), 4, (0,200,255), -1)
            if i > 0:
                cv2.line(roi_show, tuple(roi_img[i-1]), (px,py), (0,200,255), 2)
        cv2.putText(roi_show, f"ROI pts: {len(roi_img)} (Enter=close, r=reset)",
                    (20,40), FONT, 0.7, (0,200,255), 2)

cv2.namedWindow("Draw ROI polygon", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Draw ROI polygon", on_mouse_roi)
while True:
    disp = cv2.resize(roi_show,
                      (int(roi_show.shape[1]*ROI_SCALE), int(roi_show.shape[0]*ROI_SCALE)),
                      interpolation=cv2.INTER_AREA)
    cv2.putText(disp, "카운트할 바닥 ROI를 클릭(3점 이상) → Enter",
                (20,20), FONT, 0.6, (255,255,255), 2)
    cv2.imshow("Draw ROI polygon", disp)
    k = cv2.waitKey(30) & 0xFF
    if k == 27:
        cv2.destroyAllWindows(); cap.release(); raise SystemExit
    if k == ord('r'):
        roi_img = []; roi_show = first.copy()
    if k == 13 and len(roi_img) >= 3:
        break
cv2.destroyWindow("Draw ROI polygon")

roi_img = np.array(roi_img, dtype=np.float32)
roi_gnd = np.array([pix2gnd(px,py) for (px,py) in roi_img], dtype=np.float32)

# ========= 5) ROI를 10 m² 격자로 나누기 =========
cell_side = float(np.sqrt(TARGET_CELL_AREA_M2))  # ≈3.162 m
minX, minY = roi_gnd.min(axis=0)
maxX, maxY = roi_gnd.max(axis=0)
W = int(np.ceil((maxX - minX) / cell_side))
Hc= int(np.ceil((maxY - minY) / cell_side))

def point_in_poly(x, y, poly):
    """레이캐스팅: 점이 다각형 내부인지"""
    inside = False
    n = len(poly); j = n - 1
    for i in range(n):
        xi, yi = poly[i]; xj, yj = poly[j]
        intersect = ((yi > y) != (yj > y)) and (x < (xj - xi)*(y - yi)/(yj - yi + 1e-12) + xi)
        if intersect: inside = not inside
        j = i
    return inside

# 프레임 크기
fr_h, fr_w = first.shape[:2]

cell_mask = np.zeros((Hc, W), dtype=bool)
cell_polys_px = [[None]*W for _ in range(Hc)]

for i in range(Hc):
    for j in range(W):
        cx = minX + (j + 0.5) * cell_side
        cy = minY + (i + 0.5) * cell_side
        if not point_in_poly(cx, cy, roi_gnd):
            continue

        # 지면 사각형의 네 꼭짓점(좌표계 순서)
        x0 = minX + j*cell_side; x1 = x0 + cell_side
        y0 = minY + i*cell_side; y1 = y0 + cell_side
        corners_g = [(x0,y0), (x1,y0), (x1,y1), (x0,y1)]

        # 안전 역투영 + 정렬
        corners_px = []
        for (X, Y) in corners_g:
            pt = gnd2pix_safe(X, Y, Hinv, fr_w, fr_h)
            if pt is None:
                corners_px = None
                break
            corners_px.append(pt)

        if corners_px is None:
            continue  # 폭주/프레임밖 → 이 셀은 스킵

        poly = order_poly_ccw(corners_px)
        # (선택) 볼록성 최종 보정
        if not is_convex_poly(poly):
            hull = cv2.convexHull(poly)
            if len(hull) < 4:
                continue  # 너무 비정상 → 스킵
            poly = hull[:,0,:]  # Nx1x2 → Nx2

        cell_mask[i, j] = True
        cell_polys_px[i][j] = poly.astype(np.int32)


# ========= 6) YOLO 로드 =========
model = YOLO(MODEL)

# ========= 7) 8-타일 분할 =========
def split_tiles(frame, rows=ROWS, cols=COLS):
    h, w = frame.shape[:2]
    hs = [int(round(r*h/rows)) for r in range(rows+1)]
    ws = [int(round(c*w/cols)) for c in range(cols+1)]
    crops, offs = [], []
    for r in range(rows):
        for c in range(cols):
            y1,y2 = hs[r], hs[r+1]; x1,x2 = ws[c], ws[c+1]
            crops.append(frame[y1:y2, x1:x2])
            offs.append((x1, y1))
    return crops, offs

# ========= 8) Global NMS (타일 통합 중복 제거) =========
def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a[:4]; bx1, by1, bx2, by2 = b[:4]
    inter_x1 = max(ax1, bx1); inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2); inter_y2 = min(ay2, by2)
    iw = max(0.0, inter_x2 - inter_x1); ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    if inter <= 0: return 0.0
    area_a = max(0.0, (ax2-ax1)) * max(0.0, (ay2-ay1))
    area_b = max(0.0, (bx2-bx1)) * max(0.0, (by2-by1))
    return inter / (area_a + area_b - inter + 1e-9)

def nms_global(boxes, iou_th=0.55):
    """boxes: [x1,y1,x2,y2,conf] in 원본 프레임 좌표"""
    boxes = sorted(boxes, key=lambda x: x[4], reverse=True)
    keep = []
    while boxes:
        cur = boxes.pop(0)
        keep.append(cur)
        boxes = [b for b in boxes if iou_xyxy(cur, b) < iou_th]
    return keep

# ========= 9) 메인 루프 =========
while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break

    # 8-타일 배치 추론 (리스트 입력 → Results 리스트)
    crops, offs = split_tiles(frame, ROWS, COLS)
    results = model(crops, conf=CONF, imgsz=IMGSZ, device=DEVICE, half=USE_HALF, classes=[0])

    # (A) 전-타일 박스를 원본 좌표로 모으기
    all_boxes = []  # [x1,y1,x2,y2,conf]
    for res, (ox, oy) in zip(results, offs):
        boxes = res.boxes
        if boxes is None or len(boxes) == 0:
            continue
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        for (x1, y1, x2, y2), cf in zip(xyxy, conf):
            all_boxes.append([x1 + ox, y1 + oy, x2 + ox, y2 + oy, float(cf)])

    # (B) 타일 통합 NMS
    kept = nms_global(all_boxes, iou_th=0.55)  # 0.5~0.6에서 조정

    # (C) kept만 카운트: 발점→지면→셀
    counts = np.zeros((Hc, W), dtype=np.int32)
    for x1, y1, x2, y2, cf in kept:
        footx = 0.5 * (x1 + x2)
        footy = y2
        Xg, Yg = pix2gnd(footx, footy)
        j = int((Xg - minX) // cell_side)
        i = int((Yg - minY) // cell_side)
        if 0 <= i < Hc and 0 <= j < W and cell_mask[i, j]:
            counts[i, j] += 1

    # (D) 기준 초과 셀 오버레이
    overlay = frame.copy()
    for i in range(Hc):
        for j in range(W):
            if not cell_mask[i, j]:
                continue
            poly = cell_polys_px[i][j]
            if poly is None:
                continue
            n = counts[i, j]
            if n >= ALERT_THRESHOLD:
                color = (0, 0, 255)      # 빨강
            elif n > 0:
                color = (0, 180, 0)      # 초록
            else:
                color = (180, 180, 180)  # 회색
            cv2.fillPoly(overlay, [poly], color)
            cv2.polylines(overlay, [poly], isClosed=True, color=(50,50,50), thickness=1)

            if DRAW_COUNTS:
                cx = minX + (j + 0.5) * cell_side
                cy = minY + (i + 0.5) * cell_side
                px, py = gnd2pix(cx, cy)
                cv2.putText(overlay, str(int(n)), (px-6, py+6), FONT, 0.6, (0,0,0), 2)
                cv2.putText(overlay, str(int(n)), (px-6, py+6), FONT, 0.6, (255,255,255), 1)

    out = cv2.addWeighted(overlay, FILL_ALPHA, frame, 1.0 - FILL_ALPHA, 0)

    # ROI 윤곽(확인용)
    roi_np = roi_img.astype(int)
    for i in range(len(roi_np)):
        p1 = tuple(roi_np[i]); p2 = tuple(roi_np[(i+1) % len(roi_np)])
        cv2.line(out, p1, p2, (0, 200, 255), 2)

    # 안내 텍스트
    cv2.putText(out, f"Cell ~{TARGET_CELL_AREA_M2:.0f} m^2, Alert >= {ALERT_THRESHOLD} persons",
                (12, 28), FONT, 0.7, (0,0,0), 3)
    cv2.putText(out, f"Cell ~{TARGET_CELL_AREA_M2:.0f} m^2, Alert >= {ALERT_THRESHOLD} persons",
                (12, 28), FONT, 0.7, (255,255,255), 1)

    show_fit("ROI Grid Count (threshold overlay)", out)

    if cv2.waitKey(1) & 0xFF in (27, ord('q')):
        break

cap.release()
cv2.destroyAllWindows()
