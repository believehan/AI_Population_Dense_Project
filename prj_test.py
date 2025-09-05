# -*- coding: utf-8 -*-
# YouTube Live (streamlink) + YOLO11s + Homography crowd heatmap (4분할)
# 사전: streamlink 설치 및 경로 확인, pip install ultralytics opencv-python

import subprocess, shutil
import cv2
import numpy as np
from ultralytics import YOLO

# ===== 0) 입력 =====
YOUTUBE_URL = "https://www.youtube.com/watch?v=86-7Dr7yeVQ"
MODEL = "yolo11s.pt"
CONF = 0.25
VID_STRIDE = 1       # 프레임 스킵(원하면 2~3)
CELL_M = 0.5         # 지면 격자 해상도(미터 권장)
SMOOTH_SIGMA = 1.0   # 히트맵 가우시안 블러 시그마

# ===== 1) streamlink로 실제 스트림 URL 확보 =====
def resolve_stream(url: str) -> str:
    exe = (shutil.which("streamlink")
           or r"C:\Users\user\anaconda3\envs\py39\Scripts\streamlink.exe")
    cmd = [exe, "--stream-url", url, "best"]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True)
    stream_url = out.stdout.strip()
    if not stream_url:
        raise RuntimeError("streamlink로 스트림 URL을 얻지 못했습니다.")
    return stream_url

stream_url = resolve_stream(YOUTUBE_URL)
cap = cv2.VideoCapture(stream_url)
if not cap.isOpened():
    raise RuntimeError("VideoCapture 열기 실패")

# ===== 2) 첫 프레임에서 지면 기준점(픽셀) 찍기 =====
ok, first_frame = cap.read()
if not ok:
    raise RuntimeError("첫 프레임을 읽지 못했습니다.")

# 지면 측정 좌표(X,Y, m) — 당신 현장 기준으로 수정하세요 (픽셀 클릭과 1:1 대응, 4쌍 이상)
gnd_pts = np.array([
    [0.0, 0.0],
    [6.0, 0.0],
    [0.5, 4.0],
    [5.5, 4.0],
], dtype=np.float32)

img_pts = []
first_show = first_frame.copy()
def on_mouse(event, x, y, flags, param):
    global img_pts, first_show
    if event == cv2.EVENT_LBUTTONDOWN:
        img_pts.append([x, y])
        first_show = first_frame.copy()
        for px, py in img_pts:
            cv2.circle(first_show, (px, py), 6, (0,255,255), -1)
        cv2.putText(first_show, f"points: {len(img_pts)}  (Enter=OK, 'r'=reset)",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

cv2.namedWindow("pick 4+ ground points", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("pick 4+ ground points", on_mouse)

while True:
    disp = first_show.copy()
    cv2.putText(disp, "지면 기준점(픽셀) 4개 이상 클릭 → Enter 확정, r 리셋, ESC 종료",
                (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    cv2.imshow("pick 4+ ground points", disp)
    key = cv2.waitKey(30) & 0xFF
    if key == 27:  # ESC
        cv2.destroyAllWindows()
        cap.release()
        raise SystemExit
    if key == ord('r'):
        img_pts = []
        first_show = first_frame.copy()
    if key == 13 and len(img_pts) >= 4:  # Enter
        break

cv2.destroyWindow("pick 4+ ground points")
img_pts = np.array(img_pts, dtype=np.float32)
if len(img_pts) != len(gnd_pts):
    raise ValueError(f"gnd_pts({len(gnd_pts)})와 img_pts({len(img_pts)}) 개수가 같아야 합니다.")

# ===== 3) 호모그래피 계산 (이미지 → 지면) =====
H, _ = cv2.findHomography(img_pts, gnd_pts, method=0)
if H is None:
    raise RuntimeError("호모그래피 계산 실패. 점 대응을 확인하세요.")

def pixel_to_ground(x, y):
    p = np.array([x, y, 1.0], dtype=np.float64)
    a, b, c = H @ p
    return (a/c, b/c)

# ===== 4) 지면 격자 초기화 =====
minX, minY = gnd_pts.min(axis=0)
maxX, maxY = gnd_pts.max(axis=0)
grid_W = int(np.ceil((maxX - minX) / CELL_M))
grid_H = int(np.ceil((maxY - minY) / CELL_M))
crowd_map = np.zeros((grid_H, grid_W), dtype=np.float32)
def acc_to_grid(acc, X, Y):
    i = int((Y - minY) // CELL_M)  # row
    j = int((X - minX) // CELL_M)  # col
    if 0 <= i < acc.shape[0] and 0 <= j < acc.shape[1]:
        acc[i, j] += 1.0

# ===== 5) 모델 로드 =====
model = YOLO(MODEL)

# ===== 6) 실시간 루프 =====
while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break

    # 4분할
    h, w = frame.shape[:2]
    h2, w2 = h // 2, w // 2
    crops = [
        frame[0:h2,   0:w2],    # A
        frame[0:h2,   w2:w],    # B
        frame[h2:h,   0:w2],    # C
        frame[h2:h,   w2:w],    # D
    ]
    offsets = [(0,0), (w2,0), (0,h2), (w2,h2)]
    tags    = ["A","B","C","D"]

    # 한 번에 배치 추론 (리스트 입력 지원) → 리스트[Results]
    results = model(crops, conf=CONF)  # stream=False: 리스트로 반환

    vis_patches = []
    frame_map = np.zeros_like(crowd_map)

    for res, (ox, oy), tag in zip(results, offsets, tags):
        # 시각화 패치
        vis = res.plot()  # numpy annotated image
        cv2.putText(vis, tag, (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 2)
        vis_patches.append(vis)

        # "person"만 집계 (names/boxes API)
        boxes = res.boxes
        if boxes is None or len(boxes) == 0:
            continue
        names = res.names  # dict: id->name

        xyxy = boxes.xyxy.cpu().numpy()
        cls  = boxes.cls.cpu().numpy().astype(int)

        for b, c in zip(xyxy, cls):
            if names.get(c, "") != "person":
                continue
            x1, y1, x2, y2 = b
            # 각 크롭 좌표 → 전체 프레임 좌표로 보정
            foot_x = 0.5 * (x1 + x2) + ox
            foot_y = y2 + oy
            Xg, Yg = pixel_to_ground(foot_x, foot_y)
            acc_to_grid(frame_map, Xg, Yg)

    # 히트맵 업데이트 (EMA)
    if SMOOTH_SIGMA > 0:
        frame_map = cv2.GaussianBlur(frame_map, (0,0), SMOOTH_SIGMA)
    crowd_map = 0.90 * crowd_map + 0.10 * frame_map

    # 4패치 합치기
    top = cv2.hconcat([vis_patches[0], vis_patches[1]])
    bottom = cv2.hconcat([vis_patches[2], vis_patches[3]])
    combined = cv2.vconcat([top, bottom])

    # 히트맵 시각화
    norm = (crowd_map / (crowd_map.max() + 1e-6) * 255).astype(np.uint8)
    heat = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    heat_vis = cv2.resize(heat, (800, int(800 * heat.shape[0] / heat.shape[1])))

    cv2.imshow("Multi-Model Split View (batched with one model)", combined)
    cv2.imshow("BEV Heatmap", heat_vis)

    key = cv2.waitKey(1) & 0xFF
    if key in (27, ord('q')):  # ESC or q
        break

cap.release()
cv2.destroyAllWindows()
