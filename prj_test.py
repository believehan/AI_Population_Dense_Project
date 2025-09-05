# -*- coding: utf-8 -*-
# Multi-plane BEV grid: 평면(갈림길/인도/차도 등)별로 4점씩 잡아 각기 다른 H 사용
# 실행 중 'n' = 새 평면 추가, 'c' = 모든 평면 초기화 후 첫 평면부터 재등록

import os, json, subprocess
import numpy as np
import cv2
from ultralytics import YOLO

# ===== 입력/모델 =====
URL = "https://www.youtube.com/live/rnXIjl_Rzy4?si=p-3PLsiyI-wx4EbG"
STREAMLINK = r"C:\Users\user\anaconda3\envs\py39\Scripts\streamlink.exe"
MODEL = "yolo11s.pt"
CONF = 0.20
IOU  = 0.70
IMGSZ_TILE = 960
MAX_DET = 1000
OVERLAP = 40

# ===== 격자/표시 =====
N_COLS_INIT = 10
CELL_H_BEV = None          # None이면 ≈12행 자동
MIN_CELL_AREA_PX = 150
CELL_THRESH = 5
ALPHA = 0.25
COLOR_OK = (80,160,240)
COLOR_ALERT = (0,0,255)
DRAW_OUTLINE = False

# 카운팅 정책
USE_BILINEAR = True

CALIB_JSON = "calib/bev_meta_multi.json"
os.makedirs(os.path.dirname(CALIB_JSON), exist_ok=True)

# ---------- 유틸 ----------
def detect_near_is_bottom(H_inv, W_bev, H_bev, n_cols, row_h_bev):
    """BEV 상단/하단 한 셀씩 역투영해 면적이 더 큰 쪽을 근거리로 판단."""
    col_w = W_bev / max(1, n_cols)
    cx0 = (n_cols // 2) * col_w

    top = np.array([[cx0, 0],
                    [cx0+col_w, 0],
                    [cx0+col_w, min(row_h_bev, H_bev)],
                    [cx0,       min(row_h_bev, H_bev)]], np.float32)
    bot = np.array([[cx0, max(0, H_bev-row_h_bev)],
                    [cx0+col_w, max(0, H_bev-row_h_bev)],
                    [cx0+col_w, H_bev],
                    [cx0,       H_bev]], np.float32)

    area_top = cv2.contourArea(warp_points(H_inv, top).astype(np.float32))
    area_bot = cv2.contourArea(warp_points(H_inv, bot).astype(np.float32))
    return area_bot >= area_top   # True면 하단이 근거리, False면 상단이 근거리


def ensure_stream_url(url: str) -> str:
    r = subprocess.run([STREAMLINK, "--stream-url", url, "best"],
                       capture_output=True, text=True)
    s = r.stdout.strip()
    if not s: raise RuntimeError(f"streamlink 실패: {r.stderr}")
    return s

def warp_points(H, pts_xy):
    return cv2.perspectiveTransform(pts_xy.reshape(-1,1,2), H).reshape(-1,2)

class Clicker:
    def __init__(self, win, img):
        self.win = win; self.view = img.copy(); self.pts=[]
        cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
        cv2.imshow(win, self.view)
        cv2.setMouseCallback(win, self._on)
    def _on(self, event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.pts.append((x,y))
            cv2.circle(self.view, (x,y), 5, (0,255,0), -1)
            cv2.imshow(self.win, self.view)
    def get_n(self, n, prompt):
        print(prompt)
        while len(self.pts) < n:
            if cv2.waitKey(20) == 27: break
        return self.pts[-n:]

# ---------- 평면 1개 등록(4점) ----------
def add_one_plane(stream_url, target_size):
    """프레임 1장 캡처 → 동일 크기 창에서 4점 클릭 → 평면 dict 반환"""
    cap = cv2.VideoCapture(stream_url)
    for _ in range(30): cap.read()
    ok, frame = cap.read(); cap.release()
    if not ok: raise RuntimeError("프레임 캡처 실패")
    Wf, Hf = target_size
    frame_rs = cv2.resize(frame, (Wf, Hf))

    title = "Plane calib: 4 ground points (clockwise)"
    ck = Clicker(title, frame_rs)
    pts = ck.get_n(4, "[평면 추가] 같은 평면 4점을 '시계 방향'으로 클릭 (ESC=취소)")
    cv2.destroyWindow(title)
    if len(pts) < 4: raise SystemExit("평면 추가 중단: 4점 미선택")

    src = np.array(pts, np.float32)       # 원본 프레임 좌표
    roi_poly = src.astype(np.int32)       # 이 평면의 ROI(원본에서의 사변형)

    # BEV 크기(마주보는 변 평균)
    def seglen(a,b): return float(np.linalg.norm(a-b))
    W = int(round((seglen(src[1],src[0]) + seglen(src[2],src[3]))/2.0)); W=max(W,100)
    H = int(round((seglen(src[3],src[0]) + seglen(src[2],src[1]))/2.0)); H=max(H,100)

    dst = np.array([[0,0],[W-1,0],[W-1,H-1],[0,H-1]], np.float32)
    H_rect = cv2.getPerspectiveTransform(src, dst)
    H_inv  = np.linalg.inv(H_rect)

    plane = {
        "H_rect": H_rect.tolist(),
        "H_inv":  H_inv.tolist(),
        "bev_size": [W, H],
        "roi_poly": roi_poly.reshape(-1,2).tolist()  # 원본 프레임 좌표(사각형)
    }
    return plane

# ---------- 저장/로드 ----------
def save_planes(meta_path, input_size, planes):
    data = {"input_size": list(input_size), "planes": planes}
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"[Saved] {meta_path} (planes={len(planes)})")

def load_planes(meta_path, target_size):
    if not os.path.exists(meta_path): return None
    with open(meta_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if data.get("input_size") != list(target_size): return None
    return data["planes"]

# ---------- 전역 NMS ----------
def nms_numpy(boxes, scores, iou_thr=0.7):
    keep=[]; idxs=scores.argsort()[::-1]
    while idxs.size>0:
        i=idxs[0]; keep.append(i)
        if idxs.size==1: break
        rest=idxs[1:]
        xx1=np.maximum(boxes[i,0],boxes[rest,0])
        yy1=np.maximum(boxes[i,1],boxes[rest,1])
        xx2=np.minimum(boxes[i,2],boxes[rest,2])
        yy2=np.minimum(boxes[i,3],boxes[rest,3])
        w=np.maximum(0,xx2-xx1); h=np.maximum(0,yy2-yy1)
        inter=w*h
        area_i=(boxes[i,2]-boxes[i,0])*(boxes[i,3]-boxes[i,1])
        area_r=(boxes[rest,2]-boxes[rest,0])*(boxes[rest,3]-boxes[rest,1])
        iou=inter/(area_i+area_r-inter+1e-9)
        idxs=rest[iou<iou_thr]
    return np.array(keep,int)

# ---------- 격자 생성(평면별) ----------
def build_grid_polys_for_plane(plane, n_cols_init, cell_h_bev, min_cell_area_px):
    H_inv  = np.array(plane["H_inv"],  np.float64)
    W_bev, H_bev = plane["bev_size"]
    n_cols = n_cols_init
    col_w = W_bev / max(1, n_cols)
    row_h = H_bev/12.0 if cell_h_bev is None else float(cell_h_bev)

    polys=[]; gy=0; y0=0.0
    while y0 < H_bev:
        y1 = min(H_bev, y0+row_h)
        cx0=(n_cols//2)*col_w
        sample=np.array([[cx0,y0],[cx0+col_w,y0],[cx0+col_w,y1],[cx0,y1]], np.float32)
        sample_orig=warp_points(H_inv, sample).astype(np.float32)
        if cv2.contourArea(sample_orig) < min_cell_area_px: break
        for gx in range(n_cols):
            x0=gx*col_w; x1=(gx+1)*col_w
            bev_quad=np.array([[x0,y0],[x1,y0],[x1,y1],[x0,y1]], np.float32)
            orig_quad=warp_points(H_inv, bev_quad).astype(np.int32)
            polys.append((orig_quad, (gx,gy)))
        gy+=1; y0=y1

    plane["grid_polys"] = polys          # [(orig_quad, (gx,gy))]
    plane["n_cols"] = n_cols
    plane["n_rows"] = gy
    plane["row_h"]  = row_h
    return plane

# ---------- 평면 선택(발끝이 어느 평면에 속하나) ----------
def choose_plane_index_for_point(planes, pt_xy):
    # ROI 사변형 안에 들어가는 평면을 찾음(첫 번째 매치)
    ptx, pty = float(pt_xy[0]), float(pt_xy[1])
    for i, pl in enumerate(planes):
        poly = np.array(pl["roi_poly"], np.int32)
        inside = cv2.pointPolygonTest(poly, (ptx, pty), False)
        if inside >= 0:  # 내부 또는 경계
            return i
    return None

# ================= 메인 =================
def main():
    # 스트림/모델
    stream_url = ensure_stream_url(URL)
    model = YOLO(MODEL)

    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened(): raise RuntimeError("스트림 오픈 실패")
    ok, fr0 = cap.read()
    if not ok: raise RuntimeError("첫 프레임 실패")
    fr0 = cv2.resize(fr0, (1200, 600))
    Hf, Wf = fr0.shape[:2]
    w2, h2 = Wf//2, Hf//2

    # 평면 로드(없으면 1개 등록)
    planes = load_planes(CALIB_JSON, (Wf, Hf))
    if planes is None or len(planes)==0:
        planes = [add_one_plane(stream_url, (Wf, Hf))]
        save_planes(CALIB_JSON, (Wf, Hf), planes)

    # 평면별 격자 미리 셋업
    for i in range(len(planes)):
        planes[i] = build_grid_polys_for_plane(
            planes[i], N_COLS_INIT, CELL_H_BEV, MIN_CELL_AREA_PX
        )
        print(f"[Plane {i}] cols={planes[i]['n_cols']}, rows={planes[i]['n_rows']}")

    # 타일 정의
    tiles = [
        (0,            0,            w2+OVERLAP, h2+OVERLAP),
        (w2-OVERLAP,   0,            Wf,         h2+OVERLAP),
        (0,            h2-OVERLAP,   w2+OVERLAP, Hf),
        (w2-OVERLAP,   h2-OVERLAP,   Wf,         Hf),
    ]

    while True:
        ok, frame = cap.read()
        if not ok: break
        frame = cv2.resize(frame, (Wf, Hf))

        # --- 타일 추론(사람만) ---
        boxes_all, conf_all = [], []
        for (x1,y1,x2,y2) in tiles:
            crop = frame[y1:y2, x1:x2]
            r = model(crop, classes=[0], conf=CONF, iou=IOU,
                      imgsz=IMGSZ_TILE, max_det=MAX_DET)[0]
            if r.boxes is None or r.boxes.xyxy is None: continue
            b = r.boxes.xyxy.cpu().numpy()
            s = r.boxes.conf.cpu().numpy()
            b[:,[0,2]] += x1; b[:,[1,3]] += y1
            boxes_all.append(b); conf_all.append(s)

        base = frame.copy()
        boxes = None
        if boxes_all:
            boxes = np.concatenate(boxes_all, 0)
            scores= np.concatenate(conf_all, 0)
            keep = nms_numpy(boxes, scores, iou_thr=IOU)
            boxes = boxes[keep]

        # --- 평면별 카운트 초기화 ---
        for pl in planes:
            pl["counts"] = np.zeros((pl["n_rows"], pl["n_cols"]), dtype=float)

        # --- 발끝을 해당 평면에 배정 → 각 평면의 BEV로 카운트 ---
        if boxes is not None:
            for (x1,y1,x2,y2) in boxes.astype(int):
                cx = 0.5*(x1+x2); foot = (cx, y2)

                idx = choose_plane_index_for_point(planes, foot)
                if idx is None:  # 어느 평면 ROI에도 안 들어가면 스킵
                    continue
                pl = planes[idx]
                H_rect = np.array(pl["H_rect"], np.float64)
                W_bev, H_bev = pl["bev_size"]
                n_cols, n_rows, row_h = pl["n_cols"], pl["n_rows"], pl["row_h"]

                bev = warp_points(H_rect, np.array([[foot]], np.float32))[0]  # (x',y')
                if bev[0] < 0 or bev[0] >= W_bev or bev[1] < 0 or bev[1] >= H_bev:
                    continue

                col_w = W_bev / max(1, n_cols)
                if USE_BILINEAR:
                    gx = bev[0] / col_w
                    gy = bev[1] / row_h
                    ix = int(np.floor(gx)); fx = gx - ix
                    iy = int(np.floor(gy)); fy = gy - iy
                    ix = np.clip(ix, 0, n_cols-1); iy = np.clip(iy, 0, n_rows-1)
                    w00=(1-fx)*(1-fy); w10=fx*(1-fy); w01=(1-fx)*fy; w11=fx*fy
                    pl["counts"][iy, ix] += w00
                    if ix+1<n_cols: pl["counts"][iy, ix+1] += w10
                    if iy+1<n_rows: pl["counts"][iy+1, ix] += w01
                    if iy+1<n_rows and ix+1<n_cols: pl["counts"][iy+1, ix+1] += w11
                else:
                    ix = int(np.clip(bev[0] / col_w, 0, n_cols-1))
                    iy = int(np.clip(bev[1] / row_h,  0, n_rows-1))
                    pl["counts"][iy, ix] += 1.0

        # --- 격자 오버레이(먼저 칠함) ---
        overlay = base.copy()
        alert_any = False
        for (poly, (gx, gy)) in pl["grid_polys"]:
            c = pl["counts"][gy, gx] if gy < pl["counts"].shape[0] and gx < pl["counts"].shape[1] else 0.0
            c_int = int(round(c))                               # ← 정수화
            color = COLOR_ALERT if c_int >= CELL_THRESH else COLOR_OK
            if c_int >= CELL_THRESH:
                alert_any = True

            if DRAW_OUTLINE:
                cv2.polylines(overlay, [poly], True, color, 2)
            else:
                cv2.fillConvexPoly(overlay, poly, color)

            cen = poly.reshape(-1,2).mean(axis=0).astype(int)
            cv2.putText(overlay, str(c_int), (cen[0]-8, cen[1]+8),  # ← 정수 표시
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)


            # 평면 ROI 테두리(디버그용, 얇게)
            roi = np.array(pl["roi_poly"], np.int32)
            cv2.polylines(overlay, [roi], True, (0,255,0), 1)

        vis = cv2.addWeighted(overlay, ALPHA, base, 1-ALPHA, 0)

        # --- 박스는 마지막에 ---
        if boxes is not None:
            for (x1,y1,x2,y2) in boxes.astype(int):
                cv2.rectangle(vis, (x1,y1), (x2,y2), (0,200,255), 2)
        if alert_any:
            cv2.putText(vis, "경고: 셀 임계 초과!", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_ALERT, 3, cv2.LINE_AA)

        cv2.imshow("Multi-plane BEV grid (press n: add plane, c: reset)", vis)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break

        if key == ord('n'):
            # 새 평면 추가
            plane = add_one_plane(stream_url, (Wf, Hf))
            plane = build_grid_polys_for_plane(plane, N_COLS_INIT, CELL_H_BEV, MIN_CELL_AREA_PX)
            planes.append(plane)
            save_planes(CALIB_JSON, (Wf,Hf), planes)
            print(f"[Plane add] total planes = {len(planes)}")

        if key == ord('c'):
            # 모든 평면 초기화 후 첫 평면부터
            planes = [add_one_plane(stream_url, (Wf, Hf))]
            planes[0] = build_grid_polys_for_plane(planes[0], N_COLS_INIT, CELL_H_BEV, MIN_CELL_AREA_PX)
            save_planes(CALIB_JSON, (Wf,Hf), planes)
            print(f"[Plane reset] planes = 1")

    cap.release(); cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
