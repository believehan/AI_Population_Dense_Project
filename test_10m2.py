# -*- coding: utf-8 -*-
# YouTube(streamlink) + YOLO11s + Homography + ROI(10 m² grid)
# + Ground-line auto alignment + Vertical-VP depth alignment
# + Global NMS + ROI mask clip + Auto-fit display

import subprocess, shutil
import cv2
import numpy as np
from ultralytics import YOLO

# ========= 0) 기본 설정 =========
YOUTUBE_URL = "https://www.youtube.com/live/rnXIjl_Rzy4?feature=shared"
MODEL   = "yolo11s.pt"
CONF    = 0.30
IMGSZ   = 512
DEVICE  = "cpu"        # GPU면 "cuda:0"
USE_HALF= False
ROWS, COLS = 2, 4
TARGET_CELL_AREA_M2 = 10.0
ALERT_THRESHOLD = 5
FILL_ALPHA = 0.45
DRAW_COUNTS = True
AUTO_ALIGN_FROM_LINE = True   # 4.5: 바닥 기준선 2점
USE_VERTICAL_VP      = True   # 4.55: 수직 소실점으로 깊이축 정렬

# ----- 화면 표시 유틸 -----
DISPLAY_MAX_W = 1280
DISPLAY_MAX_H = 720
FONT = cv2.FONT_HERSHEY_SIMPLEX

def resize_to_fit(img, max_w=DISPLAY_MAX_W, max_h=DISPLAY_MAX_H):
    h, w = img.shape[:2]
    s = min(max_w / w, max_h / h, 1.0)
    return cv2.resize(img, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA) if s < 1.0 else img

def show_fit(win_name, img):
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.imshow(win_name, resize_to_fit(img))

def compute_fit_scale(img_w, img_h, max_w=DISPLAY_MAX_W, max_h=DISPLAY_MAX_H):
    return min(max_w / img_w, max_h / img_h, 1.0)

# ========= 1) 유튜브 열기 =========
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

stream_url = resolve_stream(YOUTUBE_URL)
cap = cv2.VideoCapture(stream_url)
if not cap.isOpened():
    raise RuntimeError("VideoCapture 열기 실패")

# ========= 2) 첫 프레임 =========
ok, first = cap.read()
if not ok:
    raise RuntimeError("첫 프레임 읽기 실패")
fr_h, fr_w = first.shape[:2]
H_SCALE = compute_fit_scale(fr_w, fr_h)

# ========= 2.5) 기하 보조 =========
def gnd2pix_safe(X, Y, Hinv, frame_w, frame_h, eps=1e-7, margin=5):
    q = np.array([X, Y, 1.0], np.float64)
    a, b, c = Hinv @ q
    if abs(c) < eps:
        return None
    x = a / c; y = b / c
    if not np.isfinite(x) or not np.isfinite(y):
        return None
    if x < -margin or x > frame_w + margin or y < -margin or y > frame_h + margin:
        return None
    return (int(round(x)), int(round(y)))

def order_poly_ccw(pts):
    pts = np.asarray(pts, dtype=np.float32)
    cx, cy = pts.mean(axis=0)
    ang = np.arctan2(pts[:,1]-cy, pts[:,0]-cx)
    idx = np.argsort(ang)
    return pts[idx].astype(np.int32)

def reorder_ccw_pts(pts):
    """다각형 꼭짓점 CCW 재정렬(결과 float32)"""
    pts = np.asarray(pts, dtype=np.float32)
    c = pts.mean(axis=0)
    ang = np.arctan2(pts[:,1]-c[1], pts[:,0]-c[0])
    idx = np.argsort(ang)
    return pts[idx].astype(np.float32)

def is_convex_poly(pts):
    p = np.asarray(pts, dtype=np.float32); n=len(p); sgn=0
    for i in range(n):
        a = p[(i+1)%n] - p[i]; b = p[(i+2)%n] - p[(i+1)%n]
        z = a[0]*b[1] - a[1]*b[0]
        if z == 0: continue
        if sgn == 0: sgn = 1 if z > 0 else -1
        else:
            if (z > 0 and sgn < 0) or (z < 0 and sgn > 0): return False
    return True

# H 클릭 제약
ENFORCE_CONVEX=True; ENFORCE_TRAPEZOID_LIKE=False; PARALLEL_TOL_DEG=18.0
def order_quad_TL_TR_BR_BL(pts4):
    pts = np.array(pts4, dtype=np.float32)
    idx = np.argsort(pts[:,1])
    top2  = pts[idx[:2]]; bot2 = pts[idx[2:]]
    TL, TR = top2[np.argsort(top2[:,0])]
    BL, BR = bot2[np.argsort(bot2[:,0])]
    return np.array([TL, TR, BR, BL], dtype=np.float32)
def is_convex_quad(ordered):
    p=ordered; v=[p[(i+1)%4]-p[i] for i in range(4)]
    z=[]; 
    for i in range(4):
        a,b=v[i],v[(i+1)%4]; z.append(a[0]*b[1]-a[1]*b[0])
    pos=sum(zz>0 for zz in z); neg=sum(zz<0 for zz in z)
    return not (pos>0 and neg>0)
def angle_deg(v): return np.degrees(np.arctan2(v[1], v[0] + 1e-12))
def wrap180(a):  return (a + 180.0) % 360.0 - 180.0
def is_trapezoid_like(ordered, tol_deg=PARALLEL_TOL_DEG):
    TL,TR,BR,BL=ordered; v_top=TR-TL; v_bottom=BR-BL
    d=abs(wrap180(angle_deg(v_top)-angle_deg(v_bottom)))
    return d<=tol_deg

# 좌표계 회전
def rotate_ground(H, theta_deg):
    th=np.deg2rad(theta_deg)
    R=np.array([[np.cos(th),-np.sin(th),0],
                [np.sin(th), np.cos(th),0],
                [0,0,1]], dtype=np.float64)
    H2=R@H
    return H2, np.linalg.inv(H2)

def apply_ground_rot_to_points(pts_gnd, theta_deg):
    th=np.deg2rad(theta_deg); c,s=np.cos(th),np.sin(th)
    R2=np.array([[c,-s,0],[s,c,0],[0,0,1]], dtype=np.float64)
    out=[]
    for (X,Y) in pts_gnd:
        q=R2@np.array([X,Y,1.0], np.float64)
        out.append([q[0], q[1]])
    return np.array(out, dtype=np.float32)

def line_from(p1, p2):
    a=np.array([p1[0], p1[1], 1.0], np.float64)
    b=np.array([p2[0], p2[1], 1.0], np.float64)
    return np.cross(a,b)

# ========= 3) 호모그래피: 바닥 4점 =========
gnd_pts = np.array([[0.0,0.0],[8.0,0.0],[8.0,6.0],[0.0,6.0]], dtype=np.float32)
img_pts=[]
def on_mouse_h(event,x,y,flags,param):
    if event==cv2.EVENT_LBUTTONDOWN and len(img_pts)<4:
        img_pts.append([int(round(x / H_SCALE)), int(round(y / H_SCALE))])
cv2.namedWindow("Pick ground points (4 only)", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Pick ground points (4 only)", on_mouse_h)
while True:
    disp=first.copy()
    for k,(px,py) in enumerate(img_pts):
        cv2.circle(disp,(px,py),6,(0,255,255),-1)
        if k>0: cv2.line(disp, tuple(img_pts[k-1]), (px,py), (0,255,255), 2)
    cv2.putText(disp,"Ground 4point click → Enter (r=reset)",(20,20),FONT,0.7,(255,255,255),2)
    show_fit("Pick ground points (4 only)", disp)
    k=cv2.waitKey(30)&0xFF
    if k==27: cv2.destroyAllWindows(); cap.release(); raise SystemExit
    if k==ord('r'): img_pts=[]
    if k==13 and len(img_pts)>=4:
        pts4=np.array(img_pts[:4],dtype=np.float32)
        ordered=order_quad_TL_TR_BR_BL(pts4)
        ok=True; msg=""
        if ENFORCE_CONVEX and not is_convex_quad(ordered):
            ok=False; msg="볼록 4각형 아님. 다시 찍어주세요."
        if ok and ENFORCE_TRAPEZOID_LIKE and not is_trapezoid_like(ordered):
            ok=False; msg="사다리꼴 형태 아님. 다시 찍어주세요."
        if ok: img_pts=ordered; break
        warn=first.copy(); cv2.putText(warn,msg,(20,60),FONT,0.9,(0,0,255),3)
        show_fit("Pick ground points (4 only)", warn); cv2.waitKey(900); img_pts=[]
cv2.destroyWindow("Pick ground points (4 only)")
img_pts=np.array(img_pts,dtype=np.float32)

H,_=cv2.findHomography(img_pts, gnd_pts, method=0)
if H is None: raise RuntimeError("호모그래피 계산 실패")
Hinv=np.linalg.inv(H)

def pix2gnd(x,y):
    p=np.array([x,y,1.0],np.float64); a,b,c=H@p; return (a/c,b/c)
def gnd2pix(X,Y):
    q=np.array([X,Y,1.0],np.float64); a,b,c=Hinv@q; return (int(round(a/c)), int(round(b/c)))

# ========= 4) ROI 다각형 =========
roi_img=[]; ROI_SCALE=H_SCALE
def on_mouse_roi(event,x,y,flags,param):
    if event==cv2.EVENT_LBUTTONDOWN:
        roi_img.append([int(round(x/ROI_SCALE)), int(round(y/ROI_SCALE))])
cv2.namedWindow("Draw ROI polygon", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Draw ROI polygon", on_mouse_roi)
while True:
    disp=first.copy()
    for i,(px,py) in enumerate(roi_img):
        cv2.circle(disp,(px,py),4,(0,200,255),-1)
        if i>0: cv2.line(disp, tuple(roi_img[i-1]), (px,py), (0,200,255), 2)
    if len(roi_img)>=3:
        cv2.line(disp, tuple(roi_img[-1]), tuple(roi_img[0]), (0,200,255), 1)
    cv2.putText(disp,"ROI click(3+) → Enter (r=reset)",(20,20),FONT,0.7,(255,255,255),2)
    show_fit("Draw ROI polygon", disp)
    k=cv2.waitKey(30)&0xFF
    if k==27: cv2.destroyAllWindows(); cap.release(); raise SystemExit
    if k==ord('r'): roi_img=[]
    if k==13 and len(roi_img)>=3: break
cv2.destroyWindow("Draw ROI polygon")
roi_img=np.array(roi_img,dtype=np.float32)
roi_gnd=np.array([pix2gnd(px,py) for (px,py) in roi_img], dtype=np.float32)

# ========= 4.5) 바닥 기준선 2점 정렬(옵션) =========
if AUTO_ALIGN_FROM_LINE:
    guide_pts=[]; ALIGN_SCALE=H_SCALE
    def on_mouse_align(event,x,y,flags,param):
        if event==cv2.EVENT_LBUTTONDOWN and len(guide_pts)<2:
            guide_pts.append([int(round(x/ALIGN_SCALE)), int(round(y/ALIGN_SCALE))])
    cv2.namedWindow("Align grid: click 2 points on a ground line", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Align grid: click 2 points on a ground line", on_mouse_align)
    while True:
        disp=first.copy()
        for (px,py) in guide_pts: cv2.circle(disp,(px,py),6,(0,200,255),-1)
        if len(guide_pts)==2: cv2.line(disp, tuple(guide_pts[0]), tuple(guide_pts[1]), (0,200,255), 2)
        cv2.putText(disp,"floor baseline 2pt → Enter (r=reset)",(20,20),FONT,0.7,(255,255,255),2)
        show_fit("Align grid: click 2 points on a ground line", disp)
        k=cv2.waitKey(30)&0xFF
        if k==27: cv2.destroyWindow("Align grid: click 2 points on a ground line"); break
        if k==ord('r'): guide_pts=[]
        if k==13 and len(guide_pts)==2:
            cv2.destroyWindow("Align grid: click 2 points on a ground line")
            (x1,y1),(x2,y2)=guide_pts
            X1,Y1=pix2gnd(x1,y1); X2,Y2=pix2gnd(x2,y2)
            theta=np.degrees(np.arctan2(Y2-Y1, X2-X1))
            H,Hinv=rotate_ground(H, -theta)
            roi_gnd=apply_ground_rot_to_points(roi_gnd, -theta)
            break

# # ========= 4.55) 수직 소실점 기반 깊이 정렬(옵션) =========
# if USE_VERTICAL_VP:
#     vz_pairs=[]; VZ_SCALE=H_SCALE
#     def on_mouse_vz(event,x,y,flags,param):
#         if event==cv2.EVENT_LBUTTONDOWN and len(vz_pairs)<4:
#             vz_pairs.append([int(round(x/VZ_SCALE)), int(round(y/VZ_SCALE))])
#     cv2.namedWindow("Pick 2 vertical lines (4 clicks)", cv2.WINDOW_NORMAL)
#     cv2.setMouseCallback("Pick 2 vertical lines (4 clicks)", on_mouse_vz)
#     while True:
#         disp=first.copy()
#         for k in range(0,len(vz_pairs),2):
#             cv2.circle(disp, tuple(vz_pairs[k]), 6, (255,200,0), -1)
#             if k+1 < len(vz_pairs):
#                 cv2.circle(disp, tuple(vz_pairs[k+1]), 6, (255,200,0), -1)
#                 cv2.line(disp, tuple(vz_pairs[k]), tuple(vz_pairs[k+1]), (255,200,0), 2)
#         cv2.putText(disp,"수직 물체 두 개: 위/아래 2점씩(총4) → Enter (r=reset)",
#                     (20,20),FONT,0.65,(255,255,255),2)
#         show_fit("Pick 2 vertical lines (4 clicks)", disp)
#         k=cv2.waitKey(30)&0xFF
#         if k==27: cv2.destroyWindow("Pick 2 vertical lines (4 clicks)"); break
#         if k==ord('r'): vz_pairs=[]
#         if k==13 and len(vz_pairs)==4:
#             cv2.destroyWindow("Pick 2 vertical lines (4 clicks)"); break
#     if len(vz_pairs)==4:
#         l1=line_from(vz_pairs[0], vz_pairs[1])
#         l2=line_from(vz_pairs[2], vz_pairs[3])
#         vZ_h=np.cross(l1,l2)
#         if abs(vZ_h[2])<1e-9:
#             dir_vz=(l1[:2]/(np.linalg.norm(l1[:2])+1e-12))
#         else:
#             vZ=vZ_h[:2]/vZ_h[2]
#             p0=roi_img.mean(axis=0)
#             dir_vz=vZ-p0; dir_vz/= (np.linalg.norm(dir_vz)+1e-12)
#         # 어느 지면축이 깊이인지 판별
#         g0=roi_gnd.mean(axis=0)
#         def img_delta_from_ground_dir(u):
#             pA=np.array(gnd2pix(g0[0],g0[1]),np.float64)
#             pB=np.array(gnd2pix(g0[0]+u[0], g0[1]+u[1]),np.float64)
#             return pB-pA
#         d1=img_delta_from_ground_dir(np.array([1.0,0.0]))
#         d2=img_delta_from_ground_dir(np.array([0.0,1.0]))
#         s1=float(np.dot(d1,dir_vz)); s2=float(np.dot(d2,dir_vz))
#         B=np.eye(2,dtype=np.float64)
#         chosen=d2
#         if abs(s1)>abs(s2):
#             B=np.array([[0,1],[1,0]],dtype=np.float64)  # 축 교환
#             chosen=d1
#         if np.dot(chosen,dir_vz)<0:
#             B=np.array([[ B[0,0],  B[0,1]],
#                         [-B[1,0], -B[1,1]]], dtype=np.float64)  # 깊이축 부호 반전
#         R_axes=np.array([[B[0,0],B[0,1],0],[B[1,0],B[1,1],0],[0,0,1]],dtype=np.float64)
#         H=R_axes@H; Hinv=np.linalg.inv(H)
#         roi_gnd=(roi_gnd@B.T).astype(np.float32)
# ========= 4.56) 픽셀-면적 단조 감소 기준으로 깊이축을 '위쪽'으로 강제 =========
# ========= 4.56) 소실점 기반으로 깊이축을 '항상 위쪽'으로 정렬 (면적법 폴백) =========
FORCE_UPWARD_DEPTH = True
if FORCE_UPWARD_DEPTH:
    # ROI 범위와 중심 (지면)
    minX, minY = roi_gnd.min(axis=0)
    maxX, maxY = roi_gnd.max(axis=0)
    g0 = roi_gnd.mean(axis=0)

    # (A) Hinv로 지면 방향(e1=[1,0], e2=[0,1])의 소실점을 이미지로 투영
    def vp_from_dir(ux, uy):
        v = Hinv @ np.array([ux, uy, 0.0], np.float64)  # 평면의 무한점 [ux,uy,0] → 이미지
        if abs(v[2]) < 1e-9:
            return None  # 너무 멀어 수치적으로 불안정
        return v[:2] / v[2]

    v1 = vp_from_dir(1.0, 0.0)  # e1의 소실점
    v2 = vp_from_dir(0.0, 1.0)  # e2의 소실점
    p0 = roi_img.mean(axis=0)   # ROI 중심(픽셀)

    # (B) '위쪽(y가 작아지는)'에 더 강하게 놓인 소실점을 깊이축 후보로 선택
    def score_upward(vp):
        if vp is None:
            return -1e18
        dy = p0[1] - vp[1]            # vp가 p0보다 위면 +, 아래면 -
        dx = abs(p0[0] - vp[0])       # 수평으로 멀면 감점
        return dy - 0.15 * dx         # 가중치는 경험값

    s1 = score_upward(v1)
    s2 = score_upward(v2)
    use_e1_as_depth = s1 > s2 and v1 is not None or (v2 is None and v1 is not None)

    # (C) 축 교환 행렬 B : new = B @ old (깊이축을 new-e2로 사용)
    B = np.eye(2, dtype=np.float64)
    if use_e1_as_depth:
        B = np.array([[0, 1],
                      [1, 0]], dtype=np.float64)  # e1 ↔ e2 교환

    # (D) 깊이(+new e2)가 '위쪽'(픽셀 y 감소)으로 향하도록 부호 보정
    def img_move_from_ground(u, step=1.0):
        pA = np.array(gnd2pix(g0[0], g0[1]), np.float64)
        pB = np.array(gnd2pix(g0[0] + u[0]*step, g0[1] + u[1]*step), np.float64)
        return pB - pA

    new_e2_old = np.array([B[1, 0], B[1, 1]], dtype=np.float64)
    d_pix = img_move_from_ground(new_e2_old, step=1.0)
    if d_pix[1] >= 0:  # 아래로 향하면 부호 반전
        B = np.array([[ B[0,0],  B[0,1]],
                      [-B[1,0], -B[1,1]]], dtype=np.float64)

    # (E) 만약 소실점이 둘 다 불안정하면(둘 다 None) → 면적 단조 감소 폴백
    if v1 is None and v2 is None:
        roi_span = max(maxX - minX, maxY - minY)
        step = max(1.0, 0.05 * float(roi_span))

        def poly_area(p):
            p = np.asarray(p, dtype=np.float64)
            x, y = p[:, 0], p[:, 1]
            return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

        def cell_area_px(gx, gy, side=1.0):
            h = side * 0.5
            corners_g = [(gx - h, gy - h), (gx + h, gy - h),
                         (gx + h, gy + h), (gx - h, gy + h)]
            pts = []
            for (X, Y) in corners_g:
                pt = gnd2pix_safe(X, Y, Hinv, fr_w, fr_h)
                if pt is None:
                    return np.nan
                pts.append(pt)
            return poly_area(pts)

        A0    = cell_area_px(g0[0],        g0[1],        side=1.0)
        A1_e1 = cell_area_px(g0[0]+step,   g0[1],        side=1.0)
        A1_e2 = cell_area_px(g0[0],        g0[1]+step,   side=1.0)
        d1 = A1_e1 - A0; d2 = A1_e2 - A0
        if (not np.isnan(d1)) and (np.isnan(d2) or d1 < d2):
            B = np.array([[0, 1],
                          [1, 0]], dtype=np.float64)  # e1을 깊이축으로
        # 위쪽 보정 재확인
        new_e2_old = np.array([B[1,0], B[1,1]], dtype=np.float64)
        d_pix = img_move_from_ground(new_e2_old, step=1.0)
        if d_pix[1] >= 0:
            B = np.array([[ B[0,0],  B[0,1]],
                          [-B[1,0], -B[1,1]]], dtype=np.float64)

    # (F) H/ROI에 적용
    R_axes = np.array([[B[0,0], B[0,1], 0],
                       [B[1,0], B[1,1], 0],
                       [0,      0,      1]], dtype=np.float64)
    H    = R_axes @ H
    Hinv = np.linalg.inv(H)
    roi_gnd = (roi_gnd @ B.T).astype(np.float32)


# ========= 4.9) ROI 최종 확정 & 마스크(항상 1회만 생성) =========
roi_gnd = reorder_ccw_pts(roi_gnd)
roi_img = np.array([gnd2pix(X,Y) for (X,Y) in roi_gnd], dtype=np.float32)

roi_poly_px = roi_img.astype(np.int32)
roi_mask_px = np.zeros((fr_h, fr_w), dtype=np.uint8)
cv2.fillPoly(roi_mask_px, [roi_poly_px], 255)

# ========= 5) ROI를 10 m² 격자로 나누기 =========
cell_side = float(np.sqrt(TARGET_CELL_AREA_M2))
minX, minY = roi_gnd.min(axis=0)
maxX, maxY = roi_gnd.max(axis=0)
W = int(np.ceil((maxX - minX) / cell_side))
Hc= int(np.ceil((maxY - minY) / cell_side))

def point_in_poly(x,y,poly):
    inside=False; n=len(poly); j=n-1
    for i in range(n):
        xi,yi=poly[i]; xj,yj=poly[j]
        inter=((yi>y)!=(yj>y)) and (x < (xj-xi)*(y-yi)/(yj-yi+1e-12)+xi)
        if inter: inside = not inside
        j=i
    return inside

cell_mask = np.zeros((Hc,W), dtype=bool)
cell_polys_px = [[None]*W for _ in range(Hc)]

for i in range(Hc):
    for j in range(W):
        cx = minX + (j+0.5)*cell_side
        cy = minY + (i+0.5)*cell_side
        if not point_in_poly(cx, cy, roi_gnd):
            continue
        x0=minX+j*cell_side; x1=x0+cell_side
        y0=minY+i*cell_side; y1=y0+cell_side
        corners_g=[(x0,y0),(x1,y0),(x1,y1),(x0,y1)]
        corners_px=[]
        for (X,Y) in corners_g:
            pt=gnd2pix_safe(X,Y,Hinv,fr_w,fr_h)
            if pt is None: corners_px=None; break
            corners_px.append(pt)
        if corners_px is None: continue
        poly=order_poly_ccw(corners_px)
        if not is_convex_poly(poly):
            hull=cv2.convexHull(poly)
            if len(hull)<4: continue
            poly=hull[:,0,:]
        cell_mask[i,j]=True
        cell_polys_px[i][j]=poly.astype(np.int32)

# ========= 6) YOLO =========
model = YOLO(MODEL)

# ========= 7) 타일 분할 & 글로벌 NMS =========
def split_tiles(frame, rows=ROWS, cols=COLS):
    h,w=frame.shape[:2]
    hs=[int(round(r*h/rows)) for r in range(rows+1)]
    ws=[int(round(c*w/cols)) for c in range(cols+1)]
    crops=[]; offs=[]
    for r in range(rows):
        for c in range(cols):
            y1,y2=hs[r],hs[r+1]; x1,x2=ws[c],ws[c+1]
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

# ========= 8) 메인 루프 =========
while cap.isOpened():
    ok, frame = cap.read()
    if not ok: break

    crops, offs = split_tiles(frame, ROWS, COLS)
    results = model(crops, conf=CONF, imgsz=IMGSZ, device=DEVICE, half=USE_HALF, classes=[0])

    all_boxes=[]
    for res,(ox,oy) in zip(results, offs):
        boxes=res.boxes
        if boxes is None or len(boxes)==0: continue
        xyxy=boxes.xyxy.cpu().numpy(); confs=boxes.conf.cpu().numpy()
        for (x1,y1,x2,y2),cf in zip(xyxy,confs):
            all_boxes.append([x1+ox, y1+oy, x2+ox, y2+oy, float(cf)])
    kept=nms_global(all_boxes, iou_th=0.55)

    counts=np.zeros((Hc,W), dtype=np.int32)
    for x1,y1,x2,y2,cf in kept:
        footx=0.5*(x1+x2); footy=y2
        Xg,Yg=pix2gnd(footx, footy)
        j=int((Xg-minX)//cell_side); i=int((Yg-minY)//cell_side)
        if 0<=i<Hc and 0<=j<W and cell_mask[i,j]:
            counts[i,j]+=1

    overlay=frame.copy()
    for i in range(Hc):
        for j in range(W):
            if not cell_mask[i,j]: continue
            poly=cell_polys_px[i][j]
            if poly is None: continue
            n=counts[i,j]
            if n>=ALERT_THRESHOLD: color=(0,0,255)
            elif n>0:              color=(0,180,0)
            else:                  color=(180,180,180)
            cv2.fillPoly(overlay, [poly], color)
            cv2.polylines(overlay, [poly], True, (50,50,50), 1)

            if DRAW_COUNTS:
                cx=minX+(j+0.5)*cell_side; cy=minY+(i+0.5)*cell_side
                px,py=gnd2pix(cx,cy)
                cv2.putText(overlay, str(int(n)), (px-6,py+6), FONT, 0.6, (0,0,0), 2)
                cv2.putText(overlay, str(int(n)), (px-6,py+6), FONT, 0.6, (255,255,255), 1)

    # ROI 마스크로 클립 후 블렌딩
    overlay_masked = cv2.bitwise_and(overlay, overlay, mask=roi_mask_px)
    out = cv2.addWeighted(overlay_masked, FILL_ALPHA, frame, 1.0 - FILL_ALPHA, 0)

    # ROI 윤곽(마스크와 동일 폴리곤)
    cv2.polylines(out, [roi_poly_px], True, (0,200,255), 2)

    cv2.putText(out, f"Cell ~{TARGET_CELL_AREA_M2:.0f} m^2, Alert >= {ALERT_THRESHOLD}",
                (12,28), FONT, 0.7, (0,0,0), 3)
    cv2.putText(out, f"Cell ~{TARGET_CELL_AREA_M2:.0f} m^2, Alert >= {ALERT_THRESHOLD}",
                (12,28), FONT, 0.7, (255,255,255), 1)

    show_fit("ROI Grid Count (threshold overlay)", out)
    if cv2.waitKey(1)&0xFF in (27, ord('q')): break

cap.release()
cv2.destroyAllWindows()
