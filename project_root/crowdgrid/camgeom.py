# -*- coding: utf-8 -*-
# crowdgrid/camgeom.py
import math
import numpy as np
from typing import Optional, Tuple

def K_from_fov(W: int, H: int,
               fov_h_deg: Optional[float],
               fov_v_deg: Optional[float]) -> Optional[Tuple[float,float,float,float]]:
    """
    수평/수직 FOV(°)로부터 (fx, fy, cx, cy) 근사 계산.
    - 둘 다 제공되면 각각으로 fx, fy를 계산
    - 하나만 제공되면 나머지는 화면비로 근사하거나 None 반환 가능
    - 둘 다 None이면 None 반환
    """
    if fov_h_deg is None and fov_v_deg is None:
        return None
    cx = W * 0.5
    cy = H * 0.5
    fx = None
    fy = None
    if fov_h_deg is not None:
        fx = (W * 0.5) / math.tan(math.radians(fov_h_deg) * 0.5)
    if fov_v_deg is not None:
        fy = (H * 0.5) / math.tan(math.radians(fov_v_deg) * 0.5)
    # 하나만 있으면 다른 하나는 화면비로 보정
    if fx is None and fy is not None:
        fx = fy * (W / H)
    if fy is None and fx is not None:
        fy = fx * (H / W)
    if fx is None or fy is None:
        return None
    return float(fx), float(fy), float(cx), float(cy)

def rot_x(pitch_deg: float) -> np.ndarray:
    """X축 회전 행렬 (아래로 숙이면 +deg 가정)."""
    th = math.radians(pitch_deg)
    c, s = math.cos(th), math.sin(th)
    return np.array([[1, 0, 0],
                     [0, c,-s],
                     [0, s, c]], dtype=np.float64)

def ray_world(u: float, v: float,
              fx: float, fy: float, cx: float, cy: float,
              pitch_deg: float) -> np.ndarray:
    """
    이미지픽셀(u,v) → 카메라좌표계 방향벡터 → 피치 회전 적용 후 '월드' 방향 벡터.
    반환 r_w는 정규화하지 않아도 교차계산에 문제 없음(비례만 중요).
    """
    x = (u - cx) / fx
    y = (v - cy) / fy
    r_c = np.array([x, y, 1.0], dtype=np.float64)
    R = rot_x(pitch_deg)
    r_w = R @ r_c
    return r_w

def pix_to_ground(u: float, v: float,
                  fx: float, fy: float, cx: float, cy: float,
                  cam_height_m: float, pitch_deg: float) -> Optional[Tuple[float,float]]:
    """
    영상 픽셀(u,v)에서 출발한 광선과 '지면(Y=0)'의 교점을 구해 (X,Z) [m]를 반환.
    카메라 위치 C=(0,H,0), 평면 Y=0 가정.
    교차불가(하늘방향 등)면 None.
    """
    r_w = ray_world(u, v, fx, fy, cx, cy, pitch_deg)  # (dx, dy, dz)
    dy = r_w[1]
    if abs(dy) < 1e-9:
        return None
    t = -cam_height_m / dy
    if t <= 0:
        return None
    X = t * r_w[0]
    Z = t * r_w[2]
    return float(X), float(Z)

def jacobian_det(u: float, v: float,
                 fx: float, fy: float, cx: float, cy: float,
                 cam_height_m: float, pitch_deg: float,
                 eps: float = 1.0) -> Optional[float]:
    """
    (X,Z) = f(u,v)의 야코비안 |det(∂(X,Z)/∂(u,v))| 을 수치미분으로 근사.
    eps: 픽셀 스텝(기본 1px)
    None(교차불가) 발생하면 None 반환.
    """
    def P(uu, vv):
        return pix_to_ground(uu, vv, fx, fy, cx, cy, cam_height_m, pitch_deg)

    P0 = P(u, v)
    Pu1 = P(u + eps, v)
    Pu2 = P(u - eps, v)
    Pv1 = P(u, v + eps)
    Pv2 = P(u, v - eps)
    if None in (P0, Pu1, Pu2, Pv1, Pv2):
        return None

    Xu, Zu = (Pu1[0] - Pu2[0]) / (2*eps), (Pu1[1] - Pu2[1]) / (2*eps)
    Xv, Zv = (Pv1[0] - Pv2[0]) / (2*eps), (Pv1[1] - Pv2[1]) / (2*eps)
    detJ = abs(Xu*Zv - Zu*Xv)  # [m^2 / px^2]
    return float(detJ)
