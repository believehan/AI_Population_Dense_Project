# -*- coding: utf-8 -*-
from typing import List, Tuple
import numpy as np
import cv2


def iou_xyxy(a: Tuple[float,float,float,float,float],
             b: Tuple[float,float,float,float,float]) -> float:
    """
    두 박스의 IoU(Intersection over Union)을 계산해 0~1 사이 값으로 반환.
    ─ 사용처: NMS(중복 제거)에서 "같은 대상을 두 번 잡았는지" 판단할 때만 사용
    ─ 입력 형식: (x1, y1, x2, y2, score) 5-튜플
      * IoU 계산에는 좌표 4개만 사용하고 score는 무시(정렬/호출 형식을 맞추기 위함)
    ─ 좌표 규약: XYXY(AABB) = 좌상단(x1,y1), 우하단(x2,y2), x1<x2, y1<y2 가정

    계산 개요
      1) 교집합 사각형 좌표 = (max(x1), max(y1)) ~ (min(x2), min(y2))
      2) 교집합 폭/높이 음수 방지: iw=max(0, ix2-ix1), ih=max(0, iy2-iy1)
      3) inter = iw*ih, area_a/area_b = 각 박스 면적
      4) IoU = inter / (area_a + area_b - inter + 1e-9)  # 0-division 보호

    엣지 케이스
      - 접점만 닿는 경우(폭 또는 높이=0) → inter=0 → IoU=0
      - 좌표가 뒤집힌 잘못된 입력은 면적 0으로 처리(상류에서 정규화 권장)
      - 회전 박스(RBox)나 마스크 IoU 용도는 아님(축정렬 AABB 전용)

    예시
      - 같은 사람을 두 번 잡은 두 박스가 크게 겹침 → IoU≈0.7 (임계 0.55↑면 중복으로 판단)
      - 서로 다른 사람이라 조금만 겹침 → IoU≈0.1 (둘 다 유지)
    
    """
    ax1, ay1, ax2, ay2 = a[:4]; bx1, by1, bx2, by2 = b[:4]
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0: return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / (area_a + area_b - inter + 1e-9)

def nms_global(boxes: List[Tuple[float,float,float,float,float]],
               iou_th: float = 0.55) -> List[Tuple[float,float,float,float,float]]:
    """
    전역 NMS(Non-Max Suppression): 타일 오버랩/중복 탐지에서 같은 대상을 가리키는
    박스들을 제거하고, 신뢰도(conf)가 가장 높은 박스 1개만 남긴다.

    절차
      1) score(conf) 내림차순 정렬
      2) 앞에서부터 하나(cur) 채택 → 나머지 중 IoU(cur, b) >= iou_th 인 b는 제거
      3) 남은 것들에 대해 반복

    파라미터 가이드
      - iou_th(기본 0.55) ↑: 더 공격적으로 중복 제거(가까운 두 사람을 하나로 오인할 위험 ↑)
      - iou_th ↓: 보수적으로 남김(중복이 남을 위험 ↑)

    성능/한계
      - O(N^2). N이 수백 개 수준이면 충분히 빠름
      - 더 큰 스케일이면 벡터화/라이브러리 NMS 고려

    주의
      - 여기서만 IoU를 사용한다. **셀 카운트는 IoU가 아니라 참조점(HEAD/FOOT) 한 점 기준**으로 셀을 1곳에만 +1 한다.
    """
    boxes = sorted(boxes, key=lambda x: x[4], reverse=True)
    keep: List[Tuple[float,float,float,float,float]] = []
    while boxes:
        cur = boxes.pop(0)
        keep.append(cur)
        boxes = [b for b in boxes if iou_xyxy(cur, b) < iou_th]
    return keep


def ref_point(x1: float, y1: float, x2: float, y2: float, use_head: bool = True) -> Tuple[float,float]:
    """사람 박스의 **참조점**을 반환.
    - use_head=True  → 머리 기준: (top-center) = ( (x1+x2)/2 , y1 )
    - use_head=False → 발끝 기준: (bottom-center) = ( (x1+x2)/2 , y2 )
    
    ROI 포함 판정이나 셀 위치 지정에 일관된 기준점을 쓰기 위한 도우미.
    """
    cx = 0.5 * (x1 + x2)
    cy = y1 if use_head else y2
    return cx, cy

def polygon_mask(h: int, w: int, poly: np.ndarray) -> np.ndarray:
    """다각형(poly) 내부를 255로 채운 단일 채널 마스크를 생성.
    - 입력 poly: (N,2) int32/float32 배열(픽셀 좌표)
    - 반환: (H,W) uint8 마스크; 내부=255, 외부=0
    """
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [poly.astype(np.int32)], 255)
    return mask


def apply_overlay_in_mask(frame: np.ndarray, overlay: np.ndarray, mask: np.ndarray, alpha: float) -> np.ndarray:
    """ROI **내부만** 반투명 합성해 색칠 결과를 입힘.
    
    - frame  : 원본 프레임(BGR)
    - overlay: 색칠/라인 등이 그려진 프레임과 동일 크기의 BGR 이미지
    - mask   : ROI 마스크(내부=255, 외부=0)
    - alpha  : 0.0~1.0, overlay 가중치
    
    동작: addWeighted 로 전체 블렌딩 이미지를 만든 다음, 마스크가 255인 영역에만 복사.
    이렇게 하면 ROI **바깥**은 원본 밝기가 그대로 유지되어 화면 왜곡이 없음.
    """
    blended = cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0)
    out = frame.copy()
    out[mask > 0] = blended[mask > 0]
    return out

def compute_fit_scale(w: int, h: int, max_w: int, max_h: int) -> float:
    return min(max_w / w, max_h / h, 1.0)


def show_fit(name: str, img: np.ndarray, max_w: int, max_h: int):
    h, w = img.shape[:2]
    s = compute_fit_scale(w, h, max_w, max_h)
    if s < 1.0:
        img = cv2.resize(img, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA)
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, img)
