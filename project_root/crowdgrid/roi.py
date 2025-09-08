# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import List, Optional, Tuple
import cv2, numpy as np
from .utils import polygon_mask

@dataclass
class Roi: 
    """
    관심영역(Region of Interest) 1개를 표현하는 데이터 모델.

    필드 설명
      - id     : 각 ROI의 고유 번호 (렌더링/그리드 캐시의 키로 사용)
      - name   : 표시/로그용 이름 (예: "Entrance", "Bench Left")
      - polygon: (N,2) 다각형 꼭짓점 배열(이미지 좌표계, 픽셀 단위)
      - mask   : (H,W) uint8 이진 마스크 — 내부=255, 외부=0
                 · 카운트: 참조점이 ROI 내부인지 빠르게 판정
                 · 렌더링: 오버레이를 ROI 내부에만 블렌딩
      - bbox   : (x,y,w,h) 다각형의 외접 사각형 — 그리드 빌드에서 셀을 깔 영역의 외곽 경계

    주의
      - polygon은 자기교차 없는 단순 다각형을 권장 (cv2.fillPoly의 전제와 일치)
      - polygon/mask/bbox는 프레임 해상도(H,W)에 종속적이므로, 해상도가 달라지면 다시 생성 필요
    """
    id: int # 각 ROI의 고유 번호. 렌더링 / 그리드 캐시에서 키로 사용
    name: str # 표시/로그용 이름
    polygon: np.ndarray  # (N,2) ROI 다각형 꼭짓점들(이미지 좌표, 픽셀 단위). 사각형뿐 아니라 임의의 다각형 가능.
    mask: np.ndarray     # (H,W) uint8 
    """ mask: np.ndarray 
    다각형 내부=255, 외부=0인 바이너리 마스크.
    → (1) 카운트: 사람 참조점이 ROI 안인지 빠르게 판정
    → (2) 렌더링: 오버레이를 ROI 내부에만 블렌딩
    """
    bbox: Tuple[int,int,int,int]  # (x,y,w,h) 다각형의 외접 사각형. 그리드 빌드에서 “셀을 깔 지역의 외곽”으로 써.

    @staticmethod
    def full_frame(h: int, w: int, idx: int=0) -> "Roi":
        """
        프레임 전체를 덮는 ROI를 생성하는 헬퍼.
        ROI가 하나도 없어도 시스템이 항상 동작하도록 기본 ROI를 보장한다.
        """
        poly = np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1]], dtype=np.int32)
        mask = polygon_mask(h, w, poly)
        x,y,ww,hh = cv2.boundingRect(poly)
        return Roi(idx, f"FULL_{idx}", poly, mask, (x,y,ww,hh))


class RoiManager:
    def __init__(self):
        """
        ROI 목록과 '활성 ROI' 상태를 관리하는 간단한 매니저.

        내부 상태
        - _rois      : 등록된 ROI 리스트
        - _active_id : 현재 활성 ROI의 id (없으면 None)
        - _next_id   : 신규 ROI에 부여할 다음 id(증분 발급기)
        """
        self._rois: List[Roi] = []
        self._active_id: Optional[int] = None
        self._next_id: int = 0

    def ensure_default(self, h: int, w: int):
        """
        ROI가 하나도 없으면 '풀프레임 ROI' 1개를 자동 생성해 초기 가동성을 확보한다.
        (UI로 ROI를 입력하지 않아도 파이프라인이 바로 동작하도록 보장)
        """
        if not self._rois:
            r = Roi.full_frame(h, w, idx=self._next_id)
            self._rois.append(r)
            self._active_id = r.id
            self._next_id += 1

    def add_polygon(self, poly: np.ndarray, h: int, w: int, name: Optional[str]=None) -> Roi:
        """
        다각형 좌표를 받아 mask/bbox를 생성하여 목록에 등록하고, 등록된 ROI를 활성으로 전환한다.
        - poly: (N,2) float/int 배열 (이미지 좌표계, 픽셀 단위)
        - h,w : 프레임 해상도(마스크 생성용)
        """
        mask = polygon_mask(h, w, poly)
        x,y,ww,hh = cv2.boundingRect(poly)
        r = Roi(self._next_id, name or f"ROI_{self._next_id}", poly.astype(np.int32), mask, (x,y,ww,hh))
        self._rois.append(r); self._active_id = r.id; self._next_id += 1
        return r

    def remove_active(self):
        """
        현재 활성 ROI를 제거한다.
        제거 후 ROI가 남아 있다면 첫 번째 ROI를 활성으로 전환한다.
        """
        if self._active_id is None: return
        self._rois = [r for r in self._rois if r.id != self._active_id]
        self._active_id = self._rois[0].id if self._rois else None

    def cycle_active(self):
        """
        활성 ROI를 목록 순서대로 순환(TAB 전환과 유사).
        UI에서 ROI 편집/삭제 시 어떤 ROI가 대상인지 명확히 하기 위해 사용.
        """
        if not self._rois: return
        ids = [r.id for r in self._rois]
        if self._active_id not in ids:
            self._active_id = ids[0]; return
        i = ids.index(self._active_id); self._active_id = ids[(i+1)%len(ids)]

    def list(self) -> List[Roi]:
        """등록된 ROI 리스트를 복사해 반환한다(외부에서 안전하게 순회하도록)."""
        return list(self._rois)

    def active(self) -> Optional[Roi]:
        """현재 활성 ROI를 반환한다(없으면 None)."""
        for r in self._rois:
            if r.id == self._active_id:
                return r
        return None