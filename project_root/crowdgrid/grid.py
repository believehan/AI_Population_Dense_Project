# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np, math
from .config import AppConfig
from .roi import Roi

@dataclass
class Cell:
    """
    하나의 정사각형 셀을 표현.
      - poly   : (4,2) int32, 시계/반시계 순서의 꼭짓점 (좌상, 우상, 우하, 좌하)
      - row_rel: 프레임 하단 기준의 전역 행 인덱스(원근 스케일 계산에 사용)
      - col    : 해당 행 내의 열 인덱스(좌→우 0,1,2,…)
      - side   : 셀 한 변 픽셀 길이(정사각형)
    """
    poly: np.ndarray  # (4,2) int32
    row_rel: int
    col: int
    side: int


@dataclass
class Grid:
    """
    ROI 1개에 대해 생성된 정사각형 셀들의 모음.
      - cells        : 셀 리스트
      - base_side_mid: 중간밴드 기준(전역) 셀 한 변 길이(px)
    """
    cells: List[Cell]
    base_side_mid: int  # mid 밴드 기준 행의 셀 변

    def locate(self, x: float, y: float) -> Optional[int]:
        """
        점(x,y)이 포함된 셀의 인덱스를 반환.
        정사각형이므로 꼭짓점 좌표 비교(AABB)만으로 빠르게 판정.
        """
        for idx, c in enumerate(self.cells):
            p = c.poly
            if (x >= p[0,0]) and (x <= p[1,0]) and (y > p[0,1]) and (y <= p[2,1]):
                return idx
        return None


class GridBuilder:
    """
    옵션 A(전역 원근) 방식으로 ROI의 bbox를 아래→위 행, 좌→우 열 순서로
    정사각형 셀로 채워 Grid를 생성한다.

    핵심 아이디어
      - 프레임 하단으로부터의 '전역 행 인덱스(row_rel)'를 사용하여
        같은 y대라면 ROI가 달라도 동일한 축소율을 적용한다.
      - 각 행의 셀 변은 base_side_mid × (1 - shrink_per_row) ** row_rel 로 결정된다.
    """
    def __init__(self, cfg: AppConfig, frame_h: int):
        self.cfg = cfg
        self.H = frame_h

    def build(self, roi: Roi, unit_mid: float, alpha: float) -> Grid:
        """
        주어진 ROI에 대해 전역 원근 스케일로 정사각형 셀 그리드를 생성한다.

        인자
          - roi      : 대상 ROI (polygon/mask/bbox 포함)
          - unit_mid : 중간밴드 대표 사람 키(px)
          - alpha    : 프리셋×미세조정 배수 (크기 튜닝)

        반환
          - Grid(cells, base_side_mid)

        절차
          1) mid 기준 셀 변 계산:
             base_side_mid = clamp(min_cell_px, max_cell_px,
                                   round(base_k_from_person_h * unit_mid * alpha))
          2) ROI bbox 하단(y_bot)에서 시작해, 아래→위로 행을 생성
             · 전역 행 인덱스(row_rel) = floor((H - y_bot) / base_side_mid) + row
          3) 각 행에서 좌→우로 side 간격으로 정사각형 셀 생성
          4) 더 이상 위로 한 행을 올릴 수 없으면 종료

        비고
          - 마지막 열은 bbox의 오른쪽 경계를 넘어 'side' 만큼 더 나갈 수 있다.
            (정사각형 유지 우선 설계. 실제 표시 단계에서 ROI 마스크로 클립되어 문제 없음)
        """
        # 1) mid 기준 셀 변 계산 (사람 키 앵커 × 튜닝 배수 → 안전 범위로 클램프)
        base_side_mid = int(max(self.cfg.min_cell_px, min(self.cfg.max_cell_px, round(self.cfg.base_k_from_person_h * unit_mid * alpha))))
        cells: List[Cell] = []
        x, y, w, h = roi.bbox
        y_bot = y + h
         # ROI 하단이 프레임 하단에서 몇 번째 '전역 행'에 해당하는지
        # (같은 y대면 ROI가 달라도 같은 row_rel이 되게 함)
        row0 = max(0, int(math.floor((self.H - y_bot) / max(1, base_side_mid))))
        row = 0
        while True:
            row_rel = row0 + row
            # 행별 셀 변(전역 원근): 아래→위로 갈수록 (1 - shrink)^row_rel 만큼 축소
            side = max(self.cfg.min_cell_px, int(round(base_side_mid * ((1.0 - self.cfg.shrink_per_row) ** row_rel))))
            y_top = y_bot - side
            if y_top < y:
                break # ROI bbox 위 경계를 넘으면 종료

            # 좌→우로 정사각형 셀 채우기
            x0 = x; col = 0
            while x0 < x + w:
                poly = np.array([[x0, y_top],[x0+side, y_top],[x0+side, y_bot],[x0, y_bot]], dtype=np.int32)
                cells.append(Cell(poly=poly, row_rel=row_rel, col=col, side=side))
                x0 += side
                col += 1
                
            # 다음 행(위쪽)
            y_bot = y_top
            row += 1
            
        return Grid(cells=cells, base_side_mid=base_side_mid)
