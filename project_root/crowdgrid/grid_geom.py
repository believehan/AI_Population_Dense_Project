# -*- coding: utf-8 -*-
# crowdgrid/grid_geom.py
import math
import numpy as np
from typing import List, Optional, Tuple

from .config import AppConfig
from .roi import Roi
from .grid import Cell, Grid    # 기존 Cell/Grid 재활용
from .camgeom import K_from_fov, jacobian_det

class GridBuilderGeom:
    """
    카메라 기하(높이/피치/K 또는 FOV)와 목표 셀 면적(m^2)을 기반으로
    각 행(v)에서 야코비안 detJ를 계산해 side(px)를 산출하는 빌더.

    요구 입력:
      - cfg.use_cam_geom == True
      - cfg.cam_height_m, cfg.cam_pitch_deg
      - (fx,fy,cx,cy) 또는 (fov_h_deg, fov_v_deg)로부터 fx/fy 유도
      - cfg.target_cell_area_m2

    전략:
      - 행 루프: ROI bbox 하단 y_bot에서 시작 → 위로 진행
      - 각 행의 대표 v에서 detJ(u0,v) 평가(u0는 ROI 중앙 x 근사)
      - side_row = sqrt( A_target / detJ ) [px], min/max로 클램프
      - 그 side_row로 좌→우 셀 채우기
    """
    def __init__(self, cfg: AppConfig, frame_w: int, frame_h: int):
        self.cfg = cfg
        self.W = int(frame_w)
        self.H = int(frame_h)

        # 내부파라미터 구성
        if (cfg.fx is not None and cfg.fy is not None and
            cfg.cx is not None and cfg.cy is not None):
            self.fx, self.fy, self.cx, self.cy = cfg.fx, cfg.fy, cfg.cx, cfg.cy
        else:
            k = K_from_fov(self.W, self.H, cfg.fov_h_deg, cfg.fov_v_deg)
            if k is None:
                # 빈 값 허용: build 호출 시 에러로 안내
                self.fx = self.fy = self.cx = self.cy = None
            else:
                self.fx, self.fy, self.cx, self.cy = k

    def _require_ready(self):
        if not self.cfg.use_cam_geom:
            raise ValueError("use_cam_geom=False: 기하 기반 모드 비활성화 상태입니다.")
        if self.cfg.cam_height_m is None or self.cfg.cam_pitch_deg is None:
            raise ValueError("cam_height_m / cam_pitch_deg 가 필요합니다.")
        if self.cfg.target_cell_area_m2 is None:
            raise ValueError("target_cell_area_m2 (목표 셀 면적 m^2)가 필요합니다.")
        if (self.fx is None or self.fy is None or self.cx is None or self.cy is None):
            raise ValueError("내부파라미터 fx/fy/cx/cy 또는 FOV가 필요합니다.")

    def _side_for_row(self, u0: float, v: float) -> Optional[float]:
        """대표 (u0,v)에서 detJ를 평가해 side(px)를 반환. 교차불가 시 None."""
        detJ = jacobian_det(
            u0, v,
            self.fx, self.fy, self.cx, self.cy,
            self.cfg.cam_height_m, self.cfg.cam_pitch_deg,
            eps=1.0
        )
        if detJ is None or detJ <= 0.0:
            return None
        A = float(self.cfg.target_cell_area_m2)
        side = math.sqrt(A / detJ)  # px
        # 안전 클램프
        side = max(self.cfg.min_cell_px, min(self.cfg.max_cell_px, int(round(side))))
        return side

    def build(self, roi: Roi) -> Grid:
        """
        기하 기반 그리드 생성. (unit_mid/alpha는 사용하지 않음)
        """
        self._require_ready()

        x, y, w, h = roi.bbox
        y_bot = y + h
        u0 = x + w * 0.5  # ROI 중앙 x

        cells: List[Cell] = []
        while True:
            # 다음 행의 대표 v (아래쪽에 더 가까운 지점): y_bot - (대략 절반 폭)
            v_probe = y_bot - max(self.cfg.min_cell_px, 16) * 0.5
            v_probe = max(y, min(v_probe, y_bot - 1))
            side = self._side_for_row(u0, v_probe)
            if side is None:
                # 교차 불가/야코비안 실패 → 작은 스텝으로 한 행 위로 이동
                y_top = y_bot - max(self.cfg.min_cell_px, 16)
            else:
                y_top = y_bot - side

            if y_top < y:
                break

            # 좌→우 셀 생성
            x0 = x
            col = 0
            cur_side = side if side is not None else max(self.cfg.min_cell_px, 16)
            while x0 < x + w:
                poly = np.array(
                    [[x0,        y_top],
                     [x0+cur_side, y_top],
                     [x0+cur_side, y_bot],
                     [x0,        y_bot]], dtype=np.int32
                )
                # row_rel은 의미상 유지(원근 LUT 대체해도 보고용으로 둠)
                cells.append(Cell(poly=poly, row_rel=0, col=col, side=int(cur_side)))
                x0 += cur_side
                col += 1

            y_bot = y_top

        # base_side_mid는 기하 모드에선 의미가 희미하므로 첫 행의 side로 표시
        base_side_mid = cells[0].side if cells else int(self.cfg.min_cell_px)
        return Grid(cells=cells, base_side_mid=base_side_mid)
