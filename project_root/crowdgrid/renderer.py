# -*- coding: utf-8 -*-
from typing import Dict, Optional
import cv2, numpy as np
from .config import AppConfig, RuntimeState
from .roi import Roi
from .grid import Grid
from .utils import apply_overlay_in_mask

class Renderer:
    """
    ROI/그리드/카운트 결과를 화면에 그리는 역할.
    - 셀 채색은 'overlay'에 먼저 그리고, 마지막에 ROI 마스크로 클립하여
      원본(frame) 위에 반투명 합성한다(ROI 외부 밝기 보존).
    - OSD(상태 텍스트)는 최상단에 얹는다.
    """
    FONT = cv2.FONT_HERSHEY_SIMPLEX

    def __init__(self, cfg: AppConfig):
        self.cfg = cfg

    def draw_roi(self, frame: np.ndarray, roi: Roi, grid: Optional[Grid], counts: Optional[Dict[int,int]]) -> np.ndarray:
        """
        단일 ROI에 대해 그리드 셀을 채색하고, ROI 마스크로 클립한 뒤
        원본 frame에 반투명 합성하여 결과 이미지를 반환한다.

        인자
          - frame : BGR 원본 프레임 (H×W×3)
          - roi   : 대상 ROI(다각형/마스크/bbox 포함)
          - grid  : 이 ROI용으로 미리 생성된 그리드 (None이면 ROI 폴리곤 윤곽만 그림)
          - counts: 셀 인덱스→인원수 매핑 (None이면 0으로 간주)

        반환
          - out: 렌더링 결과 프레임(BGR), 원본과 동일 크기

        비고
          - 여기서는 셀 정사각형(poly)이 ROI 다각형 밖으로 살짝 나가더라도
            최종 합성 시 'apply_overlay_in_mask'로 ROI 내부만 보이게 처리한다.
          - ROI 실제 다각형과의 교차 검사를 통해 '진짜 내부 셀만' 그리고 싶다면
            (성능 비용↑) 교차 판정을 추가할 수 있다.
        """
        overlay = frame.copy() # 색 채우기/라인을 그릴 임시 레이어
        if grid is not None:
            for i, c in enumerate(grid.cells):
                poly = c.poly
                
                # 셀 카운트에 따른 색상:
                #  - 경고(빨강): alert_threshold 이상
                #  - 점유(초록): 1개 이상
                #  - 비어있음(회색): 0개
                n = counts.get(i, 0) if counts is not None else 0
                if n >= self.cfg.alert_threshold: 
                    color=(0,0,255)     # BGR: Red
                elif n > 0:                      
                    color=(0,180,0)     # BGR: Green-ish
                else:                             
                    color=(180,180,180) # BGR: Gray
                    
                # 너무 작은 셀은 시각적으로 의미가 없으므로 채우기 생략
                if c.side >= self.cfg.min_cell_px:
                    cv2.fillPoly(overlay, [poly], color)
                    cv2.polylines(overlay, [poly], True, (50,50,50), 1)
                    # (선택) 셀 중앙에 숫자 찍기 원하면 아래 주석 해제
                    # cx = (poly[0,0] + poly[1,0]) // 2
                    # cy = (poly[0,1] + poly[2,1]) // 2
                    # cv2.putText(overlay, str(n), (cx-6, cy+6), self.FONT, 0.4, (0,0,0), 2, cv2.LINE_AA)
                    # cv2.putText(overlay, str(n), (cx-6, cy+6), self.FONT, 0.4, (255,255,255), 1, cv2.LINE_AA)
        
        # ROI 내부만 반투명 합성 (바깥은 원본 유지)            
        out = apply_overlay_in_mask(frame, overlay, roi.mask, self.cfg.fill_alpha)
        
        # ROI 폴리곤 윤곽선 표시(시각적 경계)
        cv2.polylines(out, [roi.polygon], True, (0,200,255), 2)
        return out

    def draw_osd(self, img: np.ndarray, rt: RuntimeState, unit_mid_txt: str, mode_txt: str):
        """
        상단 좌측에 상태(OSD) 텍스트를 그린다.

        인자
          - img          : 출력 프레임(BGR)
          - rt           : 런타임 상태(HEAD/FOOT, 프리셋/파인 배수 등)
          - unit_mid_txt : 앵커 텍스트(예: "unit_mid=62.3px" 또는 "--")
          - mode_txt     : 모드 텍스트(예: "measuring..." / "frozen")
        """
        # 1) 모드(상태) 텍스트 (두 번 그려 가독성 향상: 검은 외곽선 → 흰 텍스트)
        cv2.putText(img, f"{mode_txt}", (12,28), self.FONT, 0.7, (0,0,0), 3)
        cv2.putText(img, f"{mode_txt}", (12,28), self.FONT, 0.7, (255,255,255), 1)
        
        # 2) 파라미터 상태(앵커/배수/원근/참조점)
        ref_txt = "HEAD" if rt.use_head_point else "FOOT"
        line = f"{unit_mid_txt}  x{rt.alpha():.2f}  shrink/row={self.cfg.shrink_per_row:.3f}  ref={ref_txt} [H/F]"
        cv2.putText(img, line, (12,52), self.FONT, 0.7, (0,0,0), 3)
        cv2.putText(img, line, (12,52), self.FONT, 0.7, (255,255,255), 1)
        
        # 3) 조작법 힌트
        cv2.putText(img, "Manual: [1..5]=preset  +/-=size  R=remeasure  H/F=head/foot", (12,76), self.FONT, 0.7, (0,0,0), 3)
        cv2.putText(img, "Manual: [1..5]=preset  +/-=size  R=remeasure  H/F=head/foot", (12,76), self.FONT, 0.7, (255,255,255), 1)
