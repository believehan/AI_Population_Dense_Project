# -*- coding: utf-8 -*-
from collections import deque
from typing import Optional
import numpy as np
import time
from .config import AppConfig
from .types import Det

class AnchorCalibrator:
    """
        화면 '중간밴드(기본 20~40%)'에 있는 사람 검출들의 '키(px)=y2-y1' 표본을 모아
        1회성 앵커 `unit_mid`(대표 키)를 확정(freeze)하는 모듈.

        역할
        - 노이즈 필터(min_h_px, aspect_min)와 ROI 내부 여부를 만족하는 표본만 수집
        - 충분한 표본 수 또는 타임아웃 조건이 만족되면 대표값을 산출하고 고정

        대표값 산출(robust median)
        - 표본 배열에서 10~90% 분위만 남기고 중앙값을 취함(이상치에 강건)
        - 표본이 전무하면 보수적 대체값 사용: max(min_h_px, 0.065 × frame_h)

        준비 완료 조건(ready)
        - 표본 수 ≥ auto_min_samples, 또는
        - (경과시간 > auto_timeout_sec) AND (표본 수 ≥ max(5, auto_min_samples//2))

        주의
        - 중간밴드(y 조건) 체크는 호출 측에서 수행 후 add_sample()을 호출한다.
        - 확정된 unit_mid는 그리드 빌드의 기준 길이로 사용된다.
    """
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.hbuf_mid: deque = deque(maxlen=300)
        self.reset()

    def reset(self):
        """표본 버퍼/타임스탬프/확정값(unit_mid)을 초기화한다."""
        self.hbuf_mid.clear()
        self.started_ts = time.time()
        self.unit_mid: Optional[float] = None

    def add_sample(self, det: Det, ref_y: float, is_inside_roi: bool):
        """
        검출 1건을 표본 후보로 추가한다.

        인자
          - det           : 사람 박스(Det)
          - ref_y         : 참조점 y (HEAD/FOOT 중 선택된 기준의 y좌표, 호출 측에서 mid band 판정에 사용)
          - is_inside_roi : 참조점이 ROI 내부인지 여부(마스크 기준)

        동작
          - ROI 외부면 무시
          - 키 h=y2-y1, 가로 w=x2-x1 계산
          - h < min_h_px 또는 (h/w) < aspect_min 인 경우 필터링
          - (중간밴드 판정은 호출 측에서 이미 완료) → 표본 버퍼에 h 추가
        """
        if not is_inside_roi:
            return
        h = det.y2 - det.y1; w = det.x2 - det.x1
        if h < self.cfg.min_h_px or w <= 0: return
        if (h/(w+1e-9)) < self.cfg.aspect_min: return
        y_low, y_high = int(self.cfg.mid_band[0]*ref_y*0 + 0), int(self.cfg.mid_band[1]*ref_y*0 + 0)  # dummy (unused)
        # 실제 판정은 호출 측에서 수행(참조점 y가 mid_band 안인지)
        self.hbuf_mid.append(h)

    def ready(self) -> bool:
        """
        측정 확정 조건을 만족했는지 반환한다.
        - 표본 수 기준 또는 타임아웃 기준(절반 이상 수집) 중 하나 충족 시 True
        """
        enough = (len(self.hbuf_mid) >= self.cfg.auto_min_samples)
        timeout = ((time.time() - self.started_ts) > self.cfg.auto_timeout_sec and len(self.hbuf_mid) >= max(5, self.cfg.auto_min_samples//2))
        return enough or timeout

    def finalize(self, frame_h: int) -> float:
        """
        지금까지의 표본으로 robust median을 계산해 unit_mid를 확정한다.
        표본이 없으면 보수적 대체값을 사용한다.

        반환
          - unit_mid(float): 확정된 대표 키(px)
        """
        def robust_median(arr: np.ndarray) -> Optional[float]:
            if arr.size == 0: return None
            q10, q90 = np.percentile(arr, [10, 90])
            arr = arr[(arr >= q10) & (arr <= q90)]
            if arr.size == 0: return None
            return float(np.median(arr))
        arr = np.asarray(self.hbuf_mid, dtype=np.float32)
        unit = robust_median(arr)
        if unit is None:
            unit = max(self.cfg.min_h_px, int(0.065 * frame_h))  # 표본 0개일 때의 보수적 기본값
        self.unit_mid = unit
        return unit
