# -*- coding: utf-8 -*-
import cv2, time
from typing import Dict
from .config import AppConfig, RuntimeState
from .streams import open_capture
from .model_runner import ModelRunner
from .roi import RoiManager
from .calibrator import AnchorCalibrator
from .grid import GridBuilder, Grid
from .counter import Counter
from .renderer import Renderer
from .utils import ref_point, show_fit

def main():
    cfg = AppConfig()
    rt  = RuntimeState()

    # 0) 스트림/파일/웹캠 오픈 (유연 입력)
    cap = open_capture(cfg.source)
    if not cap.isOpened():
        raise RuntimeError("영상/카메라/스트림을 열 수 없습니다. (source/youtube_url 확인)")

    # 1) 첫 프레임 확보 → 해상도(H,W) 획득
    if not cap.isOpened():
        raise RuntimeError("VideoCapture 열기 실패")

    ok, first = cap.read()
    if not ok:
        raise RuntimeError("첫 프레임 읽기 실패")
    H, W = first.shape[:2]

    # 2) ROI 준비 (없으면 풀프레임 ROI 1개 자동 생성)
    roi_mgr = RoiManager()
    roi_mgr.ensure_default(H, W) # 보장: roi_mgr.list()[0] == 풀프레임 ROI

    # 3) 모듈 준비
    model = ModelRunner(cfg)                   # 타일 배치 추론 → 전역 NMS
    calibr = AnchorCalibrator(cfg)             # 중간밴드 표본 → unit_mid 확정
    grid_builder = GridBuilder(cfg, frame_h=H) # 옵션 A(전역 원근) 그리드
    counter = Counter(cfg)                     # 참조점 한 점 기준 카운트
    renderer = Renderer(cfg)                   # 셀 채색 + 마스크 클립 + OSD

    # ROI별 그리드 캐시(앵커 확정 후 생성/갱신)
    grid_cache: Dict[int, Grid] = {}  # roi.id → Grid

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        # 4-1) 추론(프레임당 1회)
        dets = model.infer(frame)

        # 4-2) 캘리브 단계: 중간밴드 표본 수집 (freeze 전만)
        if not rt.frozen:
            mid_low, mid_high = int(cfg.mid_band[0]*H), int(cfg.mid_band[1]*H)
            
            # 전역 앵커 포함 판정은 '풀프레임 ROI' 기준이 안전
            full_roi = roi_mgr.list()[0]  # ensure_default로 최초 ROI는 풀프레임
                        
            for d in dets:
                # (가드) 신뢰도 기준
                if d.conf < cfg.conf:
                    continue

                # 참조점(HEAD/FOOT) y가 중간밴드 안이면 표본 후보
                rx, ry = ref_point(d.x1, d.y1, d.x2, d.y2, rt.use_head_point)
                if not (mid_low <= ry <= mid_high):
                    continue

                # ROI 포함 여부(전역 앵커이므로 풀프레임 ROI 기준으로 충분)
                ix = int(min(max(rx, 0), W-1))
                iy = int(min(max(ry, 0), H-1))
                inside = (full_roi.mask[iy, ix] > 0)

                # 표본 추가(중간밴드 여부는 이미 위에서 판정)
                calibr.add_sample(d, ry, inside)

            # 4-3) 조건 충족 시 앵커 확정 → 모든 ROI 그리드 생성/캐시
            if calibr.ready():
                rt.unit_mid = calibr.finalize(frame_h=H)
                rt.frozen   = True
                grid_cache.clear()
                for roi in roi_mgr.list():
                    g = grid_builder.build(roi, unit_mid=rt.unit_mid, alpha=rt.alpha())
                    grid_cache[roi.id] = g

        # 4-4) 프리셋/± 조정은 freeze 이후 즉시 재빌드
        def rebuild_all():
            if rt.unit_mid is None:
                return
            grid_cache.clear()
            for roi in roi_mgr.list():
                g = grid_builder.build(roi, unit_mid=rt.unit_mid, alpha=rt.alpha())
                grid_cache[roi.id] = g

        # 4-5) 카운트 + 렌더링(ROI별)
        out = frame.copy()
        for roi in roi_mgr.list():
            g = grid_cache.get(roi.id)
            counts = None
            if g is not None:
                counts = counter.count(
                    dets,
                    roi,
                    g,
                    use_head_point=rt.use_head_point,
                    roi_test_with_ref=True  # ROI 포함 판정도 참조점과 동일한 점으로
                )
            out = renderer.draw_roi(out, roi, g, counts)

        # 4-6) OSD(상태/파라미터/조작법)
        mode_txt = (
            f"measuring... samples={len(calibr.hbuf_mid)}/{cfg.auto_min_samples}"
            if not rt.frozen else
            "frozen"
        )
        unit_txt = f"unit_mid={rt.unit_mid:.1f}px" if rt.unit_mid is not None else "unit_mid=--"
        renderer.draw_osd(out, rt, unit_txt, mode_txt)

        # 4-7) 디스플레이(윈도우 크기 제한 내로 축소 표시)
        show_fit(cfg.win_name, out, cfg.display_max_w, cfg.display_max_h)

        # 4-8) 키 입력 처리
        k = cv2.waitKey(1) & 0xFF
        if k in (27, ord('q'), ord('Q')):      # 종료
            break
        elif k in (ord('h'), ord('H')):        # 참조점 = HEAD
            rt.use_head_point = True
        elif k in (ord('f'), ord('F')):        # 참조점 = FOOT
            rt.use_head_point = False
        elif k in (ord('1'), ord('2'), ord('3'), ord('4'), ord('5')):
            # 프리셋 변경(앵커 확정 후에만 유효)
            if rt.frozen and rt.unit_mid is not None:
                rt.preset_idx = int(chr(k)) - 1
                rebuild_all()
        elif k in (ord('+'), ord('=')):        # 셀 크기 미세 ↑
            if rt.frozen and rt.unit_mid is not None:
                rt.fine_factor *= 1.05
                rebuild_all()
        elif k in (ord('-'), ord('_')):        # 셀 크기 미세 ↓
            if rt.frozen and rt.unit_mid is not None:
                rt.fine_factor /= 1.05
                rebuild_all()
        elif k in (ord('r'), ord('R')):        # 재측정(앵커/그리드 초기화)
            calibr.reset()
            rt.frozen = False
            rt.unit_mid = None
            rt.measure_started = time.time()
            grid_cache.clear()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()