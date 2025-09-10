# -*- coding: utf-8 -*-
import time
from typing import Dict, Optional

import cv2
import numpy as np
import streamlit as st

from crowdgrid.config import AppConfig, RuntimeState
from crowdgrid.streams import open_capture          # 또는 StreamResolver.open_capture
from crowdgrid.model_runner import ModelRunner
from crowdgrid.roi import RoiManager
from crowdgrid.calibrator import AnchorCalibrator
from crowdgrid.grid import GridBuilder, Grid
from crowdgrid.counter import Counter
from crowdgrid.renderer import Renderer
from crowdgrid.utils import ref_point

st.set_page_config(page_title="Crowd Grid (Streamlit)", layout="wide")

# ============ 세션 상태 ============
if "cfg" not in st.session_state: st.session_state.cfg = AppConfig()
if "rt"  not in st.session_state: st.session_state.rt  = RuntimeState()
if "cap" not in st.session_state: st.session_state.cap = None
for k in ("H","W","roi_mgr","model","calibr","grid_builder","counter","renderer"):
    if k not in st.session_state: st.session_state[k] = None
if "grid_cache" not in st.session_state: st.session_state.grid_cache = {}
if "running"    not in st.session_state: st.session_state.running = False
if "use_geom"   not in st.session_state: st.session_state.use_geom = False

cfg: AppConfig = st.session_state.cfg
rt:  RuntimeState = st.session_state.rt

# --- (중요) 기하 입력 필드가 AppConfig에 없어도 안전하게 기본값을 갖도록 보정 ---
def _ensure_camgeom_defaults():
    defaults = dict(
        use_cam_geom=False,
        cam_height_m=None, cam_pitch_deg=None,
        fx=None, fy=None, cx=None, cy=None,
        fov_h_deg=None, fov_v_deg=None,
        target_cell_area_m2=None,
    )
    for k, v in defaults.items():
        if not hasattr(cfg, k):
            setattr(cfg, k, v)
_ensure_camgeom_defaults()

# ============ 사이드바 ============
st.sidebar.header("Input / Model")
cfg.source = st.sidebar.text_input("source (파일/카메라 인덱스/URL)", cfg.source)
cfg.model_path = st.sidebar.text_input("model_path", cfg.model_path)
cfg.device = st.sidebar.selectbox("device", ["cpu","cuda"], index=0 if cfg.device=="cpu" else 1)
cfg.imgsz = st.sidebar.number_input("imgsz", 320, 1920, int(cfg.imgsz), step=32)
cfg.conf  = st.sidebar.slider("conf", 0.0, 1.0, float(cfg.conf), 0.01)

st.sidebar.header("Tiling")
cfg.tile_rows    = st.sidebar.slider("tile_rows", 1, 6, int(cfg.tile_rows))
cfg.tile_cols    = st.sidebar.slider("tile_cols", 1, 8, int(cfg.tile_cols))
cfg.tile_overlap = st.sidebar.slider("tile_overlap(px)", 0, 128, int(cfg.tile_overlap), 8)

st.sidebar.header("Calibration (mid band)")
low  = st.sidebar.slider("mid_low (ratio)",  0.0, 0.9,  float(cfg.mid_band[0]), 0.01)
high = st.sidebar.slider("mid_high (ratio)", 0.0, 0.95, float(cfg.mid_band[1]), 0.01)
cfg.mid_band = (min(low, high), max(low, high))
cfg.auto_min_samples = st.sidebar.slider("auto_min_samples", 4, 100, int(cfg.auto_min_samples), 1)
cfg.auto_timeout_sec = float(st.sidebar.slider("auto_timeout_sec", 3.0, 60.0, float(cfg.auto_timeout_sec), 1.0))
cfg.min_h_px   = st.sidebar.slider("min_h_px", 8, 128, int(cfg.min_h_px), 1)
cfg.aspect_min = float(st.sidebar.slider("aspect_min", 0.5, 3.0, float(cfg.aspect_min), 0.05))

st.sidebar.header("Grid / Render")
cfg.base_k_from_person_h = float(st.sidebar.slider("base_k_from_person_h", 0.5, 1.5, float(cfg.base_k_from_person_h), 0.01))
cfg.shrink_per_row       = float(st.sidebar.slider("shrink_per_row",       0.0, 0.2,  float(cfg.shrink_per_row), 0.005))
cfg.min_cell_px          = st.sidebar.slider("min_cell_px", 4, 200, int(cfg.min_cell_px))
cfg.max_cell_px          = st.sidebar.slider("max_cell_px", 40, 300, int(cfg.max_cell_px))
cfg.alert_threshold      = st.sidebar.slider("alert_threshold", 1, 15, int(cfg.alert_threshold))
cfg.fill_alpha           = float(st.sidebar.slider("fill_alpha", 0.1, 0.9, float(cfg.fill_alpha), 0.05))

st.sidebar.header("Runtime")
rt.use_head_point = (st.sidebar.radio("Reference point", ["HEAD","FOOT"], index=0 if rt.use_head_point else 1) == "HEAD")
preset_label = st.sidebar.radio("Preset (α)", ["1 Very big","2 big","3 Normal","4 small","5 Very small"], index=rt.preset_idx)
rt.preset_idx = int(preset_label[0]) - 1
rt.fine_factor = float(st.sidebar.slider("fine_factor (×α)", 0.5, 2.0, float(rt.fine_factor), 0.01))

# ---- Camera geometry (optional) ----
st.sidebar.header("Camera geometry (optional)")
cfg.use_cam_geom = st.sidebar.checkbox("Use camera-geometry grid", value=bool(cfg.use_cam_geom))

# Number inputs: 0 → None 로 저장(미입력 취급)
cfg.cam_height_m  = (st.sidebar.number_input("카메라 높이 H (m)",  value=cfg.cam_height_m  or 0.0, min_value=0.0, format="%.3f") or None)
cfg.cam_pitch_deg = (st.sidebar.number_input("Pitch θ down (+deg)",  value=cfg.cam_pitch_deg or 0.0, min_value=-89.9, max_value=89.9, format="%.2f") or None)

st.sidebar.caption("내부 매개변수(K 또는 FOV 입력)")
cK1, cK2 = st.sidebar.columns(2)
with cK1:
    cfg.fx = (st.number_input("fx (px)", value=cfg.fx or 0.0, min_value=0.0) or None)
    cfg.cx = (st.number_input("cx (px)", value=cfg.cx or 0.0, min_value=0.0) or None)
with cK2:
    cfg.fy = (st.number_input("fy (px)", value=cfg.fy or 0.0, min_value=0.0) or None)
    cfg.cy = (st.number_input("cy (px)", value=cfg.cy or 0.0, min_value=0.0) or None)

st.sidebar.caption("…or FOV if K is unknown")
cfg.fov_h_deg = (st.sidebar.number_input("FOV horizontal (deg)", value=cfg.fov_h_deg or 0.0, min_value=0.0, max_value=179.0) or None)
cfg.fov_v_deg = (st.sidebar.number_input("FOV vertical (deg)",   value=cfg.fov_v_deg or 0.0, min_value=0.0, max_value=179.0) or None)

cfg.target_cell_area_m2 = (st.sidebar.number_input("Target cell area (m²)", value=cfg.target_cell_area_m2 or 0.0, min_value=0.0, format="%.3f") or None)

# --- 빌더 선택 가드(필수값 체크) ---
def _cam_geom_ready() -> bool:
    if not cfg.use_cam_geom:
        return False
    if (cfg.cam_height_m is None) or (cfg.cam_pitch_deg is None) or (cfg.target_cell_area_m2 is None):
        return False
    has_K   = (cfg.fx is not None and cfg.fy is not None and cfg.cx is not None and cfg.cy is not None)
    has_fov = (cfg.fov_h_deg is not None) or (cfg.fov_v_deg is not None)
    return has_K or has_fov

# ============ 사이드바 버튼 ============
c1, c2, c3 = st.sidebar.columns(3)
if c1.button("Open"):
    # 기존 캡처 정리
    if st.session_state.cap:
        try: st.session_state.cap.release()
        except Exception: pass

    st.session_state.cap = open_capture(cfg.source)
    ok, first = (st.session_state.cap.read() if st.session_state.cap and st.session_state.cap.isOpened() else (False, None))
    if not ok:
        st.sidebar.error("첫 프레임을 열 수 없습니다. (source 확인)")
    else:
        st.session_state.H, st.session_state.W = first.shape[:2]
        st.session_state.roi_mgr = RoiManager()
        st.session_state.roi_mgr.ensure_default(st.session_state.H, st.session_state.W)

        st.session_state.model        = ModelRunner(cfg)
        st.session_state.calibr       = AnchorCalibrator(cfg)
        st.session_state.counter      = Counter(cfg)
        st.session_state.renderer     = Renderer(cfg)

        # --- 빌더 선택(기하 기반 가능하면 시도 → 실패시 폴백) ---
        use_geom = False
        if _cam_geom_ready():
            try:
                from crowdgrid.grid_geom import GridBuilderGeom  # 새 파일이 있으면 사용
                st.session_state.grid_builder = GridBuilderGeom(cfg, frame_w=st.session_state.W, frame_h=st.session_state.H)
                use_geom = True
            except Exception as e:
                st.sidebar.warning(f"geom builder 로드 실패(폴백): {e}")
                st.session_state.grid_builder = GridBuilder(cfg, frame_h=st.session_state.H)
        else:
            st.session_state.grid_builder = GridBuilder(cfg, frame_h=st.session_state.H)

        st.session_state.use_geom   = use_geom
        st.session_state.grid_cache = {}

        # 기하 기반이면 캘리브레이션 불필요 → 바로 고정 상태로 보고 그리드 생성
        if st.session_state.use_geom:
            rt.frozen   = True
            rt.unit_mid = None
            # 최초 그리드 생성
            for roi in st.session_state.roi_mgr.list():
                g = st.session_state.grid_builder.build(roi)  # (기하 기반은 unit/alpha 인자 없음)
                st.session_state.grid_cache[roi.id] = g
        else:
            rt.frozen = False
            rt.unit_mid = None

if c2.button("Start"): st.session_state.running = True
if c3.button("Stop"):  st.session_state.running = False

if st.sidebar.button("Re-measure (R)"):
    if not st.session_state.use_geom:
        if st.session_state.calibr: st.session_state.calibr.reset()
        rt.frozen = False; rt.unit_mid = None
    # 기하 기반은 재측정 개념이 없으므로 그리드만 재생성
    st.session_state.grid_cache = {}

# ============ 메인 영역 ============
st.title("Crowd Grid (Streamlit)")
info_col, view_col = st.columns([1,2], gap="large")
with info_col:
    st.markdown("**Status**")
    st.write(f"running: {st.session_state.running}")
    if st.session_state.use_geom:
        st.write("mode: geom-grid (no mid-band calibration)")
        st.write(f"target cell ≈ {cfg.target_cell_area_m2 if cfg.target_cell_area_m2 is not None else '--'} m²")
        unit_txt_for_info = "geom"
    else:
        st.write("mode: mid-band calibration")
        unit_txt_for_info = (f"{rt.unit_mid:.1f}px" if rt.unit_mid is not None else "--")
    st.write(f"unit_mid: {unit_txt_for_info}")
    if st.session_state.calibr and (not st.session_state.use_geom):
        st.write(f"samples: {len(st.session_state.calibr.hbuf_mid)}/{cfg.auto_min_samples}")
    st.write(f"alpha: {rt.alpha():.3f}")
    st.caption("HEAD/FOOT, 프리셋, 미세조정은 사이드바에서 조절.")
frame_holder = view_col.empty()

# ============ 1프레임 처리 ============
def process_one_frame() -> Optional[np.ndarray]:
    cap = st.session_state.cap
    if cap is None or not cap.isOpened(): return None
    ok, frame = cap.read()
    if not ok: return None

    dets = st.session_state.model.infer(frame)

    # --- 빌더 타입별 그리드 준비 ---
    if st.session_state.use_geom:
        # 기하 기반: 캘리브 X, 매 프레임 간단 재빌드(슬라이더 변경 즉시 반영)
        st.session_state.grid_cache.clear()
        for roi in st.session_state.roi_mgr.list():
            g = st.session_state.grid_builder.build(roi)  # unit/alpha 인자 없음
            st.session_state.grid_cache[roi.id] = g
    else:
        # 기존: mid-band 캘리브레이션 → unit_mid 확정 전까진 표본 수집
        if not rt.frozen:
            H, W = st.session_state.H, st.session_state.W
            mid_low, mid_high = int(cfg.mid_band[0]*H), int(cfg.mid_band[1]*H)
            full_roi = st.session_state.roi_mgr.list()[0]  # ensure_default로 첫 ROI는 풀프레임
            for d in dets:
                if d.conf < cfg.conf: continue
                rx, ry = ref_point(d.x1, d.y1, d.x2, d.y2, rt.use_head_point)
                if not (mid_low <= ry <= mid_high): continue
                ix = int(np.clip(rx, 0, W-1)); iy = int(np.clip(ry, 0, H-1))
                inside = (full_roi.mask[iy, ix] > 0)
                st.session_state.calibr.add_sample(d, ry, inside)
            if st.session_state.calibr.ready():
                rt.unit_mid = st.session_state.calibr.finalize(frame_h=H)
                rt.frozen = True
                st.session_state.grid_cache.clear()
                for roi in st.session_state.roi_mgr.list():
                    g = st.session_state.grid_builder.build(roi, unit_mid=rt.unit_mid, alpha=rt.alpha())
                    st.session_state.grid_cache[roi.id] = g

        # 프리즈 후에는 프리셋/미세조정 반영 위해 재빌드
        if rt.frozen and rt.unit_mid is not None:
            st.session_state.grid_cache.clear()
            for roi in st.session_state.roi_mgr.list():
                g = st.session_state.grid_builder.build(roi, unit_mid=rt.unit_mid, alpha=rt.alpha())
                st.session_state.grid_cache[roi.id] = g

    # 카운트 + 렌더
    out = frame.copy()
    for roi in st.session_state.roi_mgr.list():
        g = st.session_state.grid_cache.get(roi.id)
        counts = None
        if g is not None:
            counts = st.session_state.counter.count(
                dets, roi, g,
                use_head_point=rt.use_head_point,
                roi_test_with_ref=True
            )
        out = st.session_state.renderer.draw_roi(out, roi, g, counts)

    # OSD
    if st.session_state.use_geom:
        mode_txt = "geom-grid"
        unit_txt = "unit_mid=-- (geom)"
    else:
        mode_txt = f"measuring... samples={len(st.session_state.calibr.hbuf_mid)}/{cfg.auto_min_samples}" if not rt.frozen else "frozen"
        unit_txt = f"unit_mid={rt.unit_mid:.1f}px" if rt.unit_mid is not None else "unit_mid=--"
    st.session_state.renderer.draw_osd(out, rt, unit_txt, mode_txt)
    return out

# ============ 실행 루프 (rerun) ============
MAX_FRAMES_PER_RUN = 150  # ~5초@30fps 처리 후 rerun
if st.session_state.running and st.session_state.cap is not None:
    for _ in range(MAX_FRAMES_PER_RUN):
        frame_bgr = process_one_frame()
        if frame_bgr is None:
            st.session_state.running = False
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_holder.image(frame_rgb, channels="RGB", use_column_width=True)
        time.sleep(0.01)
    if st.session_state.running:
        st.experimental_rerun()
