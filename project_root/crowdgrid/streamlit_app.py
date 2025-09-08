# streamlit_app.py
# -*- coding: utf-8 -*-
import time
from typing import Dict, Optional

import cv2
import numpy as np
import streamlit as st
from importlib import import_module

# 네 모듈 파일명(확장자 .py 제외)
CORE_MODULE = "crowdgrid_core"

core = import_module(CORE_MODULE)
AppConfig        = core.AppConfig
RuntimeState     = core.RuntimeState
open_capture     = core.open_capture
ModelRunner      = core.ModelRunner
RoiManager       = core.RoiManager
AnchorCalibrator = core.AnchorCalibrator
GridBuilder      = core.GridBuilder
Counter          = core.Counter
Renderer         = core.Renderer
ref_point        = core.ref_point

st.set_page_config(page_title="Crowd Grid (Streamlit)", layout="wide")

# ---- 세션 상태 초기화 ----
if "cfg" not in st.session_state: st.session_state.cfg = AppConfig()
if "rt"  not in st.session_state: st.session_state.rt  = RuntimeState()
if "cap" not in st.session_state: st.session_state.cap = None
for key in ("H","W","roi_mgr","model","calibr","grid_builder","counter","renderer"):
    if key not in st.session_state: st.session_state[key] = None
if "grid_cache" not in st.session_state: st.session_state.grid_cache: Dict[int,'Grid'] = {}
if "running"    not in st.session_state: st.session_state.running = False

cfg: AppConfig = st.session_state.cfg
rt:  RuntimeState = st.session_state.rt

# ---- 사이드바 ----
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
cfg.min_h_px = st.sidebar.slider("min_h_px", 8, 128, int(cfg.min_h_px), 1)
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

col1, col2, col3 = st.sidebar.columns(3)
if col1.button("Open"):
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
        st.session_state.grid_builder = GridBuilder(cfg, frame_h=st.session_state.H)
        st.session_state.counter      = Counter(cfg)
        st.session_state.renderer     = Renderer(cfg)
        st.session_state.grid_cache   = {}
        rt.frozen = False; rt.unit_mid = None

if col2.button("Start"): st.session_state.running = True
if col3.button("Stop"):  st.session_state.running = False

if st.sidebar.button("Re-measure (R)"):
    if st.session_state.calibr: st.session_state.calibr.reset()
    rt.frozen = False; rt.unit_mid = None
    st.session_state.grid_cache = {}

# ---- 메인 화면 ----
st.title("Crowd Grid (Streamlit)")
info_col, view_col = st.columns([1,2], gap="large")
with info_col:
    st.markdown("**Status**")
    st.write(f"running: {st.session_state.running}")
    st.write(f"unit_mid: {rt.unit_mid if rt.unit_mid is not None else '--'} px")
    if st.session_state.calibr:
        st.write(f"samples: {len(st.session_state.calibr.hbuf_mid)}/{cfg.auto_min_samples}")
    st.write(f"alpha: {rt.alpha():.3f}")
    st.caption("HEAD/FOOT, 프리셋, 미세조정은 사이드바에서 조절.")
frame_holder = view_col.empty()

def process_one_frame() -> Optional[np.ndarray]:
    cap = st.session_state.cap
    if cap is None or not cap.isOpened(): return None
    ok, frame = cap.read()
    if not ok: return None

    dets = st.session_state.model.infer(frame)

    # 캘리브(프리즈 전)
    if not rt.frozen:
        H, W = st.session_state.H, st.session_state.W
        mid_low, mid_high = int(cfg.mid_band[0]*H), int(cfg.mid_band[1]*H)
        full_roi = st.session_state.roi_mgr.list()[0]
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

    # 프리즈 후엔 프리셋/미세조정 반영 위해 재빌드(간단히 항상)
    if rt.frozen and rt.unit_mid is not None:
        st.session_state.grid_cache.clear()
        for roi in st.session_state.roi_mgr.list():
            g = st.session_state.grid_builder.build(roi, unit_mid=rt.unit_mid, alpha=rt.alpha())
            st.session_state.grid_cache[roi.id] = g

    # 카운트+렌더
    out = frame.copy()
    for roi in st.session_state.roi_mgr.list():
        g = st.session_state.grid_cache.get(roi.id)
        counts = None
        if g is not None:
            counts = st.session_state.counter.count(dets, roi, g, use_head_point=rt.use_head_point, roi_test_with_ref=True)
        out = st.session_state.renderer.draw_roi(out, roi, g, counts)

    # OSD
    mode_txt = f"measuring... samples={len(st.session_state.calibr.hbuf_mid)}/{cfg.auto_min_samples}" if not rt.frozen else "frozen"
    unit_txt = f"unit_mid={rt.unit_mid:.1f}px" if rt.unit_mid is not None else "unit_mid=--"
    st.session_state.renderer.draw_osd(out, rt, unit_txt, mode_txt)
    return out

MAX_FRAMES_PER_RUN = 150
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
