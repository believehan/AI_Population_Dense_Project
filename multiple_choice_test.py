# -*- coding: utf-8 -*-
"""
멀티-ROI 군중 밀집 파이프라인 (모듈화 버전)
- 설계 합의 사항 반영:
  * YOLO 단 1회/프레임 추론 (타일 오버랩 + 글로벌 NMS)
  * 캘리브레이션: 화면 중간밴드(20~40%)에서 사람 "키(px)" 표본 수집 → robust median → 1회 확정(freeze)
  * 그리드: 프레임 하단 기준 전역 원근 함수로 행별 셀 크기(side)를 결정(옵션 A)
  * 멀티-ROI: ROI 유무 상관없이 동일 인터페이스(ROI 없으면 풀프레임 자동 ROI 1개 생성)
  * 렌더링: ROI 내부만 블렌딩(바깥은 원본 밝기 유지)
  * 키입력: [1..5], +/- (프리셋/파인), H/F (참조점), R(재측정), Q/ESC(종료)

참고(Ultralytics 사용 포인트):
- 모델 로딩: YOLO(MODEL)
- 배치 추론: results = model(list_of_numpy_images, conf=..., imgsz=..., device=..., half=..., classes=[0])
- bbox/점수 접근: for res in results: res.boxes.xyxy, res.boxes.conf

※ 위 인터페이스는 Ultralytics 공식 API에 기반한 일반적인 사용법입니다.
"""

from __future__ import annotations
import cv2, numpy as np, time, math, subprocess, shutil
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
from collections import deque
from ultralytics import YOLO

# =============================
# 0) 공용 타입/유틸
# =============================
@dataclass
class Det:
    x1: float; y1: float; x2: float; y2: float; conf: float; cls: int = 0


def iou_xyxy(a: Tuple[float,float,float,float,float], b: Tuple[float,float,float,float,float]) -> float:
    ax1, ay1, ax2, ay2 = a[:4]; bx1, by1, bx2, by2 = b[:4]
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0: return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / (area_a + area_b - inter + 1e-9)


def nms_global(boxes: List[Tuple[float,float,float,float,float]], iou_th: float=0.55) -> List[Tuple[float,float,float,float,float]]:
    """간단한 전역 NMS. 점수 내림차순 정렬 후, IoU 임계치 이상을 제거."""
    boxes = sorted(boxes, key=lambda x: x[4], reverse=True)
    keep = []
    while boxes:
        cur = boxes.pop(0)
        keep.append(cur)
        boxes = [b for b in boxes if iou_xyxy(cur, b) < iou_th]
    return keep


def ref_point(x1: float, y1: float, x2: float, y2: float, use_head: bool=True) -> Tuple[float,float]:
    """사람 박스의 참조점(머리: top-center / 발끝: bottom-center)."""
    cx = 0.5 * (x1 + x2)
    cy = y1 if use_head else y2
    return cx, cy


def polygon_mask(h: int, w: int, poly: np.ndarray) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [poly.astype(np.int32)], 255)
    return mask


def apply_overlay_in_mask(frame: np.ndarray, overlay: np.ndarray, mask: np.ndarray, alpha: float) -> np.ndarray:
    """ROI 내부만 반투명 합성(바깥은 원본 유지)."""
    blended = cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0)
    out = frame.copy()
    out[mask > 0] = blended[mask > 0]
    return out


# =============================
# 1) 설정/상태
# =============================
@dataclass
class AppConfig:
    # 입력/모델
    youtube_url: str = "https://www.youtube.com/live/rnXIjl_Rzy4?feature=shared"
    model_path: str = "epoch30.pt"
    device: str = "cpu"
    imgsz: int = 640
    conf: float = 0.35  # 탐지 신뢰도 임계치(기본 0.35)
    use_half: bool = False
    classes: List[int] = field(default_factory=lambda: [0])  # person only

    # 타일링
    tile_rows: int = 2
    tile_cols: int = 4
    tile_overlap: int = 32

    # 캘리브레이션(중간밴드)
    mid_band: Tuple[float,float] = (0.20, 0.40) # 표본을 모을 세로 밴드(동영상 화면기준 20%에서 40% 사이에서 사람 크기 확인)
    auto_min_samples: int = 12      # 측정 확정 조건 (기본 12 사람 디텍팅 후 확정)
    auto_timeout_sec: float = 25.0  # 측정 확정 조건 (사람을 auto_min_samples값만큼 잡지 못한 경우 25초 후에 강제 확정)
    min_h_px: int = 32
    aspect_min: float = 1.25

    # 그리드/표시
    base_k_from_person_h: float = 0.90  # 셀 변 = unit_mid × α × K의 K값
    shrink_per_row: float = 0.05 # 위로 갈수록 셀 줄이는 비율(기본 0.05) ex) 행 별로  0.05씩 비율이 줄어들음
    min_cell_px: int = 12
    max_cell_px: int = 120
    alert_threshold: int = 5 # 알림 기준 카운트(기본 5) ex) 5명 이상이면 빨간색으로 표시
    fill_alpha: float = 0.45

    # 디스플레이
    display_max_w: int = 1280
    display_max_h: int = 720
    win_name: str = "Crowd Grid (Modular)"


def compute_fit_scale(w: int, h: int, max_w: int, max_h: int) -> float:
    return min(max_w / w, max_h / h, 1.0)


def show_fit(name: str, img: np.ndarray, max_w: int, max_h: int):
    h, w = img.shape[:2]
    s = compute_fit_scale(w, h, max_w, max_h)
    if s < 1.0:
        img = cv2.resize(img, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA)
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, img)


@dataclass
class RuntimeState:
    use_head_point: bool = True      # HEAD(True)/FOOT(False)
    preset_idx: int = 2              # 0..4 (Very big .. Very small) # α(배수) 조정 값
    fine_factor: float = 1.0         # α(배수) 조정 값
    alpha_values: Tuple[float,...] = (1.95, 1.75, 1.45, 1.20, 1.00)

    unit_mid: Optional[float] = None # 확정된 앵커(px)
    frozen: bool = False
    measure_started: float = field(default_factory=time.time)

    def alpha(self) -> float:
        return self.alpha_values[self.preset_idx] * self.fine_factor


# =============================
# 2) 입력/타일러/모델러
# =============================
class StreamResolver:
    """YouTube → stream URL (streamlink)."""
    @staticmethod
    def resolve(url: str) -> str:
        exe = (shutil.which("streamlink") or r"C:\\Users\\user\\anaconda3\\envs\\py39\\Scripts\\streamlink.exe")
        if not exe:
            raise RuntimeError("streamlink 실행파일을 찾지 못했습니다.")
        out = subprocess.run([exe, "--stream-url", url, "best"], capture_output=True, text=True, check=True)
        s = out.stdout.strip()
        if not s:
            raise RuntimeError("streamlink로 스트림 URL 얻기 실패")
        return s


class Tiler:
    """프레임을 rows×cols 타일로 나누되 경계에 overlap을 적용."""
    def __init__(self, rows: int, cols: int, overlap: int):
        self.rows, self.cols, self.overlap = rows, cols, overlap

    def split(self, frame: np.ndarray) -> Tuple[List[np.ndarray], List[Tuple[int,int]]]:
        h, w = frame.shape[:2]
        hs = [int(round(r*h/self.rows)) for r in range(self.rows+1)]
        ws = [int(round(c*w/self.cols)) for c in range(self.cols+1)]
        crops, offs = [], []
        for r in range(self.rows):
            for c in range(self.cols):
                y1, y2 = hs[r], hs[r+1]
                x1, x2 = ws[c], ws[c+1]
                y1e = max(0, y1 - (self.overlap if r > 0 else 0))
                y2e = min(h, y2 + (self.overlap if r < self.rows-1 else 0))
                x1e = max(0, x1 - (self.overlap if c > 0 else 0))
                x2e = min(w, x2 + (self.overlap if c < self.cols-1 else 0))
                crops.append(frame[y1e:y2e, x1e:x2e])
                offs.append((x1e, y1e))
        return crops, offs


class ModelRunner:
    """Ultralytics YOLO 추론(타일 배치 → 글로벌 NMS)"""
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.model = YOLO(cfg.model_path)
        self.tiler = Tiler(cfg.tile_rows, cfg.tile_cols, cfg.tile_overlap)

    def infer(self, frame: np.ndarray) -> List[Det]:
        crops, offs = self.tiler.split(frame)
        results = self.model(crops, conf=self.cfg.conf, imgsz=self.cfg.imgsz,
                             device=self.cfg.device, half=self.cfg.use_half,
                             classes=self.cfg.classes)
        pooled = []  # (x1,y1,x2,y2,conf)
        for res, (ox, oy) in zip(results, offs):
            boxes = res.boxes
            if boxes is None or len(boxes) == 0:
                continue
            xyxy = boxes.xyxy.cpu().numpy(); confs = boxes.conf.cpu().numpy()
            for (x1,y1,x2,y2), cf in zip(xyxy, confs):
                pooled.append((x1+ox, y1+oy, x2+ox, y2+oy, float(cf)))
        kept = nms_global(pooled, iou_th=0.55)
        return [Det(*b[:4], b[4], 0) for b in kept]


# =============================
# 3) ROI
# =============================
@dataclass
class Roi:
    # 현재 버전은 ROI가 없으면 풀프레임 1개 ROI를 자동 생성해서 동작
    id: int
    name: str
    polygon: np.ndarray  # (N,2)
    mask: np.ndarray     # (H,W) uint8
    bbox: Tuple[int,int,int,int]  # (x,y,w,h)

    @staticmethod
    def full_frame(h: int, w: int, idx: int=0) -> "Roi":
        poly = np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1]], dtype=np.int32)
        mask = polygon_mask(h, w, poly)
        x,y,ww,hh = cv2.boundingRect(poly)
        return Roi(idx, f"FULL_{idx}", poly, mask, (x,y,ww,hh))


class RoiManager:
    def __init__(self):
        self._rois: List[Roi] = []
        self._active_id: Optional[int] = None
        self._next_id: int = 0

    def ensure_default(self, h: int, w: int):
        if not self._rois:
            r = Roi.full_frame(h, w, idx=self._next_id)
            self._rois.append(r)
            self._active_id = r.id
            self._next_id += 1

    # 수동으로 ROI 추가
    def add_polygon(self, poly: np.ndarray, h: int, w: int, name: Optional[str]=None) -> Roi: 
        mask = polygon_mask(h, w, poly)
        x,y,ww,hh = cv2.boundingRect(poly)
        r = Roi(self._next_id, name or f"ROI_{self._next_id}", poly.astype(np.int32), mask, (x,y,ww,hh))
        self._rois.append(r); self._active_id = r.id; self._next_id += 1
        return r

    def remove_active(self):
        if self._active_id is None: return
        self._rois = [r for r in self._rois if r.id != self._active_id]
        self._active_id = self._rois[0].id if self._rois else None

    def cycle_active(self):
        if not self._rois: return
        ids = [r.id for r in self._rois]
        if self._active_id not in ids:
            self._active_id = ids[0]; return
        i = ids.index(self._active_id); self._active_id = ids[(i+1)%len(ids)]

    def list(self) -> List[Roi]:
        return list(self._rois)

    def active(self) -> Optional[Roi]:
        for r in self._rois:
            if r.id == self._active_id:
                return r
        return None


# =============================
# 4) 앵커(중간밴드) 캘리브레이션
# =============================
class AnchorCalibrator:
    """중간밴드(20~40%)에서 사람 높이(h=y2-y1) 표본을 모아 1회 unit_mid 확정."""
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.hbuf_mid: deque = deque(maxlen=300)
        self.reset()

    def reset(self):
        self.hbuf_mid.clear()
        self.started_ts = time.time()
        self.unit_mid: Optional[float] = None

    def add_sample(self, det: Det, ref_y: float, is_inside_roi: bool):
        if not is_inside_roi:
            return
        h = det.y2 - det.y1; w = det.x2 - det.x1
        if h < self.cfg.min_h_px or w <= 0: return
        if (h/(w+1e-9)) < self.cfg.aspect_min: return
        y_low, y_high = int(self.cfg.mid_band[0]*ref_y*0 + 0), int(self.cfg.mid_band[1]*ref_y*0 + 0)  # dummy (unused)
        # 실제 판정은 호출 측에서 수행(참조점 y가 mid_band 안인지)
        self.hbuf_mid.append(h)

    def ready(self) -> bool:
        enough = (len(self.hbuf_mid) >= self.cfg.auto_min_samples)
        timeout = ((time.time() - self.started_ts) > self.cfg.auto_timeout_sec and len(self.hbuf_mid) >= max(5, self.cfg.auto_min_samples//2))
        return enough or timeout

    def finalize(self, frame_h: int) -> float:
        def robust_median(arr: np.ndarray) -> Optional[float]:
            if arr.size == 0: return None
            q10, q90 = np.percentile(arr, [10, 90])
            arr = arr[(arr >= q10) & (arr <= q90)]
            if arr.size == 0: return None
            return float(np.median(arr))
        arr = np.asarray(self.hbuf_mid, dtype=np.float32)
        unit = robust_median(arr)
        if unit is None:
            unit = max(self.cfg.min_h_px, int(0.065 * frame_h))  # 보수적 대체값
        self.unit_mid = unit
        return unit


# =============================
# 5) 그리드(옵션 A: 전역 원근)
# =============================
@dataclass
class Cell:
    poly: np.ndarray  # (4,2) int32
    row_rel: int
    col: int
    side: int


@dataclass
class Grid:
    cells: List[Cell]
    base_side_mid: int  # mid 밴드 기준 행의 셀 변

    def locate(self, x: float, y: float) -> Optional[int]:
        # 빠른 AABB 체크(정사각형이므로 좌표 비교)
        for idx, c in enumerate(self.cells):
            p = c.poly
            if (x >= p[0,0]) and (x <= p[1,0]) and (y > p[0,1]) and (y <= p[2,1]):
                return idx
        return None


class GridBuilder:
    def __init__(self, cfg: AppConfig, frame_h: int):
        self.cfg = cfg
        self.H = frame_h

    def build(self, roi: Roi, unit_mid: float, alpha: float) -> Grid:
        # 1) mid 기준 셀 변 계산
        base_side_mid = int(max(self.cfg.min_cell_px, min(self.cfg.max_cell_px, round(self.cfg.base_k_from_person_h * unit_mid * alpha))))
        cells: List[Cell] = []
        x, y, w, h = roi.bbox
        y_bot = y + h
        # ROI의 하단이 프레임 하단에서 몇 번째 행인지(전역)
        row0 = max(0, int(math.floor((self.H - y_bot) / max(1, base_side_mid))))
        row = 0
        while True:
            row_rel = row0 + row
            # 행별 side(y)
            side = max(self.cfg.min_cell_px, int(round(base_side_mid * ((1.0 - self.cfg.shrink_per_row) ** row_rel))))
            y_top = y_bot - side
            if y_top < y:
                break
            x0 = x; col = 0
            while x0 < x + w:
                poly = np.array([[x0, y_top],[x0+side, y_top],[x0+side, y_bot],[x0, y_bot]], dtype=np.int32)
                cells.append(Cell(poly=poly, row_rel=row_rel, col=col, side=side))
                x0 += side; col += 1
            y_bot = y_top; row += 1
        return Grid(cells=cells, base_side_mid=base_side_mid)


# =============================
# 6) 카운터
# =============================
class Counter:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg

    def count(self, dets: List[Det], roi: Roi, grid: Grid, use_head_point: bool, roi_test_with_ref: bool=True) -> Dict[int,int]:
        counts: Dict[int,int] = {}
        H, W = roi.mask.shape
        for d in dets:
            if d.conf < self.cfg.conf: continue
            rx, ry = ref_point(d.x1, d.y1, d.x2, d.y2, use_head_point)
            # ROI 포함 검사(참조점 기준)
            testx, testy = (rx, ry) if roi_test_with_ref else ref_point(d.x1, d.y1, d.x2, d.y2, use_head=False)
            ix = int(min(max(testx, 0), W-1)); iy = int(min(max(testy, 0), H-1))
            if roi.mask[iy, ix] == 0: continue
            idx = grid.locate(rx, ry)
            if idx is None: continue
            counts[idx] = counts.get(idx, 0) + 1
        return counts


# =============================
# 7) 렌더러(OSD 포함)
# =============================
class Renderer:
    FONT = cv2.FONT_HERSHEY_SIMPLEX

    def __init__(self, cfg: AppConfig):
        self.cfg = cfg

    def draw_roi(self, frame: np.ndarray, roi: Roi, grid: Optional[Grid], counts: Optional[Dict[int,int]]) -> np.ndarray:
        overlay = frame.copy()
        if grid is not None:
            for i, c in enumerate(grid.cells):
                poly = c.poly
                # ROI의 실제 폴리곤 내부만 그리도록 하려면 교차 판정 추가 가능(단, 비용↑)
                n = counts.get(i, 0) if counts is not None else 0
                if n >= self.cfg.alert_threshold: color=(0,0,255)
                elif n > 0:                      color=(0,180,0)
                else:                             color=(180,180,180)
                if c.side >= self.cfg.min_cell_px:
                    cv2.fillPoly(overlay, [poly], color)
                    cv2.polylines(overlay, [poly], True, (50,50,50), 1)
        out = apply_overlay_in_mask(frame, overlay, roi.mask, self.cfg.fill_alpha)
        cv2.polylines(out, [roi.polygon], True, (0,200,255), 2)
        return out

    def draw_osd(self, img: np.ndarray, rt: RuntimeState, unit_mid_txt: str, mode_txt: str):
        # 상단 상태 텍스트
        cv2.putText(img, f"{mode_txt}", (12,28), self.FONT, 0.7, (0,0,0), 3)
        cv2.putText(img, f"{mode_txt}", (12,28), self.FONT, 0.7, (255,255,255), 1)
        ref_txt = "HEAD" if rt.use_head_point else "FOOT"
        line = f"{unit_mid_txt}  x{rt.alpha():.2f}  shrink/row={cfg.shrink_per_row:.3f}  ref={ref_txt} [H/F]"
        cv2.putText(img, line, (12,52), self.FONT, 0.7, (0,0,0), 3)
        cv2.putText(img, line, (12,52), self.FONT, 0.7, (255,255,255), 1)
        cv2.putText(img, "Manual: [1..5]=preset  +/-=size  R=remeasure  H/F=head/foot", (12,76), self.FONT, 0.7, (0,0,0), 3)
        cv2.putText(img, "Manual: [1..5]=preset  +/-=size  R=remeasure  H/F=head/foot", (12,76), self.FONT, 0.7, (255,255,255), 1)


# =============================
# 8) 메인 루프
# =============================
if __name__ == "__main__":
    cfg = AppConfig()
    rt  = RuntimeState()

    # 스트림 오픈(YouTube → streamlink)
    stream_url = StreamResolver.resolve(cfg.youtube_url)
    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        raise RuntimeError("VideoCapture 열기 실패")

    ok, first = cap.read()
    if not ok:
        raise RuntimeError("첫 프레임 읽기 실패")
    H, W = first.shape[:2]

    # ROI 준비(없으면 풀프레임 1개 자동 생성)
    roi_mgr = RoiManager()
    roi_mgr.ensure_default(H, W)

    # 모듈 준비
    model = ModelRunner(cfg)
    calibr = AnchorCalibrator(cfg)
    grid_builder = GridBuilder(cfg, frame_h=H)
    counter = Counter(cfg)
    renderer = Renderer(cfg)

    grid_cache: Dict[int, Grid] = {}  # roi.id → Grid

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        # 1) 추론(1회/프레임)
        dets = model.infer(frame)

        # 2) 측정 단계: mid 밴드에서 표본 수집 (freeze 전)
        if not rt.frozen:
            mid_low, mid_high = int(cfg.mid_band[0]*H), int(cfg.mid_band[1]*H)
            for d in dets:
                if d.conf < cfg.conf: continue
                rx, ry = ref_point(d.x1, d.y1, d.x2, d.y2, rt.use_head_point)
                if not (mid_low <= ry <= mid_high):
                    continue
                # ROI 포함 여부: 전역 앵커이므로 활성 ROI가 아닌, 기본 ROI(풀프레임)로 판정 OK
                r0 = roi_mgr.active() or roi_mgr.list()[0]
                ix = int(min(max(rx, 0), W-1)); iy = int(min(max(ry, 0), H-1))
                inside = (r0.mask[iy, ix] > 0)
                calibr.add_sample(d, ry, inside)

            # 3) 조건 충족 시 앵커 확정 → 모든 ROI 그리드 생성
            if calibr.ready():
                rt.unit_mid = calibr.finalize(frame_h=H)
                rt.frozen = True
                grid_cache.clear()
                for roi in roi_mgr.list():
                    g = grid_builder.build(roi, unit_mid=rt.unit_mid, alpha=rt.alpha())
                    grid_cache[roi.id] = g

        # 프리셋/± 조정은 freeze 이후 바로 재빌드
        def rebuild_all():
            grid_cache.clear()
            for roi in roi_mgr.list():
                g = grid_builder.build(roi, unit_mid=rt.unit_mid, alpha=rt.alpha())
                grid_cache[roi.id] = g

        # 4) 카운트 + 렌더링(ROI별)
        out = frame.copy()
        for roi in roi_mgr.list():
            g = grid_cache.get(roi.id)
            counts = None
            if g is not None:
                counts = counter.count(dets, roi, g, use_head_point=rt.use_head_point, roi_test_with_ref=True)
            out = renderer.draw_roi(out, roi, g, counts)

        # OSD
        mode_txt = f"measuring... samples={len(calibr.hbuf_mid)}/{cfg.auto_min_samples}" if not rt.frozen else "frozen"
        unit_txt = f"unit_mid={rt.unit_mid:.1f}px" if rt.unit_mid is not None else "unit_mid=--"
        renderer.draw_osd(out, rt, unit_txt, mode_txt)

        # 디스플레이
        show_fit(cfg.win_name, out, cfg.display_max_w, cfg.display_max_h)

        # 5) 키 입력
        k = cv2.waitKey(1) & 0xFF
        if k in (27, ord('q'), ord('Q')):
            break
        elif k in (ord('h'), ord('H')):
            rt.use_head_point = True
        elif k in (ord('f'), ord('F')):
            rt.use_head_point = False
        elif k in (ord('1'), ord('2'), ord('3'), ord('4'), ord('5')):
            if rt.frozen and rt.unit_mid is not None:
                rt.preset_idx = int(chr(k)) - 1
                rebuild_all()
        elif k in (ord('+'), ord('=')):
            if rt.frozen and rt.unit_mid is not None:
                rt.fine_factor *= 1.05
                rebuild_all()
        elif k in (ord('-'), ord('_')):
            if rt.frozen and rt.unit_mid is not None:
                rt.fine_factor /= 1.05
                rebuild_all()
        elif k in (ord('r'), ord('R')):
            # 재측정: 앵커/그리드 초기화
            calibr.reset(); rt.frozen = False; rt.unit_mid = None; rt.measure_started = time.time(); grid_cache.clear()

    cap.release()
    cv2.destroyAllWindows()
