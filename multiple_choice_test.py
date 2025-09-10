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
from urllib.parse import urlparse
from pathlib import Path

# =============================
# 0) 공용 타입/유틸
# =============================

@dataclass
class Det:
    """단일 탐지 결과를 표현하는 가벼운 컨테이너.
    
    좌표계: 이미지 픽셀 좌표, 원점(0,0)은 좌상단, x→오른쪽(+), y→아래(+)
    - x1, y1: 박스 좌상단 (top-left)
    - x2, y2: 박스 우하단 (bottom-right)
    - conf  : 신뢰도 점수(0.0~1.0)
    - cls   : 클래스 인덱스(여기선 person=0 고정)
    
    주의: float로 보관해 후처리(스케일 변환, NMS 등) 오차 누적을 줄임.
    """
    x1: float; y1: float; 
    x2: float; y2: float; 
    conf: float; 
    cls: int = 0

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

# =============================
# 1) 설정/상태
# =============================
@dataclass
class AppConfig:
    """
    사용법 
    
    오검출 많다 → conf↑, min_h_px↑, aspect_min↑. (필요시 classes=[0] 확인) 

    셀 크기 과/소 → 프리셋(1..5) 바꾸고 fine_factor로 미세 조정, 또는 base_k_from_person_h 조절.

    경계 누락/중복 → tile_overlap 조금↑; 중복은 NMS가 정리.

    위쪽 셀 너무 작다/크다 → shrink_per_row 조정.

    GPU인데 속도 더 → device="cuda", use_half=True(지원 GPU 한정).
    """
    # 입력/모델
    # source: str = "https://www.youtube.com/live/rnXIjl_Rzy4?feature=shared" # Streamlink로 여는 YouTube 주소 또는 파일경로. 소스 교체 시만 변경.
    source: str = "./test_bucheon.mp4" # Streamlink로 여는 YouTube 주소 또는 파일경로. 소스 교체 시만 변경.
    model_path: str = "epoch40.pt"
    device: str = "cpu" # "cpu" 또는 "cuda"/"cuda:0". GPU 사용 시 "cuda" 권장. (예측 인자에 device 지원)
    imgsz: int = 640 # 입력 리사이즈 크기. 기본 640. 작게 하면 빠르고, 크게 하면 정확도↑(속도↓). (예측 인자 imgsz)
    conf: float = 0.35  # 탐지 최소 신뢰도 임계치. 낮추면 더 많이 잡히지만 오검출↑. (예측 인자 conf)
    use_half: bool = False # FP16(half) 추론 여부. 지원 GPU에서만 속도 이점. CPU에선 효과 없음. (예측 인자 half)
    classes: List[int] = field(default_factory=lambda: [0])  
    # 탐지할 클래스 ID 필터. [0]이면 COCO의 person만. 커스텀 데이터셋이면 클래스 맵 확인 필요. (예측 인자에 classes 지원: 특정 클래스만 반환)

    # 타일링
    '''
    tile_rows, tile_cols: 프레임을 몇 행×몇 열로 나눠 추론할지. 커다란 프레임/원거리 소물체에 유리.
    tile_overlap: 타일 경계에서 잘리는 걸 줄이려고 겹치는 폭(px). 너무 작으면 경계 누락, 너무 크면 중복↑(하지만 뒤에서 글로벌 NMS로 정리함).
    '''
    tile_rows: int = 2
    tile_cols: int = 4
    tile_overlap: int = 32

    # 캘리브레이션(중간밴드)
    mid_band: Tuple[float,float] = (0.20, 0.40) # 사람 키 표본을 모을 세로 밴드(비율). (0.20, 0.40) = 화면 높이의 20~40% 구간.
    auto_min_samples: int = 12     # freeze 전까지 모아야 하는 샘플 수. 사람이 적으면 낮춰도 됨.
    auto_timeout_sec: float = 25.0 # 시간이 지나면 모인 샘플로 강제 확정.
    min_h_px: int = 32             # 표본으로 인정할 최소 사람 높이(px). 너무 낮으면 왜곡/오검출 유입.
    aspect_min: float = 1.25   # 사람 박스 세로/가로 비 최소값 필터. 사람 아닌 얇은 물체 걸러내는 용도.

    # 그리드/표시
    base_k_from_person_h: float = 0.90 # 셀 한 변 기준 = (중간밴드 사람키 px) × α × K에서의 K값. 기본 스케일 앵커
    shrink_per_row: float = 0.05 # 위로 갈수록 셀 줄이는 비율. 0.05=행당 5% 축소
    # min_cell_px / max_cell_px: 셀 한 변 하한/상한(px) 클램프.
    min_cell_px: int = 12   
    max_cell_px: int = 120
    alert_threshold: int = 5 # 셀 내 사람 수가 이 값 이상이면 “경고 색”으로 표시.
    fill_alpha: float = 0.45 # ROI 내부 오버레이 투명도.

    # 디스플레이
    # 표시용 리사이즈 상한. 너무 큰 원본도 보기 좋게 축소.
    display_max_w: int = 1280
    display_max_h: int = 720
    win_name: str = "Crowd Grid (Modular)" # OpenCV 창 이름.


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
    use_head_point: bool = True      # HEAD(True) / FOOT(False) 참조점(머리 또는 발을 기준으로 카운트). 카운트는 “참조점이 들어간 단 한 셀”에만 +1.
    preset_idx: int = 2              # 프리셋 5단(화면에 보이는 사람키가 아주 큼→아주 작음). 셀 스케일 배수 α의 인덱스.
    fine_factor: float = 1.0         # α에 미세 조정 곱(키보드 +/-).
    alpha_values: Tuple[float,...] = (1.95, 1.75, 1.45, 1.20, 1.00) # 프리셋별 α 테이블. 최종 α = alpha_values[preset_idx] × fine_factor.

    unit_mid: Optional[float] = None # 캘리브레이션으로 확정된 사람 키(px). freeze 전엔 None, 확정되면 수치 저장.
    frozen: bool = True   # True면 그리드 고정(재측정 전까지 유지).
    measure_started: float = field(default_factory=time.time) # 캘리브레이션 시작 시각(타임아웃 계산용).

    def alpha(self) -> float: # 현재 최종 배수 α를 산출(프리셋×미세조정).
        return self.alpha_values[self.preset_idx] * self.fine_factor


# =============================
# 2) 입력/타일러/모델러
# =============================
class StreamResolver:
        
    def open_capture(src: str) -> cv2.VideoCapture:
        """
        입력 소스를 유연하게 여는 헬퍼:
        - 로컬 파일 경로  → cv2.VideoCapture(파일)
        - 숫자 문자열     → cv2.VideoCapture(웹캠 인덱스)
        - http/https URL → streamlink 로 시도 후 실패시 직접 URL 시도
        - 그 외           → OpenCV 기본 처리
        """
        p = Path(src)
        if p.exists():                      # 로컬 파일
            return cv2.VideoCapture(str(p))

        if src.isdigit():                   # 웹캠 인덱스
            return cv2.VideoCapture(int(src))

        u = urlparse(src)
        if u.scheme in ("http", "https"):   # URL (YouTube 포함)
            try:
                stream_url = StreamResolver.resolve(src)  # streamlink
                cap = cv2.VideoCapture(stream_url)
                if cap.isOpened():
                    return cap
            except Exception:
                pass
            # streamlink 실패시 직접 URL로 시도 (직접 mp4 주소 등)
            return cv2.VideoCapture(src)

        # 기타 케이스는 OpenCV에 위임
        return cv2.VideoCapture(src)
    
    """YouTube → stream URL (streamlink). 유튜브 실시간 영상 실행 코드"""
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
    """
        큰 프레임을 rows×cols 타일로 균등 분할한다.

    - 경계에 `overlap`(px)을 더해 타일 경계에서 객체가 잘리는 문제를 완화한다.
    - 분할 결과로 타일 영상 목록(`crops`)과 각 타일의 원본 기준 좌상단 오프셋(`offs=(ox, oy)`)을 함께 반환한다.
      → 타일 좌표의 박스를 전역 좌표로 복원할 때 `(x+ox, y+oy)`로 사용.

    매개변수
      - rows    : 세로 타일 수
      - cols    : 가로 타일 수
      - overlap : 타일 경계 양쪽에 추가할 겹침 폭(px)

    주의
      - 마지막 행·열에서도 프레임 크기를 벗어나지 않도록 좌표를 클램프한다.
      - `overlap`이 클수록 중복 탐지가 늘 수 있으나, 이후 전역 NMS에서 정리된다.
    """
    def __init__(self, rows: int, cols: int, overlap: int):
        self.rows, self.cols, self.overlap = rows, cols, overlap

    def split(self, frame: np.ndarray) -> Tuple[List[np.ndarray], List[Tuple[int,int]]]:
        """
        프레임을 타일로 잘라 `(crops, offs)`를 반환한다.

        반환
          - `crops[i]` : `frame[y1e:y2e, x1e:x2e]` (overlap 적용된 타일 영상)
          - `offs[i]`  : `(x1e, y1e)` — 타일의 전역 좌상단 오프셋(타일→전역 좌표 복원용)

        알고리즘
          1) 균등 분할 경계선 배열 `hs`, `ws` 생성
          2) 각 셀을 `overlap` 만큼 확장해 경계 잘림 완화
          3) 확장된 좌표를 프레임 크기 안으로 클램프 후 누적

        복잡도: O(rows×cols)
        """
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
    """YOLO 추론 파이프라인(타일 배치 → 전역 NMS).
    - 한 프레임에 대해 모델 추론은 딱 1회만 수행한다.
    - 타일별 결과를 전역 좌표로 복원 후, 중복(NMS)까지 마친 Det 리스트를 반환한다.
    """
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.model = YOLO(cfg.model_path)
        self.tiler = Tiler(cfg.tile_rows, cfg.tile_cols, cfg.tile_overlap)

    def infer(self, frame: np.ndarray) -> List[Det]:
        """
        한 프레임을 타일로 나눠 배치 추론하고, 전역 NMS로 중복을 정리해
        최종 사람 박스들만 전역 좌표계로 반환한다.

        역할
        - 큰 프레임을 rows×cols 타일로 나눠 모델에 배치 입력
        - 타일 경계 중복(같은 사람을 여러 타일이 잡은 경우) → 전역 NMS로 하나만 남김

        입력
        - frame: BGR numpy 이미지(H×W×3)

        출력
        - List[Det]: 전역 좌표계 xyxy(픽셀), conf, cls(=0)이 채워진 감지 결과 목록

        처리 순서
        1) 타일 분할 → (crops, offs)  # offs=(ox,oy)는 타일→전역 좌표 복원용
        2) model(crops, conf, imgsz, device, half, classes)  # 배치 추론
        3) 각 타일 박스 좌표에 offs를 더해 전역 좌표로 변환해 모음
        4) nms_global(...)로 중복 제거(타일 경계에서 동일 인물 중복 제거)

        주의/팁
        - GPU: device='cuda' + use_half=True(지원 GPU)에서 속도 향상
        - classes=[0]이면 person만 필터링(커스텀 데이터셋이면 해당 ID 확인)
        - conf/imgsz는 Ultralytics 예측 인자로 정상 지원됨
        
        # [요약] 타일 배치 추론 → 전역 좌표 복원 → 전역 NMS로 중복 제거 → List[Det] 반환
        """
        crops, offs = self.tiler.split(frame)  # 타일 영상들과 각 타일의 전역 좌상단 오프셋(ox,oy)
        # Ultralytics YOLO 배치 추론: 각 crop에 대한 Results가 리스트로 반환됨
        results = self.model(
            crops,
            conf=self.cfg.conf,      # 최소 신뢰도
            imgsz=self.cfg.imgsz,    # 입력 리사이즈
            device=self.cfg.device,  # 'cpu' 또는 'cuda'
            half=self.cfg.use_half,  # FP16(지원 GPU에서만)
            classes=self.cfg.classes # 특정 클래스만(여기선 person=0)
        )
        pooled = []  # 전역 좌표계 박스 풀: (x1,y1,x2,y2,conf)
        for res, (ox, oy) in zip(results, offs):
            boxes = res.boxes
            if boxes is None or len(boxes) == 0:
                continue
            xyxy = boxes.xyxy.cpu().numpy()    # 타일 좌표계의 박스
            confs = boxes.conf.cpu().numpy()   # 신뢰도
            # 타일→전역 좌표 복원: (x+ox, y+oy)
            for (x1, y1, x2, y2), cf in zip(xyxy, confs):
                pooled.append((x1 + ox, y1 + oy, x2 + ox, y2 + oy, float(cf)))
                
        # 전역 NMS: 타일 경계 중복(같은 사람 여러 박스)을 IoU 기준으로 하나만 남김
        kept = nms_global(pooled, iou_th=0.55)

        # Det 리스트로 변환(전역 좌표계)
        return [Det(*b[:4], b[4], 0) for b in kept]


# =============================
# 3) ROI
# =============================
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

# =============================
# 4) 앵커(중간밴드) 캘리브레이션
# =============================
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

# =============================
# 5) 그리드(옵션 A: 전역 원근)
# =============================
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


# =============================
# 6) 카운터
# =============================
class Counter:
    """
    전역 NMS가 끝난 감지들(Det 리스트)을 받아, ROI 내부이면서
    참조점(HEAD/FOOT) '한 점'이 포함된 **딱 한 셀**에만 +1을 누적한다.

    정책(중요)
      - 박스 vs 셀 **겹침 비율**로 분배하지 않는다.
      - 경계에 걸린 경우도 **참조점이 들어간 셀**만 카운트(+1).
      - IoU는 여기서 사용하지 않음(NMS 단계에서만 사용됨).

    확장 포인트(필요 시 커스텀)
      - 최대 겹침 셀 할당: 박스∩셀 면적이 가장 큰 셀에 +1
      - 분수 카운트: 박스∩셀 비율만큼 0~1 가중치로 분배
      - ROI별 임계치/색상/경고선 등 정책 분리
    """
    # 셀 할당 정책(중요):
    #  - 박스 vs 셀 겹침 면적(비율)로 배분하지 않는다.
    #  - 사람 박스에서 뽑은 참조점(HEAD=윗변 중앙 / FOOT=아랫변 중앙) '한 점'이 포함된
    #    '딱 한 셀'에만 +1을 한다. 경계에 걸린 경우도 참조점이 들어간 셀로만 카운트.
    #  - (확장 가능) 만약 '최대 겹침 셀에 할당'이나 '분수 카운트(겹침 비율로 분배)'
    #    방식이 필요하면 여기 로직을 교체하면 된다.
    
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg

    def count(self, dets: List[Det], roi: Roi, grid: Grid, use_head_point: bool, roi_test_with_ref: bool=True) -> Dict[int,int]:
        """
        셀별 인원수를 계산해 반환한다.

        인자
          - dets            : 전역 NMS 이후의 감지 목록(중복 제거 완료)
          - roi             : 검사 대상 ROI(폴리곤/마스크/bbox 포함)
          - grid            : 해당 ROI에 대해 생성된 정사각형 셀 그리드
          - use_head_point  : 참조점 선택(True=HEAD, False=FOOT)
          - roi_test_with_ref: ROI 포함 판정을 '참조점과 같은 점'으로 할지 여부
                               · True  → HEAD/FOOT 중 선택된 그 점으로 판정
                               · False → ROI 포함 여부는 항상 'FOOT' 기준으로 판정

        반환
          - counts: Dict[셀인덱스(int) → 인원수(int)]

        절차
          1) 신뢰도 필터(cfg.conf 미만 스킵)
          2) 참조점(HEAD/FOOT) 좌표 계산
          3) ROI 마스크로 내부 여부 판정(프레임 범위로 좌표 클램프)
          4) grid.locate(rx,ry)로 셀 인덱스 조회 → 해당 셀에 +1
        """
        counts: Dict[int,int] = {}
        
        # 마스크 해상도(H,W) — 인덱스 접근 전에 좌표 클램프에 사용
        H, W = roi.mask.shape
        
        for d in dets:
            # (0) 신뢰도 안전망: 감지 단계에서도 conf를 걸었지만 여기서 한 번 더 필터
            if d.conf < self.cfg.conf: 
                continue
            # (1) 참조점(HEAD/FOOT) 계산
            rx, ry = ref_point(d.x1, d.y1, d.x2, d.y2, use_head_point)
            
            # (2) ROI 포함 여부 검사
            #     - roi_test_with_ref=True : 참조점 자체로 ROI 포함 판정
            #     - roi_test_with_ref=False: 포함 판정은 'FOOT' 기준(과거 관행 유지 옵션)
            testx, testy = (rx, ry) if roi_test_with_ref else ref_point(d.x1, d.y1, d.x2, d.y2, use_head=False)
            
            # 프레임 경계로 인덱스 클램프 (배열 인덱스 오류 방지)
            ix = int(min(max(testx, 0), W-1)); iy = int(min(max(testy, 0), H-1))
            
            # ROI 밖이면 스킵
            if roi.mask[iy, ix] == 0: continue
            
            # (3) 셀 찾기(정사각형 AABB 비교 기반의 빠른 locate)
            idx = grid.locate(rx, ry)
            if idx is None: continue
            
            # (4) 카운트 누적
            counts[idx] = counts.get(idx, 0) + 1
        return counts


# =============================
# 7) 렌더러(OSD 포함)
# =============================
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
        line = f"{unit_mid_txt}  x{rt.alpha():.2f}  shrink/row={cfg.shrink_per_row:.3f}  ref={ref_txt} [H/F]"
        cv2.putText(img, line, (12,52), self.FONT, 0.7, (0,0,0), 3)
        cv2.putText(img, line, (12,52), self.FONT, 0.7, (255,255,255), 1)
        
        # 3) 조작법 힌트
        cv2.putText(img, "Manual: [1..5]=preset  +/-=size  R=remeasure  H/F=head/foot", (12,76), self.FONT, 0.7, (0,0,0), 3)
        cv2.putText(img, "Manual: [1..5]=preset  +/-=size  R=remeasure  H/F=head/foot", (12,76), self.FONT, 0.7, (255,255,255), 1)


# =============================
# 8) 메인 루프
# =============================
if __name__ == "__main__":
    cfg = AppConfig()
    rt  = RuntimeState()

    # 0) 스트림/파일/웹캠 오픈 (유연 입력)
    cap = StreamResolver.open_capture(cfg.source)
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

    # 5) 자원 정리
    cap.release()
    cv2.destroyAllWindows()