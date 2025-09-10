# -*- coding: utf-8 -*-
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import time

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
    source: str = "AI_Population_Dense_Project/buchoen_station_square_3.mp4" # Streamlink로 여는 YouTube 주소 또는 파일경로. 소스 교체 시만 변경.
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
    
    # === 카메라 기하 기반 스케일 모드 ===
    use_cam_geom: bool = False               # 체크되면 기하 기반 모드 사용
    cam_height_m: Optional[float] = None     # 카메라 높이 H (m)
    cam_pitch_deg: Optional[float] = None    # 피치각 θ (아래로 +deg)

    # 내부파라미터(직접 입력용). 전부 None이면 FOV로부터 fx/fy를 계산 시도
    fx: Optional[float] = None
    fy: Optional[float] = None
    cx: Optional[float] = None
    cy: Optional[float] = None

    # FOV 입력용(선택): fx/fy가 None일 때만 사용
    fov_h_deg: Optional[float] = None
    fov_v_deg: Optional[float] = None

    # 목표 셀 실면적(m^2). None이면 기하 모드 사용 불가
    target_cell_area_m2: Optional[float] = None


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
