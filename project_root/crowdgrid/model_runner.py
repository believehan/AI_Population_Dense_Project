# -*- coding: utf-8 -*-
from typing import List
import numpy as np
from ultralytics import YOLO
from .config import AppConfig
from .tiler import Tiler
from .types import Det
from .utils import nms_global


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