# -*- coding: utf-8 -*-
from dataclasses import dataclass

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
