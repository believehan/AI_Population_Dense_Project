# -*- coding: utf-8 -*-
from typing import Dict, List
from .config import AppConfig
from .types import Det
from .roi import Roi
from .grid import Grid
from .utils import ref_point

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
