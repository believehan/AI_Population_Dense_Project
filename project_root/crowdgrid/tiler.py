# -*- coding: utf-8 -*-
from typing import List, Tuple
import numpy as np

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
