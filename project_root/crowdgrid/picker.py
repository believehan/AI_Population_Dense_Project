# -*- coding: utf-8 -*-
"""
좌표 픽커: 영상 위에 마우스로 '사각형 ROI'를 드래그해서 찍고,
zone_coords 딕셔너리(절대 px)와 HTML 퍼센트 좌표를 출력/저장합니다.

사용 예)
python -m crowdgrid.picker --source static/videos/bucheon.mp4 --count 4 --out zones_bucheon.py
"""
import argparse, sys, os
import cv2
import numpy as np

def _open_capture(src: str) -> cv2.VideoCapture:
    # 숫자면 웹캠, 파일/URL은 OpenCV에 위임 (필요시 streams.open_capture로 바꿔도 됨)
    if str(src).isdigit():
        return cv2.VideoCapture(int(src))
    return cv2.VideoCapture(src)

class RectPicker:
    def __init__(self, frame):
        self.h, self.w = frame.shape[:2]
        self.zones = []          # [(x1,y1,x2,y2), ...]
        self.drawing = False
        self.p0 = None
        self.p1 = None
        self.preview = frame.copy()

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.p0 = (x, y)
            self.p1 = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.p1 = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and self.drawing:
            self.drawing = False
            x1, y1 = self.p0
            x2, y2 = self.p1
            x1, x2 = sorted([x1, x2]); y1, y2 = sorted([y1, y2])
            if (x2 - x1) >= 2 and (y2 - y1) >= 2:
                self.zones.append((x1, y1, x2, y2))
            self.p0 = None; self.p1 = None

    def draw(self, frame):
        img = frame.copy()
        # 이미 확정된 존
        for i, (x1, y1, x2, y2) in enumerate(self.zones, start=1):
            cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,255), 2)
            cv2.putText(img, f"ZONE {i}", (x1, max(0,y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
        # 드래그 미리보기
        if self.drawing and self.p0 and self.p1:
            x1, y1 = self.p0; x2, y2 = self.p1
            cv2.rectangle(img, (x1, y1), (x2, y2), (0,200,255), 1)
        return img

    def export_py(self, out_path=None):
        # 절대 px
        lines = []
        lines.append("zone_coords = {")
        for i, (x1, y1, x2, y2) in enumerate(self.zones, start=1):
            lines.append(f"    {i}: ({x1}, {y1}, {x2}, {y2}),")
        lines.append("}\n")

        # 퍼센트(HTML 오버레이용)
        lines.append("# HTML overlay(%) 예시 좌표")
        lines.append("zone_coords_pct = {")
        for i, (x1, y1, x2, y2) in enumerate(self.zones, start=1):
            left   = 100.0 * x1 / self.w
            top    = 100.0 * y1 / self.h
            width  = 100.0 * (x2 - x1) / self.w
            height = 100.0 * (y2 - y1) / self.h
            lines.append(f"    {i}: {{'left':'{left:.1f}%', 'top':'{top:.1f}%', 'width':'{width:.1f}%', 'height':'{height:.1f}%'}},")
        lines.append("}\n")

        text = "\n".join(lines)
        if out_path:
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"[Saved] {out_path}")
        print("\n==== paste this into your Flask app ====\n")
        print(text)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="영상 파일 경로 또는 웹캠 인덱스(숫자)")
    ap.add_argument("--count", type=int, default=4, help="찍을 구역(사각형) 개수")
    ap.add_argument("--out", type=str, default=None, help="저장할 .py 파일 경로(예: zones_*.py)")
    args = ap.parse_args()

    cap = _open_capture(args.source)
    if not cap.isOpened():
        print(f"[ERR] Cannot open source: {args.source}")
        sys.exit(1)

    ok, frame = cap.read()
    if not ok:
        print("[ERR] Cannot read first frame")
        sys.exit(1)

    picker = RectPicker(frame)
    win = "Rect Picker"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, picker.on_mouse)

    info = "[Drag=LMB] 사각형, [u]=되돌리기, [s]=저장, [q/ESC]=종료"
    print(info)

    while True:
        ok, frame = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = cap.read()
            if not ok:
                break

        img = picker.draw(frame)
        cv2.putText(img, f"zones: {len(picker.zones)} / target: {args.count}", (12, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3)
        cv2.putText(img, f"zones: {len(picker.zones)} / target: {args.count}", (12, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1)
        cv2.putText(img, info, (12, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3)
        cv2.putText(img, info, (12, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 1)

        cv2.imshow(win, img)
        k = cv2.waitKey(1) & 0xFF
        if k in (27, ord('q'), ord('Q')):
            break
        elif k in (ord('u'), ord('U')) and picker.zones:
            picker.zones.pop()
        elif k in (ord('s'), ord('S')):
            picker.export_py(args.out)
        # 자동 완료: 목표 개수 채우면 저장 후 종료
        if len(picker.zones) >= args.count:
            picker.export_py(args.out)
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
