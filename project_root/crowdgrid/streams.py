# -*- coding: utf-8 -*-
import subprocess, shutil
from urllib.parse import urlparse
from pathlib import Path
import cv2

class StreamResolver:
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