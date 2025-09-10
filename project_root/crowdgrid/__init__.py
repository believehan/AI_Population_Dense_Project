# crowdgrid/__init__.py
# -*- coding: utf-8 -*-
"""
crowdgrid: 멀티-ROI 군중 밀집 파이프라인 (모듈화 버전)
"""
from .config import AppConfig, RuntimeState

__all__ = [
    "AppConfig", "RuntimeState"
]
