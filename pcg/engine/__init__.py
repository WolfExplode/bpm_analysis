"""The analysis engine: pure computation, PCG recording in, beats / states / BPM out.

Everything outside this package depends on it, never the reverse
(tests/test_engine_boundary.py).
"""
from .config import DEFAULT_PARAMS, param
from .run import EngineResult, run_analysis

__all__ = ["DEFAULT_PARAMS", "EngineResult", "param", "run_analysis"]
