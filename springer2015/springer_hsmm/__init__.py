"""
Springer et al. Logistic Regression-HSMM-based Heart Sound Segmentation.

Self-contained Python translation of the MATLAB code from:
D. Springer et al., "Logistic Regression-HSMM-based Heart Sound Segmentation,"
IEEE Trans. Biomed. Eng., 2016.

Original code: PhysioNet / logistic-regression-hsmm-based-heart-sound-segmentation
"""

from springer_hsmm.options import default_springer_hsmm_options
from springer_hsmm.run import run_springer_segmentation_algorithm

__all__ = [
    "default_springer_hsmm_options",
    "run_springer_segmentation_algorithm",
]
