"""
ASAG-Hybrid-Framework: A unified hybrid framework for automated short answer grading
with token-level explainability.
"""

__version__ = "1.0.0"
__author__ = "Mahmoud Waleed Khalil"
__email__ = "makhalil@ucas.edu.ps"

from .corrector import SpellingCorrector
from .embeddings import FeatureExtractor
from .fusion import ScoreFusionNet
from .pipeline import ASAGPipeline
from .explainer import IntegratedGradientsExplainer

__all__ = [
    "SpellingCorrector",
    "FeatureExtractor",
    "ScoreFusionNet",
    "ASAGPipeline",
    "IntegratedGradientsExplainer",
    "__version__",
]