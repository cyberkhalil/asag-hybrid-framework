"""Top-level ASAG pipeline that orchestrates correction, feature extraction,
fusion, and grade-band assignment.
"""

import logging
from typing import Any, Dict, Optional

import numpy as np
import torch

from .corrector import SpellingCorrector
from .embeddings import FeatureExtractor
from .fusion import ScoreFusionNet

logger = logging.getLogger(__name__)


class ASAGPipeline:
    """End-to-end automated short answer grading pipeline.

    Composes:
        - :class:`SpellingCorrector`
        - :class:`FeatureExtractor`
        - :class:`ScoreFusionNet`

    Typical usage::

        pipeline = ASAGPipeline()
        result = pipeline.score("Photosynthesis makes food.", "Photosynthesis converts light energy.")
        print(result["score"], result["grade_band"])
    """

    # Thresholds for the three-tier grade-band mapping.
    _THRESHOLDS: Dict[str, Dict[str, float]] = {
        "analytical": {"low": 0.60, "high": 0.78},   # الأسئلة التحليلية (أكثر مرونة)
        "fact-dense": {"low": 0.65, "high": 0.82}    # الأسئلة العلمية (أكثر صرامة)
    }

    def __init__(
        self,
        dense_model_name: str = "bert-base-uncased",
        question_type: str = "analytical",
        tfidf_vectorizer_path: Optional[str] = None,
        spelling_dict_path: str = "frequency_dictionary_en_82_765.txt",
        device: Optional[str] = None,
    ) -> None:
        """Initialises the pipeline and all sub-components.

        Args:
            dense_model_name: Hugging Face model identifier.
            question_type: ``"analytical"`` or ``"fact-dense"``.
                Determines alpha and grade thresholds.
            tfidf_vectorizer_path: Path to a pre-fitted TF-IDF
                vectoriser (joblib).  If ``None``, the lexical
                component will not be available and the score will
                rely solely on the dense representation.
            spelling_dict_path: Path to the SymSpell frequency
                dictionary.
            device: Torch device for the transformer model.  Auto-
                detects CUDA when ``None``.
        """
        self.dense_model_name = dense_model_name
        self.question_type = question_type.lower()
        if self.question_type not in self._THRESHOLDS:
            logger.warning(
                "Unknown question_type '%s' – falling back to 'analytical'.",
                self.question_type,
            )
            self.question_type = "analytical"

        self.spelling_dict_path = spelling_dict_path
        self.tfidf_vectorizer_path = tfidf_vectorizer_path
        self.device = device

        # Sub-components are created immediately to allow inspection;
        # heavy model loading happens inside FeatureExtractor.__init__.
        self.corrector = SpellingCorrector(dictionary_path=spelling_dict_path)
        self.feature_extractor = FeatureExtractor(
            dense_model_name=dense_model_name,
            tfidf_vectorizer_path=tfidf_vectorizer_path,
            device=device,
        )
        self.fusion = ScoreFusionNet(
            alpha=ScoreFusionNet.default_alpha_for_question_type(
                self.question_type
            ),
            learnable_alpha=False,
            question_type=self.question_type,
        )

        logger.info(
            "ASAGPipeline initialised – model=%s, question_type=%s, "
            "alpha=%.4f",
            dense_model_name,
            self.question_type,
            self.fusion.get_alpha(),
        )

    def set_question_type(self, question_type: str) -> None:
        """Switches the question type, updating alpha and thresholds.

        Args:
            question_type: ``"analytical"`` or ``"fact-dense"``.
        """
        self.question_type = question_type.lower()
        if self.question_type not in self._THRESHOLDS:
            logger.warning(
                "Unknown question_type '%s' – keeping previous.",
                self.question_type,
            )
            return
        alpha = ScoreFusionNet.default_alpha_for_question_type(
            self.question_type
        )
        self.fusion.set_alpha(alpha)
        self.fusion.question_type = self.question_type
        logger.info(
            "Switched to %s – alpha=%.4f, thresholds=%s",
            self.question_type,
            alpha,
            self.get_thresholds(),
        )

    def get_thresholds(self) -> Dict[str, float]:
        """Returns the current grade-band thresholds.

        Returns:
            Dictionary with keys ``"low"`` and ``"high"``.
        """
        return self._THRESHOLDS[self.question_type].copy()

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Computes cosine similarity between two vectors."""
        a = np.asarray(a).flatten()
        b = np.asarray(b).flatten()
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def _determine_grade_band(self, score: float) -> str:
        """Maps a continuous score to a categorical grade band.

        Args:
            score: Hybrid score in ``[0, 1]``.

        Returns:
            ``"incorrect"``, ``"partial"``, or ``"correct"``.
        """
        thresholds = self.get_thresholds()
        if score < thresholds["low"]:
            return "incorrect"
        if score < thresholds["high"]:
            return "partial"
        return "correct"

    def score(
        self, student_answer: str, model_answer: str
    ) -> Dict[str, Any]:
        """Scores a student answer against a reference answer.

        Args:
            student_answer: Raw student text.
            model_answer: Reference (model) text.

        Returns:
            Dictionary with keys:
                - ``score`` (float): Hybrid similarity [0, 1].
                - ``grade_band`` (str): ``"incorrect"``,
                  ``"partial"``, or ``"correct"``.
                - ``s_tfidf`` (float): TF-IDF cosine similarity.
                - ``s_dense`` (float): Dense cosine similarity.
                - ``alpha`` (float): Blending weight used.
                - ``student_corrected`` (str): Student answer after
                  spell correction.
                - ``thresholds`` (dict): Current grade boundaries.

        Raises:
            ValueError: If either input string is empty.
        """
        if not student_answer or not student_answer.strip():
            raise ValueError("student_answer must not be empty.")
        if not model_answer or not model_answer.strip():
            raise ValueError("model_answer must not be empty.")

        # 1. Spell correction
        student_corrected = self.corrector.correct_text(
            student_answer.strip()
        )
        logger.debug("Corrected student answer: %s", student_corrected)

        # 2. Feature extraction (numpy arrays)
        feat_student = self.feature_extractor.extract_features(
            student_corrected
        )
        feat_model = self.feature_extractor.extract_features(
            model_answer.strip()
        )

        # 3. Cosine similarities
        s_dense = self._cosine_similarity(
            feat_student["pooled_embedding"],
            feat_model["pooled_embedding"],
        )

        tfidf_vec_s = feat_student.get("tfidf_vector")
        tfidf_vec_m = feat_model.get("tfidf_vector")
        if tfidf_vec_s is not None and tfidf_vec_m is not None:
            s_tfidf = self._cosine_similarity(tfidf_vec_s, tfidf_vec_m)
        else:
            logger.warning(
                "TF-IDF vectoriser not fitted; setting s_tfidf = 0."
            )
            s_tfidf = 0.0

        # 4. Fusion (convert to tensors, keep graph)
        s_tfidf_t = torch.tensor(s_tfidf, dtype=torch.float32)
        s_dense_t = torch.tensor(s_dense, dtype=torch.float32)
        hybrid_score = self.fusion(s_tfidf_t, s_dense_t)
        score_val = hybrid_score.item()

        # 5. Grade band
        band = self._determine_grade_band(score_val)

        return {
            "score": score_val,
            "grade_band": band,
            "s_tfidf": s_tfidf,
            "s_dense": s_dense,
            "alpha": self.fusion.get_alpha(),
            "student_corrected": student_corrected,
            "thresholds": self.get_thresholds(),
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Minimal example – requires a downloaded SymSpell dictionary and
    # an internet connection for the transformer model.
    pipeline = ASAGPipeline(question_type="analytical")
    result = pipeline.score(
        "The proces of fotosynthesis is importnt.",
        "Photosynthesis converts light energy into chemical energy.",
    )
    for k, v in result.items():
        print(f"{k}: {v}")