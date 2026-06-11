"""Differentiable score fusion module for ASAG-Hybrid.

Combines TF-IDF and dense semantic similarities via a weighted sum.
The blending weight can be fixed or learnable, preserving the full
PyTorch computational graph for gradient-based explainability.
"""

import logging
from typing import Optional, Union

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ScoreFusionNet(nn.Module):
    """Learns (or applies) a weighted fusion of lexical and semantic scores.

    The forward method implements:

        S_hybrid = alpha * S_TF-IDF + (1 - alpha) * S_dense

    All operations remain inside the autograd graph so that Integrated
    Gradients can flow gradients back to the input embeddings.

    Attributes:
        alpha: Blending weight (0 ≤ alpha ≤ 1).
            Higher values favour lexical similarity.
        learnable_alpha: If True, ``alpha`` is a ``nn.Parameter``.
        question_type: Descriptive tag for the kind of question
            ("fact-dense" or "analytical").
    """

    def __init__(
        self,
        alpha: float = 0.25,
        learnable_alpha: bool = False,
        question_type: str = "analytical",
    ) -> None:
        """Initialises the fusion module.

        Args:
            alpha: Initial blending weight.
            learnable_alpha: If True, ``alpha`` becomes a learnable
                parameter that can be optimised with gradient descent.
            question_type: Question category (determines default
                ``alpha`` when not learnable).
        """
        super().__init__()
        self.question_type = question_type

        if learnable_alpha:
            self.alpha = nn.Parameter(torch.tensor(alpha))
            self.learnable_alpha = True
            logger.info(
                "Learnable alpha initialised to %.4f", alpha
            )
        else:
            self.register_buffer("alpha", torch.tensor(alpha))
            self.learnable_alpha = False
            logger.info("Fixed alpha set to %.4f", alpha)

    def forward(
        self, s_tfidf: torch.Tensor, s_dense: torch.Tensor
    ) -> torch.Tensor:
        """Computes the hybrid score.

        Args:
            s_tfidf: TF-IDF cosine similarity (0-dim or 1-dim tensor).
            s_dense: Dense (transformer) cosine similarity (0-dim or
                1-dim tensor).

        Returns:
            Scalar tensor containing the fused score in ``[0, 1]``.
        """
        alpha_val = self.alpha  # nn.Parameter or buffer
        s_hybrid = alpha_val * s_tfidf + (1.0 - alpha_val) * s_dense
        return s_hybrid

    def get_alpha(self) -> float:
        """Returns the current blending weight as a plain float."""
        if self.learnable_alpha:
            return self.alpha.detach().cpu().item()
        return self.alpha.item()

    def set_alpha(self, alpha: float) -> None:
        """Updates the blending weight.

        Args:
            alpha: New value in ``[0, 1]``.

        Note:
            If ``learnable_alpha`` is True, the parameter data is
            updated in-place.
        """
        if self.learnable_alpha:
            self.alpha.data.fill_(alpha)
        else:
            self.alpha.fill_(alpha)
        logger.debug("Alpha changed to %.4f", alpha)

    @staticmethod
    def default_alpha_for_question_type(question_type: str) -> float:
        """Returns the recommended alpha for a given question type.

        Args:
            question_type: Either ``"fact-dense"`` or ``"analytical"``.

        Returns:
            Default alpha value.

        Raises:
            ValueError: If ``question_type`` is unknown.
        """
        mapping = {
            "fact-dense": 0.30,
            "analytical": 0.20,
        }
        try:
            return mapping[question_type.lower()]
        except KeyError:
            raise ValueError(
                f"Unknown question_type '{question_type}'. "
                f"Choose from {list(mapping.keys())}."
            )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Fixed alpha
    net = ScoreFusionNet(alpha=0.25, learnable_alpha=False)
    s_tfidf = torch.tensor(0.72)
    s_dense = torch.tensor(0.86)
    score = net(s_tfidf, s_dense)
    print(f"Fixed fusion score: {score.item():.4f}")

    # Learnable alpha
    net_learn = ScoreFusionNet(alpha=0.25, learnable_alpha=True)
    score_learn = net_learn(s_tfidf, s_dense)
    print(f"Learnable fusion score: {score_learn.item():.4f}, "
          f"alpha={net_learn.get_alpha():.4f}")

    # Default per question type
    print("Analytical default:", ScoreFusionNet.default_alpha_for_question_type("analytical"))
    print("Fact-dense default:", ScoreFusionNet.default_alpha_for_question_type("fact-dense"))