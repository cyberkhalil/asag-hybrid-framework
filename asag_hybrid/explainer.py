"""Token-level Integrated Gradients explainer for the ASAG hybrid score.

Computes feature attributions with respect to the dense component of
the hybrid score so that every token's contribution can be visualised.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.autograd import grad as autograd_grad

from .embeddings import FeatureExtractor
from .fusion import ScoreFusionNet

logger = logging.getLogger(__name__)


class IntegratedGradientsExplainer:
    """Generates token-level attributions using Integrated Gradients.

    The explainer uses the transformer inside ``FeatureExtractor``
    directly to keep the computational graph intact.  Because the
    TF-IDF component is non-differentiable with respect to the input
    tokens, the explanation is generated for the *dense* part of the
    hybrid score (by supplying ``s_tfidf = 0.0`` to the fusion
    network).

    Usage::

        extractor = FeatureExtractor()
        fusion = ScoreFusionNet(alpha=0.2)
        explainer = IntegratedGradientsExplainer(extractor, fusion)
        exp = explainer.explain("Photosynthesis makes food.", "Photosynthesis converts light.")
    """

    def __init__(
        self,
        feature_extractor: FeatureExtractor,
        fusion_net: ScoreFusionNet,
        m_steps: int = 50,
        batch_size: int = 8,
    ) -> None:
        """Initialises the explainer.

        Args:
            feature_extractor: A ``FeatureExtractor`` that contains a
                loaded transformer model and tokenizer.
            fusion_net: The fusion module used to produce the hybrid
                score.
            m_steps: Number of interpolation steps for Integrated
                Gradients.
            batch_size: (Reserved for future batched processing;
                currently a single sentence is explained.)
        """
        self.extractor = feature_extractor
        self.fusion_net = fusion_net
        self.m_steps = m_steps
        self.batch_size = batch_size

        # Ensure the model stays in eval mode but allows gradients.
        self.extractor.model.eval()
        for p in self.extractor.model.parameters():
            p.requires_grad = False

    def _pool_embeddings(
        self,
        last_hidden: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Mean-pool the last hidden state over non-padded tokens.

        Args:
            last_hidden: ``(batch, seq_len, hidden_dim)``.
            attention_mask: ``(batch, seq_len)``.

        Returns:
            Pooled embedding ``(batch, hidden_dim)``.
        """
        mask_exp = attention_mask.unsqueeze(-1).float()
        masked = last_hidden * mask_exp
        summed = masked.sum(dim=1)
        counts = mask_exp.sum(dim=1)
        return summed / counts

    def _compute_dense_score_from_embeddings(
        self,
        student_embeds: torch.Tensor,
        student_mask: torch.Tensor,
        model_pooled: torch.Tensor,
    ) -> torch.Tensor:
        """Computes the hybrid score using the dense signal only.

        Args:
            student_embeds: Student input embeddings (batch, seq, dim).
            student_mask: Attention mask for student.
            model_pooled: Pre-computed pooled embedding of the model answer.

        Returns:
            Scalar tensor – the hybrid score.
        """
        encoder_outputs = self.extractor.model(
            inputs_embeds=student_embeds,
            attention_mask=student_mask,
            output_hidden_states=True,
            output_attentions=False,
        )
        last_hidden = encoder_outputs.last_hidden_state
        student_pooled = self._pool_embeddings(last_hidden, student_mask)

        s_dense = torch.cosine_similarity(
            student_pooled, model_pooled.unsqueeze(0)
        ).squeeze(0)

        s_tfidf = torch.tensor(0.0, device=s_dense.device)
        score = self.fusion_net(s_tfidf, s_dense)
        return score

    def explain(
        self, student_answer: str, model_answer: str
    ) -> Dict[str, Any]:
        """Generates token-level attributions.

        Args:
            student_answer: Raw student text.
            model_answer: Reference (model) text.

        Returns:
            Dictionary with:
                - ``tokens``: List of token strings.
                - ``attributions``: Normalized attribution per token
                  (sum of absolute values = 1).
                - ``top_positive``: Top 5 tokens with positive
                  influence.
                - ``top_negative``: Top 5 tokens with negative
                  influence.
                - ``aggregated_words``: List of dicts, each with keys
                  ``"word_id"`` (int), ``"word"`` (str), and
                  ``"attribution"`` (float).  Preserves sequential
                  order and handles duplicate words correctly.
        """
        device = self.extractor.device
        tokenizer = self.extractor.tokenizer

        # ----- Pre-compute model answer pooled embedding -----
        enc_model = tokenizer(
            model_answer,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        enc_model = {k: v.to(device) for k, v in enc_model.items()}
        with torch.no_grad():
            model_outputs = self.extractor.model(**enc_model)
        model_pooled = self._pool_embeddings(
            model_outputs.last_hidden_state,
            enc_model["attention_mask"],
        ).squeeze(0)

        # ----- Tokenize student answer -----
        enc_student = tokenizer(
            student_answer,
            return_tensors="pt",
            padding=True,
            truncation=True,
            return_offsets_mapping=False,
        )
        input_ids = enc_student["input_ids"].to(device)
        attention_mask = enc_student["attention_mask"].to(device)

        # Get word-to-token mapping for aggregation
        mapping_enc = tokenizer(
            student_answer,
            return_offsets_mapping=True,
            truncation=True,
            add_special_tokens=True,
        )
        word_ids = mapping_enc.word_ids()

        # ----- Obtain input embeddings -----
        embed_layer = self.extractor.model.get_input_embeddings()
        input_embeds = embed_layer(input_ids)
        input_embeds.requires_grad_(True)

        # Baseline: zero tensor of same shape
        baseline_embeds = torch.zeros_like(input_embeds)

        # Integrated Gradients loop
        grads = torch.zeros_like(input_embeds)
        for step in range(self.m_steps):
            alpha = step / self.m_steps
            interpolated = baseline_embeds + alpha * (
                input_embeds - baseline_embeds
            )
            interpolated.retain_grad()

            score = self._compute_dense_score_from_embeddings(
                interpolated, attention_mask, model_pooled
            )

            grad_interp = autograd_grad(
                score,
                interpolated,
                retain_graph=(step < self.m_steps - 1),
            )[0]
            grads += grad_interp

        # Average gradients
        avg_grads = grads / self.m_steps

        # Integrated Gradients formula
        ig = (input_embeds - baseline_embeds) * avg_grads
        token_attrs = ig.sum(dim=-1).squeeze(0)

        # ---- Mask special tokens ----
        special_mask = torch.tensor(
            [
                id_.item()
                not in {tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id}
                for id_ in input_ids[0]
            ],
            device=device,
            dtype=torch.bool,
        )
        token_attrs[~special_mask] = 0.0

        # Convert to numpy for further processing
        token_attrs_np = token_attrs.detach().cpu().numpy()
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

        # Normalize by absolute sum
        total_abs = np.abs(token_attrs_np).sum()
        if total_abs > 0:
            norm_attrs = token_attrs_np / total_abs
        else:
            norm_attrs = token_attrs_np

        # ---- Top positive / negative ----
        idx_sorted = np.argsort(norm_attrs)
        top_neg_idx = idx_sorted[:5]
        top_pos_idx = idx_sorted[-5:][::-1]

        top_positive = [
            (tokens[i], float(norm_attrs[i])) for i in top_pos_idx
        ]
        top_negative = [
            (tokens[i], float(norm_attrs[i])) for i in top_neg_idx
        ]

        # ---- Word-level aggregation (FIXED) ----
        aggregated_words_list: List[Dict[str, Any]] = []
        current_word_id = None
        current_word = None
        current_attr = 0.0

        for tok_idx, (attr, w_id) in enumerate(zip(norm_attrs, word_ids)):
            if w_id is None:
                continue

            # If we're starting a new word, save the previous one
            if w_id != current_word_id:
                if current_word is not None:
                    aggregated_words_list.append({
                        "word_id": int(current_word_id),
                        "word": current_word,
                        "attribution": float(current_attr),
                    })
                current_word_id = w_id
                current_attr = 0.0

            # Accumulate attribution for current word
            current_attr += attr

            # Decode the word (use the first token of the word for display)
            if current_attr == attr:  # First token of this word
                current_word = tokenizer.decode(
                    input_ids[0, tok_idx],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )

        # Don't forget the last word
        if current_word is not None:
            aggregated_words_list.append({
                "word_id": int(current_word_id),
                "word": current_word,
                "attribution": float(current_attr),
            })

        return {
            "tokens": tokens,
            "attributions": norm_attrs.tolist(),
            "top_positive": top_positive,
            "top_negative": top_negative,
            "aggregated_words": aggregated_words_list,
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    extractor = FeatureExtractor(dense_model_name="bert-base-uncased")
    fusion = ScoreFusionNet(alpha=0.2, learnable_alpha=False)
    explainer = IntegratedGradientsExplainer(extractor, fusion, m_steps=20)

    result = explainer.explain(
        "Photosynthesis makes food from sunlight.",
        "Photosynthesis converts light energy to chemical energy.",
    )

    print("Tokens:", result["tokens"])
    print("Attributions:", result["attributions"])
    print("Top positive:", result["top_positive"])
    print("Top negative:", result["top_negative"])
    print("Word attributions:", result["aggregated_words"])