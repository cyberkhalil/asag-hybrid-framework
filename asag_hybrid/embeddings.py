"""
Feature extraction layer combining lexical (TF-IDF) and dense semantic
(BERT / SciBERT) representations.

Uses raw Hugging Face ``transformers`` to retain full access to hidden
states, which is essential for token-level explainability (Integrated
Gradients).
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Extracts lexical and dense features from short answer texts.

    The dense component uses a pre-trained transformer to obtain
    token-level embeddings and attention masks.  The lexical component
    uses a fitted ``TfidfVectorizer`` for sparse bag-of-words
    representations.

    Attributes:
        dense_model_name: Hugging Face model identifier.
        device: Torch device used for inference.
        tokenizer: AutoTokenizer instance.
        model: AutoModel instance.
        tfidf_vectorizer: Fitted TfidfVectorizer (may be None).
    """

    def __init__(
        self,
        dense_model_name: str = "bert-base-uncased",
        tfidf_vectorizer_path: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        """Initialises the feature extractor.

        Args:
            dense_model_name: Hugging Face model name for the dense encoder.
            tfidf_vectorizer_path: Path to a pre-fitted TfidfVectorizer
                (joblib format).  If not provided, the lexical component
                must be fitted before use.
            device: Torch device (e.g., ``"cuda:0"``, ``"cpu"``).
                If ``None``, CUDA is used when available.
        """
        self.dense_model_name = dense_model_name
        self.device = self._resolve_device(device)
        logger.info("Using device: %s", self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(dense_model_name)
        self.model = AutoModel.from_pretrained(
            dense_model_name,
            output_hidden_states=True,
            output_attentions=True,
        )
        self.model.to(self.device)
        self.model.eval()

        self.tfidf_vectorizer: Optional[TfidfVectorizer] = None
        if tfidf_vectorizer_path is not None:
            self.load_tfidf(tfidf_vectorizer_path)

        logger.info(
            "FeatureExtractor initialised with model=%s", dense_model_name
        )

    @staticmethod
    def _resolve_device(device: Optional[str]) -> torch.device:
        """Auto-detects the best available device.

        Args:
            device: Requested device string or ``None``.

        Returns:
            A ``torch.device`` object.
        """
        if device is not None:
            return torch.device(device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_device(self) -> str:
        """Returns the current device as a string."""
        return str(self.device)

    # ------------------------------------------------------------------
    # Lexical component (TF-IDF)
    # ------------------------------------------------------------------
    def fit_tfidf(
        self,
        texts: List[str],
        **kwargs: Any,
    ) -> None:
        """Fits the TF-IDF vectoriser on a corpus.

        Args:
            texts: List of training texts.
            **kwargs: Additional arguments passed to
                ``TfidfVectorizer``.
        """
        default_params = dict(
            sublinear_tf=True,
            max_df=0.5,
            stop_words="english",
            ngram_range=(1, 2),
        )
        default_params.update(kwargs)
        self.tfidf_vectorizer = TfidfVectorizer(**default_params)
        self.tfidf_vectorizer.fit(texts)
        logger.info(
            "TF-IDF vectoriser fitted on %d documents.", len(texts)
        )

    def save_tfidf(self, path: str) -> None:
        """Persists the fitted TF-IDF vectoriser to disk.

        Args:
            path: Output file path (joblib format).

        Raises:
            RuntimeError: If the vectoriser has not been fitted.
        """
        if self.tfidf_vectorizer is None:
            raise RuntimeError("TfidfVectorizer has not been fitted yet.")
        joblib.dump(self.tfidf_vectorizer, path)
        logger.info("TF-IDF vectoriser saved to %s", path)

    def load_tfidf(self, path: str) -> None:
        """Loads a pre-fitted TF-IDF vectoriser from disk.

        Args:
            path: Path to the joblib file.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        self.tfidf_vectorizer = joblib.load(path)
        logger.info("TF-IDF vectoriser loaded from %s", path)

    # ------------------------------------------------------------------
    # Dense embedding extraction
    # ------------------------------------------------------------------
    def extract_features(self, text: str) -> Dict[str, Any]:
        """Extracts dense and (optionally) lexical features for a single text.

        Args:
            text: Input string (student or model answer).

        Returns:
            Dictionary with the following keys:
                - ``"pooled_embedding"``: Mean-pooled sentence embedding
                  ``(768,)``.
                - ``"token_embeddings"``: Token-level embeddings
                  ``(seq_len, 768)``.
                - ``"attention_mask"``: Attention mask ``(seq_len,)``.
                - ``"last_hidden_state"``: Last hidden state of the model
                  ``(seq_len, 768)``.
                - ``"hidden_states"``: Tuple of hidden states from all
                  layers (each ``(seq_len, 768)``).
                - ``"tokens"``: List of tokens produced by the tokenizer.
                - ``"tfidf_vector"``: Dense TF-IDF vector ``(vocab_size,)``,
                  or ``None`` if the vectoriser has not been fitted.

        Raises:
            RuntimeError: If an out-of-memory error occurs on GPU.
        """
        # Tokenisation
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Forward pass
        try:
            with torch.no_grad():
                outputs = self.model(**inputs)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error("CUDA out of memory; falling back to CPU.")
                self.device = torch.device("cpu")
                self.model.to(self.device)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = self.model(**inputs)
            else:
                raise

        # Unpack outputs
        last_hidden_state = outputs.last_hidden_state  # (1, seq_len, 768)
        # All hidden states: tuple of (1, seq_len, 768) for each layer
        hidden_states = outputs.hidden_states
        attention_mask = inputs["attention_mask"]  # (1, seq_len)

        # Mean pooling (excluding padding)
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        )
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, dim=1)
        sum_mask = input_mask_expanded.sum(dim=1)
        pooled = (sum_embeddings / sum_mask).squeeze(0)  # (768,)

        # Convert to numpy
        token_embeddings = last_hidden_state.squeeze(0).cpu().numpy()
        attention_mask_np = attention_mask.squeeze(0).cpu().numpy()
        pooled_np = pooled.cpu().numpy()

        # Hidden states across all layers
        hs_cpu = tuple(
            layer.squeeze(0).cpu().numpy() for layer in hidden_states
        )

        # Token list
        tokens = self.tokenizer.convert_ids_to_tokens(
            inputs["input_ids"][0]
        )

        # Lexical features
        tfidf_vec = None
        if self.tfidf_vectorizer is not None:
            sparse_vec = self.tfidf_vectorizer.transform([text])
            tfidf_vec = sparse_vec.toarray().flatten()

        return {
            "pooled_embedding": pooled_np,
            "token_embeddings": token_embeddings,
            "attention_mask": attention_mask_np,
            "last_hidden_state": token_embeddings,
            "hidden_states": hs_cpu,
            "tokens": tokens,
            "tfidf_vector": tfidf_vec,
        }


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    extractor = FeatureExtractor(dense_model_name="bert-base-uncased")

    sample = "Photosynthesis converts light energy into chemical energy."
    features = extractor.extract_features(sample)

    print("Pooled embedding shape:", features["pooled_embedding"].shape)
    print("Tokens:", features["tokens"])
    print("Attention mask:", features["attention_mask"])
    print(
        "Hidden states layers:",
        [layer.shape for layer in features["hidden_states"]],
    )
    print("TF-IDF vector (before fitting):", features["tfidf_vector"])

    # Fit TF-IDF on dummy corpus
    corpus = [
        "Photosynthesis is a process used by plants.",
        "Light energy is converted to chemical energy.",
        "Glucose is produced during photosynthesis.",
    ]
    extractor.fit_tfidf(corpus)
    features_with_tfidf = extractor.extract_features(sample)
    print(
        "TF-IDF vector shape:",
        features_with_tfidf["tfidf_vector"].shape
        if features_with_tfidf["tfidf_vector"] is not None
        else None,
    )