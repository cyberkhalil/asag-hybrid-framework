"""
Contextual token-level spelling correction module for short answers.

Uses SymSpell for fast, non-intrusive typo correction while preserving
scientific and domain-specific vocabulary.
"""

import logging
import re
from typing import List, Optional

from symspellpy import SymSpell, Verbosity

logger = logging.getLogger(__name__)


class SpellingCorrector:
    """Performs token-level spelling correction using SymSpell.

    The standard English frequency dictionary must be obtained separately.
    Download ``frequency_dictionary_en_82_765.txt`` from:
        https://github.com/mammothb/symspellpy/blob/master/symspellpy/frequency_dictionary_en_82_765.txt

    Attributes:
        max_edit_distance: Maximum edit distance for candidate generation.
        prefix_length: Prefix length used by SymSpell for indexing.
        sym_spell: Underlying SymSpell instance.
        domain_terms: Set of protected terms that must not be corrected.
    """

    def __init__(
        self,
        dictionary_path: str = "frequency_dictionary_en_82_765.txt",
        max_edit_distance: int = 2,
        prefix_length: int = 7,
    ) -> None:
        """Initialises the corrector and loads the frequency dictionary.

        Args:
            dictionary_path: Path to the SymSpell frequency dictionary.
            max_edit_distance: Maximum edit distance for corrections.
            prefix_length: Prefix length for SymSpell index.

        Raises:
            FileNotFoundError: If the dictionary file cannot be found.
        """
        self.max_edit_distance = max_edit_distance
        self.prefix_length = prefix_length
        self.domain_terms: set = set()

        self.sym_spell = SymSpell(
            max_dictionary_edit_distance=max_edit_distance,
            prefix_length=prefix_length,
        )

        self._load_dictionary(dictionary_path)
        logger.info(
            "SpellingCorrector initialised with dictionary=%s",
            dictionary_path,
        )

    def _load_dictionary(self, dictionary_path: str) -> None:
        """Loads the SymSpell frequency dictionary.

        Args:
            dictionary_path: Path to the dictionary file.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        try:
            if not self.sym_spell.load_dictionary(
                dictionary_path, term_index=0, count_index=1
            ):
                raise FileNotFoundError(
                    f"Failed to load dictionary from {dictionary_path}. "
                    "Please download the file from "
                    "https://github.com/mammothb/symspellpy/blob/master/"
                    "symspellpy/frequency_dictionary_en_82_765.txt"
                )
        except FileNotFoundError:
            logger.exception("Dictionary file not found.")
            raise

    def add_domain_terms(self, terms: List[str]) -> None:
        """Adds domain-specific terms that must be preserved during correction.

        The terms are inserted into the SymSpell dictionary with a high
        frequency count to prevent them from being considered misspellings.

        Args:
            terms: List of terms (case-insensitive) to protect.
        """
        for term in terms:
            term_lower = term.lower().strip()
            if term_lower not in self.domain_terms:
                # Use a very high count so the term is never flagged.
                self.sym_spell.create_dictionary_entry(
                    term_lower, 100_000_000
                )
                self.domain_terms.add(term_lower)
                logger.debug("Protected domain term: %s", term)

    def correct_text(self, text: str) -> str:
        """Corrects spelling errors in the input text.

        Whitespace is normalised, and every token is processed by SymSpell
        while respecting the protected domain vocabulary.

        Args:
            text: Raw student answer.

        Returns:
            Corrected text with the original casing transferred when possible.
        """
        if not text or not text.strip():
            return ""

        # Normalise whitespace
        text = " ".join(text.split())

        # SymSpell lookup_compound handles tokenisation, correction,
        # and re-casing automatically.
        suggestions = self.sym_spell.lookup_compound(
            text,
            max_edit_distance=self.max_edit_distance,
            transfer_casing=True,
            ignore_non_words=True,
        )

        # The best (first) suggestion is the corrected compound.
        if suggestions:
            corrected = suggestions[0].term
        else:
            corrected = text  # No correction found; return original

        logger.debug("Original: %s\nCorrected: %s", text, corrected)
        return corrected


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # The dictionary file must be present in the working directory.
    # Download it from:
    # https://github.com/mammothb/symspellpy/blob/master/symspellpy/frequency_dictionary_en_82_765.txt
    corrector = SpellingCorrector(
        dictionary_path="frequency_dictionary_en_82_765.txt"
    )
    corrector.add_domain_terms(["SciBERT", "TF-IDF", "BERT"])

    sample = "The proces of fotosynthesis is importnt."
    result = corrector.correct_text(sample)
    print(f"Input:    {sample}")
    print(f"Corrected: {result}")