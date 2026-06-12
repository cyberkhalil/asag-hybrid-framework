"""
Streamlit web interface for the ASAG-Hybrid-Framework.

Run with:
    streamlit run app.py
"""

import logging
from typing import Dict, List

import streamlit as st

from asag_hybrid import ASAGPipeline, IntegratedGradientsExplainer

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ASAG Hybrid Framework",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Path to SymSpell dictionary (auto‑downloaded by the spelling corrector)
SPELLING_DICT = "frequency_dictionary_en_82_765.txt"

# ---------------------------------------------------------------------------
# Cached resource loading
# ---------------------------------------------------------------------------
@st.cache_resource
@st.cache_resource
def load_pipeline(model_name: str, spelling_dict_path: str) -> ASAGPipeline:
    """Initialises and caches the heavy ASAG pipeline."""
    logger.info("Loading ASAGPipeline with model=%s", model_name)
    return ASAGPipeline(
        dense_model_name=model_name,
        question_type="analytical",
        tfidf_vectorizer_path="models/tfidf_vectorizer.joblib",  # <-- تم الإصلاح هنا
        spelling_dict_path=spelling_dict_path,
    )

# ---------------------------------------------------------------------------
# Helper: render coloured text for XAI
# ---------------------------------------------------------------------------
def render_colored_text(aggregated_words: List[Dict]) -> str:
    """Render HTML with coloured spans based on attribution values.

    Args:
        aggregated_words: List of dicts with keys ``"word_id"``,
            ``"word"``, and ``"attribution"``.

    Returns:
        HTML string where each word is wrapped in a ``<span>`` with
        background colour proportional to its attribution.
    """
    if not aggregated_words:
        return ""

    max_attr = max(abs(w["attribution"]) for w in aggregated_words) or 1.0

    html_parts = []
    for w in aggregated_words:
        attr = w["attribution"]
        normalized = attr / max_attr

        if normalized > 0:
            colour = f"rgba(0, 200, 0, {normalized:.2f})"
        else:
            colour = f"rgba(200, 0, 0, {abs(normalized):.2f})"

        safe_word = (
            w["word"]
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )
        html_parts.append(
            f'<span style="background-color: {colour}; color: white; '
            f'padding: 2px 4px; margin: 2px; border-radius: 3px; '
            f'font-weight: bold;">{safe_word}</span>'
        )

    return " ".join(html_parts)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.header("Configuration")

model_name = st.sidebar.selectbox(
    "Transformer Model",
    options=["bert-base-uncased", "allenai/scibert_scivocab_uncased"],
    index=0,
    help="Dense encoder used for semantic similarity.",
)

question_type = st.sidebar.radio(
    "Question Type",
    options=["analytical", "fact-dense"],
    index=0,
    help="Controls the TF‑IDF / dense blending weight (alpha) and grade thresholds.",
)

show_xai = st.sidebar.checkbox(
    "Show Token-Level XAI Explanation", value=True
)

n_steps = st.sidebar.slider(
    "Integrated Gradients Steps",
    min_value=10,
    max_value=100,
    value=50,
    step=10,
    help="Number of interpolation steps for Integrated Gradients.",
)

# ---------------------------------------------------------------------------
# Load pipeline (cached)
# ---------------------------------------------------------------------------
pipeline = load_pipeline(model_name, SPELLING_DICT)
pipeline.set_question_type(question_type)

# ---------------------------------------------------------------------------
# Main interface
# ---------------------------------------------------------------------------
st.title("📝 ASAG Hybrid Framework")
st.markdown(
    "Automated Short Answer Grading with real‑time, privacy‑preserving, "
    "and explainable scoring."
)

col1, col2 = st.columns(2)
with col1:
    model_answer = st.text_area(
        "Model Answer (Reference)",
        placeholder="Enter the reference answer...",
        height=200,
    )
with col2:
    student_answer = st.text_area(
        "Student Answer",
        placeholder="Enter the student's answer...",
        height=200,
    )

button_disabled = not (model_answer.strip() and student_answer.strip())
grade_button = st.button(
    "🎯 Grade Answer",
    type="primary",
    use_container_width=True,
    disabled=button_disabled,
)

# ---------------------------------------------------------------------------
# Results section (triggered by the button)
# ---------------------------------------------------------------------------
if grade_button:
    with st.spinner("Processing and grading answer..."):
        try:
            result = pipeline.score(student_answer, model_answer)

            xai_result = None
            if show_xai:
                explainer = IntegratedGradientsExplainer(
                    feature_extractor=pipeline.feature_extractor,
                    fusion_net=pipeline.fusion,
                    m_steps=n_steps,
                )
                xai_result = explainer.explain(
                    student_answer, model_answer
                )

            st.subheader("📊 Scoring Results")

            col_metrics, col_spell = st.columns(2)

            with col_metrics:
                score = result["score"]
                st.metric(
                    label="Hybrid Score",
                    value=f"{score:.4f}",
                    delta=None,
                )

                band = result["grade_band"]
                if band == "correct":
                    colour = "green"
                    badge = "✅ Correct"
                elif band == "partial":
                    colour = "orange"
                    badge = "⚠️ Partial"
                else:
                    colour = "red"
                    badge = "❌ Incorrect"
                st.markdown(
                    f"<h3 style='color:{colour};'>{badge}</h3>",
                    unsafe_allow_html=True,
                )

                st.metric(
                    label="TF‑IDF Similarity",
                    value=f"{result['s_tfidf']:.4f}",
                )
                st.metric(
                    label="Dense Similarity",
                    value=f"{result['s_dense']:.4f}",
                )
                st.metric(label="Alpha (Blending)", value=f"{result['alpha']:.2f}")

            with col_spell:
                corrected = result["student_corrected"]
                if corrected != student_answer.strip():
                    st.markdown("**Spell Correction Applied:**")
                    st.markdown(
                        f"<span style='color:gray;'>Original:</span> "
                        f"_{student_answer}_",
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f"<span style='color:green;'>Corrected:</span> "
                        f"**{corrected}**",
                        unsafe_allow_html=True,
                    )
                else:
                    st.info("No spelling corrections were necessary.")

            # ------------------------------------------------------------------
            # XAI visualisation
            # ------------------------------------------------------------------
            if show_xai and xai_result is not None:
                st.markdown("---")
                st.subheader("🔍 Token-Level Explainability (Integrated Gradients)")

                # aggregated_words is now a List[Dict] directly
                agg_words = xai_result["aggregated_words"]

                # Coloured text rendering
                st.markdown(
                    render_colored_text(agg_words),
                    unsafe_allow_html=True,
                )

                col_pos, col_neg = st.columns(2)
                with col_pos:
                    st.markdown("**Top 5 Positive Tokens**")
                    if xai_result["top_positive"]:
                        pos_df = {
                            "Token": [t for t, _ in xai_result["top_positive"]],
                            "Attribution": [f"{a:.4f}" for _, a in xai_result["top_positive"]],
                        }
                        st.table(pos_df)
                    else:
                        st.write("None")

                with col_neg:
                    st.markdown("**Top 5 Negative Tokens**")
                    if xai_result["top_negative"]:
                        neg_df = {
                            "Token": [t for t, _ in xai_result["top_negative"]],
                            "Attribution": [f"{a:.4f}" for _, a in xai_result["top_negative"]],
                        }
                        st.table(neg_df)
                    else:
                        st.write("None")

                with st.expander("📋 All Word Attributions"):
                    st.dataframe(
                        {
                            "Word": [w["word"] for w in agg_words],
                            "Attribution": [w["attribution"] for w in agg_words],
                        }
                    )

        except Exception as exc:
            logger.exception("Grading failed.")
            st.error(f"An error occurred: {exc}")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray;'>"
    "<em>Developed by Mahmoud Waleed Khalil | Supervised by Dr. Aiman Ahmed Abusamra</em><br>"
    "Deanship of Engineering, Islamic University of Gaza"
    "</p>",
    unsafe_allow_html=True,
)