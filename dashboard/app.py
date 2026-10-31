"""Local Streamlit demonstration dashboard."""

from __future__ import annotations

import io
import json
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Streamlit executes this file as a script, so explicitly expose the project
# root for imports shared with the Flask service.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flask_app.app import current_model_name, get_model_vectorizer, predict_outputs
from project_utils import preprocess_comment
from src.data.youtube_comments import YouTubeCommentsError, fetch_youtube_comments


st.set_page_config(page_title="YouTube Sentiment Lab", page_icon="💬", layout="wide")
st.title("YouTube Comment Sentiment Lab")
st.caption("Local-only inference and experiment review")


@st.cache_resource
def load_artifacts():
    return get_model_vectorizer()


tab_youtube, tab_predict, tab_batch, tab_results = st.tabs(
    ["YouTube Analysis", "Predict", "Batch CSV", "Experiments"]
)


def predict_frame(frame: pd.DataFrame, text_column: str) -> tuple[pd.DataFrame, str]:
    """Run the cached local model over a comment DataFrame."""
    model, vectorizer = load_artifacts()
    comments = frame[text_column].fillna("").astype(str).tolist()
    predictions, confidence = predict_outputs(
        model, vectorizer, [preprocess_comment(item) for item in comments]
    )
    result = frame.copy()
    result["sentiment"] = predictions
    result["confidence"] = confidence
    return result, current_model_name(model)


with tab_youtube:
    st.subheader("Analyze a YouTube video")
    st.caption(
        "Comments are retrieved only after you submit a URL. Sentiment inference and visualization run locally."
    )
    video_url = st.text_input(
        "YouTube video URL",
        placeholder="https://www.youtube.com/watch?v=...",
        key="youtube_video_url",
    )
    max_comments = st.slider("Maximum top-level comments", 50, 1000, 500, step=50)
    if st.button("Analyze comment section", type="primary"):
        api_key = os.environ.get("YOUTUBE_API_KEY", "")
        if not video_url.strip():
            st.warning("Enter a YouTube video URL first.")
        elif not api_key.strip():
            st.error("Set YOUTUBE_API_KEY in your local .env file before retrieving comments.")
        else:
            try:
                with st.spinner("Retrieving comments and running local sentiment analysis..."):
                    comments = fetch_youtube_comments(
                        video_url, api_key, max_comments=max_comments
                    )
                    result, model_name = predict_frame(comments, "comment")

                counts = result["sentiment"].value_counts().reindex(
                    ["1", "0", "-1"], fill_value=0
                )
                total = int(counts.sum())
                percentages = (counts / total * 100).round(1)
                labels = {"1": "Positive", "0": "Neutral", "-1": "Negative"}
                dominant = labels[str(counts.idxmax())]
                st.success(f"Analyzed {total:,} comments with `{model_name}`.")
                metric_columns = st.columns(4)
                metric_columns[0].metric("Most common sentiment", dominant)
                metric_columns[1].metric("Positive", f"{int(counts['1']):,} ({percentages['1']:.1f}%)")
                metric_columns[2].metric("Neutral", f"{int(counts['0']):,} ({percentages['0']:.1f}%)")
                metric_columns[3].metric("Negative", f"{int(counts['-1']):,} ({percentages['-1']:.1f}%)")

                chart = pd.DataFrame(
                    {"sentiment": [labels[key] for key in ["1", "0", "-1"]], "comments": counts.values}
                ).set_index("sentiment")
                st.bar_chart(chart)

                try:
                    from wordcloud import WordCloud

                    words = " ".join(
                        preprocess_comment(comment)
                        for comment in result["comment"].astype(str)
                        if comment.strip()
                    )
                    if words.strip():
                        wordcloud = WordCloud(
                            width=1000,
                            height=450,
                            background_color="black",
                            collocations=False,
                        ).generate(words)
                        st.image(wordcloud.to_image(), caption="Most common words in the comment section")
                except ValueError:
                    st.info("There were not enough usable words to create a word cloud.")

                st.dataframe(result, use_container_width=True)
                st.download_button(
                    "Download YouTube predictions.csv",
                    result.to_csv(index=False).encode("utf-8"),
                    "youtube_predictions.csv",
                    "text/csv",
                )
            except (YouTubeCommentsError, ValueError) as exc:
                st.error(str(exc))
            except Exception:
                st.error("The video could not be analyzed. Check the API key, URL, and model setup.")

with tab_predict:
    comment = st.text_area("Comment", placeholder="This explanation was really helpful!")
    if st.button("Predict sentiment", type="primary"):
        if not comment.strip():
            st.warning("Enter a comment first.")
        else:
            try:
                model, vectorizer = load_artifacts()
                st.caption(f"Model: `{current_model_name(model)}`")
                prediction, confidence = predict_outputs(
                    model, vectorizer, [preprocess_comment(comment)]
                )
                st.metric("Sentiment", prediction[0])
                if confidence[0] is not None:
                    st.metric("Confidence", f"{confidence[0]:.1%}")
            except Exception as exc:
                st.error(f"Model unavailable: {exc}")

with tab_batch:
    uploaded = st.file_uploader("Upload a CSV with comment, text, or clean_comment", type="csv")
    if uploaded is not None:
        frame = pd.read_csv(uploaded)
        column = next((name for name in ("comment", "text", "clean_comment") if name in frame), None)
        if column is None:
            st.error("The CSV must contain comment, text, or clean_comment.")
        elif st.button("Run batch prediction"):
            result, _ = predict_frame(frame, column)
            st.dataframe(result, use_container_width=True)
            st.download_button(
                "Download predictions.csv",
                result.to_csv(index=False).encode("utf-8"),
                "predictions.csv",
                "text/csv",
            )
            st.bar_chart(result["sentiment"].value_counts())

with tab_results:
    comparison = PROJECT_ROOT / "reports" / "model_comparison.csv"
    if comparison.exists():
        comparison_frame = pd.read_csv(comparison)
        st.dataframe(comparison_frame, use_container_width=True)
        selected = st.selectbox("Confusion matrix", comparison_frame["model_name"].tolist())
        figure = PROJECT_ROOT / "reports" / "figures" / f"{selected}_confusion_matrix.png"
        if figure.exists():
            st.image(str(figure), caption=f"{selected} test confusion matrix")
        report = PROJECT_ROOT / "models" / f"{selected}_report.json"
        if report.exists():
            metadata = json.loads(report.read_text(encoding="utf-8"))
            st.json({
                "model_name": metadata.get("model_name"),
                "run_id": metadata.get("run_id"),
                "artifact_path": metadata.get("artifact_path"),
                "supports_probability": metadata.get("supports_probability"),
            })
    else:
        st.info("Run `python -m src.training.compare` to create measured model results.")
