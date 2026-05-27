from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import streamlit as st

from src.core.config import settings

st.set_page_config(page_title=settings.project_name, page_icon=":bar_chart:", layout="wide")
st.title(settings.project_name)
st.subheader("Enterprise Financial RAG Platform")
st.write("Created with Ziad Mohamed Gamal")

api_base_url = f"http://{settings.api_host}:{settings.api_port}".replace("0.0.0.0", "127.0.0.1")

if "messages" not in st.session_state:
    st.session_state.messages = []


def call_query_api(question: str) -> dict[str, Any]:
    with httpx.Client(timeout=120.0) as client:
        response = client.post(f"{api_base_url}/query", json={"question": question})
        response.raise_for_status()
        return response.json()


def call_evaluate_api(input_path: str, sample_size: int, output_path: str) -> dict[str, Any]:
    with httpx.Client(timeout=3600.0) as client:
        response = client.post(
            f"{api_base_url}/evaluate",
            json={"input_path": input_path, "sample_size": sample_size, "output_path": output_path},
        )
        response.raise_for_status()
        return response.json()


def load_report_preview(path: str) -> dict[str, Any] | None:
    report_path = Path(path)
    if not report_path.exists():
        return None
    try:
        with report_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return None


rag_tab, eval_tab = st.tabs(["RAG Assistant", "Evaluation Dashboard"])

with rag_tab:
    st.markdown("### Ask FinLedger.AI")
    st.caption("The assistant uses hybrid retrieval and citation-grounded financial reasoning.")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                with st.expander("Sources"):
                    st.json(message["sources"])

    user_question = st.chat_input("Ask a financial question about your indexed filings")
    if user_question:
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)
        with st.chat_message("assistant"):
            with st.spinner("Generating answer..."):
                try:
                    result = call_query_api(user_question)
                    st.markdown(result["answer"])
                    with st.expander("Sources"):
                        st.json(result["sources"])
                    st.session_state.messages.append(
                        {"role": "assistant", "content": result["answer"], "sources": result["sources"]}
                    )
                except Exception as exc:
                    st.error(f"Query failed: {exc}")

with eval_tab:
    st.markdown("### Ragas Evaluation")
    st.caption("Run benchmark evaluation on DocFinQA validation set and inspect quality metrics.")
    col1, col2 = st.columns(2)
    with col1:
        input_path = st.text_input("Validation JSON Path", value=settings.docfinqa_validation_path)
        sample_size = st.number_input("Sample Size", min_value=1, value=settings.evaluation_sample_size, step=1)
    with col2:
        output_path = st.text_input("Report Output Path", value=settings.evaluation_output_path)
        st.text_input("Retrieval Mode", value=settings.retrieval_mode, disabled=True)

    if st.button("Run Evaluation", type="primary"):
        with st.spinner("Running evaluation pipeline..."):
            try:
                summary = call_evaluate_api(input_path=input_path, sample_size=int(sample_size), output_path=output_path)
                st.success("Evaluation completed.")
                metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                metric_col1.metric("Samples", summary["total_samples"])
                metric_col2.metric("Faithfulness", f"{summary['faithfulness']:.4f}")
                metric_col3.metric("Answer Relevancy", f"{summary['answer_relevancy']:.4f}")
                metric_col4.metric("Context Recall", f"{summary['context_recall']:.4f}")
            except Exception as exc:
                st.error(f"Evaluation failed: {exc}")

    report_preview = load_report_preview(output_path)
    if report_preview:
        st.markdown("### Latest Report Preview")
        st.json(report_preview.get("summary", {}))
        rows = report_preview.get("rows", [])
        if rows:
            st.markdown("### Sample Rows")
            st.dataframe(
                [
                    {
                        "question": row.get("question", ""),
                        "answer": row.get("answer", "")[:300],
                        "ground_truth": row.get("ground_truth", "")[:300],
                    }
                    for row in rows[:10]
                ],
                use_container_width=True,
            )
