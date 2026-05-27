from __future__ import annotations

import streamlit as st

from src.core.config import settings

st.set_page_config(page_title=settings.project_name, page_icon=":bar_chart:", layout="wide")
st.title(settings.project_name)
st.subheader("Enterprise Financial RAG Platform")
st.write("Created with Ziad Mohamed Gamal")
st.info("Phase 1 completed. Continue to Phase 2 to build ingestion and parsing.")
