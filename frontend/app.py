from __future__ import annotations
import logging
import sys
from pathlib import Path
import pandas as pd
import streamlit as st
from google.api_core.exceptions import ResourceExhausted

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "backend"))
sys.path.insert(0, str(_ROOT / "utils"))

from schema_loader import load_schema
from sql_generator import generate_sql
from query_executor import execute_query
from chart_selector import generate_dashboard_charts
from chart_renderer import render_chart
from insights_generator import generate_insights
from css_loader import load_css
from followup_resolver import resolve_followup

@st.cache_data(show_spinner=False)
def cached_pipeline(df, schema, query):
    sql_query = generate_sql(query, schema)
    result_df = execute_query(df, sql_query)
    chart_types = generate_dashboard_charts(result_df, query)
    insights = generate_insights(result_df)

    return sql_query, result_df, chart_types, insights


logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s — %(message)s")
logger = logging.getLogger(__name__)
            
if "last_query" not in st.session_state:
    st.session_state.last_query = None

st.set_page_config(
    page_title="Prompt2Dashboard",
    page_icon="🥝",
    layout="wide",
    initial_sidebar_state="collapsed",
)

load_css(_ROOT / "frontend" / "style.css")

st.markdown("""
<div class="p2d-header">
    <p class="p2d-title">Prompt<span>2</span>Dashboard</p>
    <p class="p2d-subtitle">🥝 Conversational AI</p>
</div>
""", unsafe_allow_html=True)


sidebar, main = st.columns([1, 2.4], gap="large")


with sidebar:
    st.markdown('<div class="section-label">01 · Upload Dataset</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        label="CSV file",
        type=["csv"],
        label_visibility="collapsed",
    )

    if uploaded_file:
        try:
            _preview_df, _schema_str = load_schema(uploaded_file)
            uploaded_file.seek(0)

            row_count, col_count = _preview_df.shape
            st.markdown(
                f'<div class="chip-row">'
                f'<span class="chip"><strong>{row_count}</strong> rows</span>'
                f'<span class="chip"><strong>{col_count}</strong> columns</span>'
                f'<span class="chip"><strong>{uploaded_file.name}</strong></span>'
                f'</div>',
                unsafe_allow_html=True,
            )
            st.markdown('<div class="section-label">Schema</div>', unsafe_allow_html=True)
            st.code(_schema_str, language="text")
            
            st.success("Dataset loaded successfully. You can now ask questions.")
        except Exception as _e:
            st.error(f"Could not read file: {_e}")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">02 · Ask a Question</div>', unsafe_allow_html=True)
    if uploaded_file is None:
        st.caption("Upload a CSV to start asking questions.")
    user_query = st.text_input(
        label="Natural language query",
        value=st.session_state.get("example_query", ""),
        placeholder="e.g. Show total revenue by region",
        label_visibility="collapsed",
        disabled=uploaded_file is None
    )

    st.markdown("<br>", unsafe_allow_html=True)
    generate_btn = st.button("🥝  Generate Dashboard", use_container_width=True, disabled=(uploaded_file is None or not user_query.strip()))

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Example queries</div>', unsafe_allow_html=True)

    _examples = [
        "Show total views by category",
        "Show monthly views trend",
        "Which regions have the highest engagement",
        "Compare sentiment score by category",
    ]
    
    cols = st.columns(2)

    for i, ex in enumerate(_examples):
        if cols[i % 2].button(ex, disabled=uploaded_file is None):
            st.session_state["example_query"] = ex
            st.rerun()

with main:
    if not generate_btn:
        st.markdown("""
        <div style="
            display:flex; flex-direction:column; align-items:center;
            justify-content:center; height:420px; gap:1rem;
            border: 1px dashed #1e2330; border-radius:12px;
            color:#2a2f40; text-align:center; padding:2rem;
        ">
            <div style="font-size:3rem;">🥝</div>
            <div style="font-family:'Syne',sans-serif; font-size:1.1rem; color:#3a3f52;">
                Upload a CSV and ask a question
            </div>
            <div style="font-size:0.78rem; color:#2a2f40; letter-spacing:0.05em;">
                AI will generate SQL · visualise results · surface insights
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        if not uploaded_file:
            st.error("Please upload a CSV file first.")
            st.stop()
        if not user_query or not user_query.strip():
            st.error("Please enter a question before generating.")
            st.stop()

        try:
            with st.spinner("Reading dataset schema…"):
                uploaded_file.seek(0)
                df, schema = load_schema(uploaded_file)
                
            query = user_query

            if st.session_state.last_query:
                query = resolve_followup(st.session_state.last_query, user_query)

            st.session_state.last_query = query

            sql_query, result_df, chart_types, insights = cached_pipeline(df, schema, query)

        except ResourceExhausted as e:
            st.warning(
                "⚠️ AI service temporarily unavailable (quota exceeded).\n"
                "Please retry in a few seconds."
                )
            st.stop()
        except PermissionError as e:
            st.error(f"🔒 Access denied: {e}")
            st.stop()
        except ValueError as e:
            st.error(f"⚠️ Data error: {e}")
            st.stop()
        except RuntimeError as e:
            st.error(f"⚙️ Processing error: {e}")
            st.stop()
        except Exception as e:
            st.error(f"Unexpected error — {type(e).__name__}: {e}")
            logger.exception("Unhandled pipeline error")
            st.stop()


        st.subheader("Generated Dashboard")
        st.markdown('<div class="section-label">Visualisation</div>', unsafe_allow_html=True)
        for chart_type in chart_types:
            fig = render_chart(result_df, chart_type)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        
        if not result_df.empty:
            metric_cols = st.columns(3)

            with metric_cols[0]:
                st.metric("Rows Returned", len(result_df))

            with metric_cols[1]:
                numeric_cols = result_df.select_dtypes(include="number").columns
                if len(numeric_cols) > 0:
                    st.metric("Total Value", round(result_df[numeric_cols[0]].sum(), 2))
                else:
                    st.metric("Columns", len(result_df.columns))

            with metric_cols[2]:
                st.metric("Columns", len(result_df.columns))
        st.markdown("<hr>", unsafe_allow_html=True)


        ins_col, sql_col = st.columns([1.1, 1], gap="large")

        with ins_col:
            st.markdown('<div class="section-label">Business Insights</div>', unsafe_allow_html=True)
            bullets_html = "".join(
                f'<div class="insight-item">'
                f'<span class="insight-dot">•</span>'
                f'<span>{b.lstrip("•").strip()}</span>'
                f'</div>'
                for b in insights
            )
            st.markdown(
                f'<div class="insight-card">{bullets_html}</div>',
                unsafe_allow_html=True,
            )

        with sql_col:
            st.markdown('<div class="section-label">Generated SQL</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="sql-block">{sql_query}</div>', unsafe_allow_html=True)

        st.markdown("<hr>", unsafe_allow_html=True)


        st.markdown(
            f'<div class="section-label">Result · {len(result_df)} rows · {len(result_df.columns)} columns</div>',
            unsafe_allow_html=True,
        )
        st.dataframe(result_df, use_container_width=True, hide_index=True)


        csv_bytes = result_df.to_csv(index=False).encode()
        st.download_button(
            label="↓  Download result as CSV",
            data=csv_bytes,
            file_name="query_result.csv",
            mime="text/csv",
        )