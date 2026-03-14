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

for key, default in {
    "last_query": None,
    "messages": [],
    "df": None,
    "schema": None,
    "file_name": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

st.set_page_config(
    page_title="Prompt2Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

load_css(_ROOT / "frontend" / "style.css")

_msgs         = st.session_state.get("messages", [])
_last_is_user = len(_msgs) > 0 and _msgs[-1]["role"] == "user"
_has_pending  = bool(st.session_state.get("pending_query"))
HIDE_EXAMPLES = _last_is_user or _has_pending


def push_message(role: str, content: str, data: dict | None = None):
    st.session_state.messages.append(
        {"role": role, "content": content, "data": data or {}}
    )

if st.session_state.df is None:
    st.markdown("""
    <div class="landing-wrap">
        <div class="landing-logo">📊</div>
        <h1 class="landing-title">Prompt<span>2</span>Dashboard</h1>
        <p class="landing-sub">Drop a CSV and start asking questions</p>
    </div>
    """, unsafe_allow_html=True)

    col_l, col_c, col_r = st.columns([1, 2, 1])
    with col_c:
        uploaded_file = st.file_uploader(
            label="Drop CSV here",
            type=["csv"],
            label_visibility="collapsed",
            key="landing_uploader",
        )

        if uploaded_file:
            try:
                with st.spinner("Reading schema…"):
                    df, schema = load_schema(uploaded_file)
                st.session_state.df = df
                st.session_state.schema = schema
                st.session_state.file_name = uploaded_file.name
                push_message(
                    "assistant",
                    f"**{uploaded_file.name}** loaded — {df.shape[0]} rows × {df.shape[1]} columns.\n\n"
                    "I've analysed the schema below. Ask me anything about your data!",
                    {"schema": schema, "shape": df.shape},
                )
                st.rerun()
            except Exception as e:
                st.error(f"Could not read file: {e}")
    st.stop()

if st.button("＋", key="new_csv_btn", help="Upload new data"):
    for k in ["df", "schema", "file_name", "messages", "last_query"]:
        st.session_state[k] = None if k != "messages" else []
    st.rerun()

with st.container():
    for msg in st.session_state.messages:
        role = msg["role"]
        data = msg.get("data", {})

        if role == "assistant":
            st.markdown(f"""
            <div class="chat-row assistant-row">
                <div class="avatar-col">
                    <div class="avatar ai-avatar">📊</div>
                </div>
                <div class="bubble assistant-bubble">
            """, unsafe_allow_html=True)
            st.markdown(msg["content"])

            if "schema" in data:
                st.code(data["schema"], language="text")

            if "chart_types" in data and "result_df" in data:
                result_df = data["result_df"]
                chart_types = data["chart_types"]
                for ct in chart_types:
                    try:
                        chart = render_chart(result_df, ct)
                        if isinstance(chart, dict) and chart.get("type") == "metric":
                            label = chart["label"].replace("_", " ").title()
                            if chart["category"]:
                                st.metric(label=f"Top {label}", value=f"{chart['category']} ({chart['value']:,.0f})")
                            else:
                                st.metric(label=label, value=f"{chart['value']:,.0f}")
                        else:
                            st.plotly_chart(chart, use_container_width=True)
                    except Exception:
                        pass

                if not result_df.empty and "pie" not in chart_types:
                    mc = st.columns(3)
                    mc[0].metric("Rows Returned", len(result_df))
                    num_cols = result_df.select_dtypes(include="number").columns
                    if len(num_cols):
                        mc[1].metric("Total Value", round(result_df[num_cols[0]].sum(), 2))
                    else:
                        mc[1].metric("Columns", len(result_df.columns))
                    mc[2].metric("Columns", len(result_df.columns))

                if len(result_df) > 1 and len(result_df.columns) >= 2:
                    top_row = result_df.sort_values(result_df.columns[1], ascending=False).iloc[0]

                    label = result_df.columns[0].replace("_", " ").title()

                    st.metric(
                        f"Top {label}",
                        f"{top_row.iloc[0]} ({top_row.iloc[1]:,.0f})"
                    )

            if "insights" in data and "sql_query" in data:
                ins_col, sql_col = st.columns([1.1, 1], gap="large")
                with ins_col:
                    st.markdown('<div class="section-label">Business Insights</div>', unsafe_allow_html=True)
                    bullets_html = "".join(
                        f'<div class="insight-item"><span class="insight-dot">•</span><span>{b.lstrip("•").strip()}</span></div>'
                        for b in data["insights"]
                    )
                    st.markdown(f'<div class="insight-card">{bullets_html}</div>', unsafe_allow_html=True)
                with sql_col:
                    st.markdown('<div class="section-label">Generated SQL</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="sql-block">{data["sql_query"]}</div>', unsafe_allow_html=True)

            if "result_df" in data and not data["result_df"].empty:
                result_df = data["result_df"]
                st.markdown(
                    f'<div class="section-label">Result · {len(result_df)} rows · {len(result_df.columns)} cols</div>',
                    unsafe_allow_html=True,
                )
                st.dataframe(result_df, use_container_width=True, hide_index=True)
                csv_bytes = result_df.to_csv(index=False).encode()
                st.download_button("↓ Download CSV", csv_bytes, "query_result.csv", "text/csv", key=f"dl_{id(msg)}")

            st.markdown("</div></div>", unsafe_allow_html=True)

        else:  # user
            st.markdown(f"""
            <div class="chat-row user-row">
                <div class="bubble user-bubble">{msg["content"]}</div>
                <div class="avatar-col">
                    <div class="avatar user-avatar">👤</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

_examples = [
    "Show total views by category",
    "Which category has the highest engagement",
    "Show the top 10 videos by views",
    "Compare views and likes",
    "Show total likes by category",
    "Show the number of videos per category",
    "Show the monthly likes trend",
    "Compare views and comments by category",
    "Which category has the highest views",
    "Show monthly views trend",
    "Which regions have the highest engagement",
    "Compare sentiment score by category",
]
disabled_class = "examples-disabled" if HIDE_EXAMPLES else ""
st.markdown(f'<div class="example-label">Try an example</div><div class="examples-wrap {disabled_class}">', unsafe_allow_html=True)
cols = st.columns(3)
for i, ex in enumerate(_examples):
    if cols[i % 3].button(ex, key=f"ex_{i}", disabled=HIDE_EXAMPLES):
        push_message("user", ex)
        st.session_state["pending_query"] = ex
        st.rerun()
st.markdown('</div>', unsafe_allow_html=True)

user_input = st.chat_input(
    placeholder="Ask anything about your data…",
    key="chat_input",
)

if user_input and user_input.strip():
    push_message("user", user_input.strip())
    st.session_state["pending_query"] = user_input.strip()
    st.rerun()

if st.session_state.get("pending_query"):
    user_input_to_run = st.session_state.pop("pending_query")

    try:
        query = user_input_to_run
        if st.session_state.last_query:
            query = resolve_followup(st.session_state.last_query, query)
        st.session_state.last_query = query

        with st.spinner("Generating dashboard…"):
            sql_query, result_df, chart_types, insights = cached_pipeline(
                st.session_state.df, st.session_state.schema, query
            )

        push_message(
            "assistant",
            f"Here's what I found for **{user_input_to_run}**:",
            {
                "sql_query": sql_query,
                "result_df": result_df,
                "chart_types": chart_types,
                "insights": insights,
            },
        )

    except ResourceExhausted:
        push_message("assistant", "⚠️ AI quota exceeded — please retry in a few seconds.")
    except PermissionError as e:
        push_message("assistant", f"🔒 Access denied: {e}")
    except ValueError as e:
        push_message("assistant", f"⚠️ Data error: {e}")
    except RuntimeError as e:
        push_message("assistant", f"⚙️ Processing error: {e}")
    except Exception as e:
        push_message("assistant", f"Unexpected error — {type(e).__name__}: {e}")
        logger.exception("Unhandled pipeline error")

    st.rerun()