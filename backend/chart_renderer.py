from __future__ import annotations
import logging
from typing import Optional
import pandas as pd
import plotly.express as px
from plotly.graph_objects import Figure

logger = logging.getLogger(__name__)

VALID_CHART_TYPES: frozenset[str] = frozenset({"bar", "line", "pie", "scatter"})

_COLOR_SEQUENCE = px.colors.qualitative.Plotly


def _get_categorical_columns(df: pd.DataFrame):
    return [
        col
        for col in df.columns
        if pd.api.types.is_string_dtype(df[col])
        or pd.api.types.is_object_dtype(df[col])
        or isinstance(df[col].dtype, pd.CategoricalDtype)
    ]


def _get_numeric_columns(df: pd.DataFrame):
    return [
        col
        for col in df.columns
        if pd.api.types.is_numeric_dtype(df[col])
        and not pd.api.types.is_bool_dtype(df[col])
    ]


def _detect_datetime_column(df: pd.DataFrame):
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return col
    return None


def _detect_axes(
    df: pd.DataFrame,
    prefer_datetime_x: bool = True,
):
    if df.shape[1] < 1:
        raise ValueError("DataFrame has no columns.")

    numeric_cols = _get_numeric_columns(df)
    categorical_cols = _get_categorical_columns(df)
    datetime_col = _detect_datetime_column(df)

    x_col: Optional[str] = None

    if prefer_datetime_x and datetime_col:
        x_col = datetime_col
    elif categorical_cols:
        x_col = categorical_cols[0]
    elif numeric_cols:
        x_col = numeric_cols[0]

    if x_col is None:
        x_col = df.columns[0]

    y_candidates = [c for c in numeric_cols if c != x_col]

    if not y_candidates:
        y_candidates = [c for c in df.columns if c != x_col]

    if not y_candidates:
        raise ValueError(
            "Cannot determine a y-axis column — DataFrame must have at least "
            "two columns, with at least one numeric column for the value axis."
        )

    y_col = y_candidates[0]

    logger.debug("Auto-detected axes: x=%r, y=%r", x_col, y_col)
    return x_col, y_col


def _detect_pie_columns(df: pd.DataFrame):
    categorical_cols = _get_categorical_columns(df)
    numeric_cols = _get_numeric_columns(df)

    if not numeric_cols:
        raise ValueError(
            "Pie chart requires at least one numeric column for slice values."
        )

    names_col = categorical_cols[0] if categorical_cols else df.columns[0]
    values_col = numeric_cols[0]

    if names_col == values_col:
        remaining = [c for c in df.columns if c != names_col]
        if not remaining:
            raise ValueError(
                "Pie chart requires at least two columns (a label column and a value column)."
            )
        values_col = remaining[0]

    logger.debug(
        "Auto-detected pie columns: names=%r, values=%r", names_col, values_col
    )
    return names_col, values_col


def _friendly_title(x_col: str, y_col: str, chart_type: str):
    x_label = x_col.replace("_", " ").title()
    y_label = y_col.replace("_", " ").title()
    return f"{y_label} by {x_label}"


def _bar(df: pd.DataFrame):
    x_col, y_col = _detect_axes(df, prefer_datetime_x=False)
    title = _friendly_title(x_col, y_col, "bar")
    fig = px.bar(
        df,
        x=x_col,
        y=y_col,
        title=title,
        color_discrete_sequence=_COLOR_SEQUENCE,
        text_auto=True,
    )
    fig.update_traces(
        hovertemplate="<b>%{x}</b><br>Value: %{y}<extra></extra>",
        textposition="outside",
    )
    fig.update_layout(
        xaxis_title=x_col.replace("_", " ").title(),
        yaxis_title=y_col.replace("_", " ").title(),
    )
    return fig


def _line(df: pd.DataFrame):
    x_col, y_col = _detect_axes(df, prefer_datetime_x=True)
    df = df.sort_values(by=x_col)
    title = _friendly_title(x_col, y_col, "line")
    fig = px.line(
        df,
        x=x_col,
        y=y_col,
        title=title,
        color_discrete_sequence=_COLOR_SEQUENCE,
        markers=True,
    )
    fig.update_traces(
        hovertemplate="<b>%{x}</b><br>Value: %{y}<extra></extra>",
        textposition="outside",
    )
    fig.update_layout(
        xaxis_title=x_col.replace("_", " ").title(),
        yaxis_title=y_col.replace("_", " ").title(),
    )
    return fig


def _pie(df: pd.DataFrame):
    names_col, values_col = _detect_pie_columns(df)
    df = df.nlargest(10, values_col)
    title = f"{values_col.replace('_', ' ').title()} by {names_col.replace('_', ' ').title()}"
    fig = px.pie(
        df,
        names=names_col,
        values=values_col,
        title=title,
        color_discrete_sequence=_COLOR_SEQUENCE,
        hole=0.3,
    )
    fig.update_traces(
        textposition="inside",
        textinfo="percent+label",
        hovertemplate="<b>%{label}</b><br>Value: %{value}<br>%{percent}<extra></extra>",
    )
    return fig


def _scatter(df: pd.DataFrame):
    numeric_cols = _get_numeric_columns(df)
    if len(numeric_cols) >= 2:
        x_col, y_col = numeric_cols[0], numeric_cols[1]
        logger.debug("Auto-detected scatter axes: x=%r, y=%r", x_col, y_col)
    else:
        x_col, y_col = _detect_axes(df, prefer_datetime_x=False)
    title = _friendly_title(x_col, y_col, "scatter")

    categorical_cols = _get_categorical_columns(df)
    color_col = categorical_cols[0] if categorical_cols else None

    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        color=color_col,
        title=title,
        color_discrete_sequence=_COLOR_SEQUENCE,
        trendline="ols" if len(df) >= 3 else None,
    )
    fig.update_traces(
        hovertemplate="<b>%{x}</b><br>Value: %{y}<extra></extra>",
    )
    fig.update_layout(
        xaxis_title=x_col.replace("_", " ").title(),
        yaxis_title=y_col.replace("_", " ").title(),
    )
    return fig


_CHART_BUILDERS = {
    "bar": _bar,
    "line": _line,
    "pie": _pie,
    "scatter": _scatter,
}


def render_chart(df: pd.DataFrame, chart_type: str):

    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"df must be a pandas DataFrame, got {type(df).__name__!r}.")
    if df.empty:
        raise ValueError("Cannot render a chart for an empty DataFrame.")

    chart_type = chart_type.strip().lower()
    if chart_type not in VALID_CHART_TYPES:
        raise ValueError(
            f"Unsupported chart type '{chart_type}'. "
            f"Choose one of: {', '.join(sorted(VALID_CHART_TYPES))}."
        )

    logger.debug("Rendering '%s' chart with columns: %s", chart_type, list(df.columns))

    if len(df) == 1 and chart_type in {"bar", "pie"}:
        return {
            "type": "metric",
            "label": df.columns[0],
            "value": df.iloc[0, 0] if df.shape[1] == 1 else df.iloc[0, 1],
            "category": df.iloc[0, 0] if df.shape[1] > 1 else None
        }

    builder = _CHART_BUILDERS[chart_type]
    fig = builder(df)

    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=13),
        margin=dict(t=50, l=10, r=10, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25),
        hovermode="x unified",
    )
    
    fig.update_xaxes(showgrid=True, gridcolor="rgba(200,200,200,0.1)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(200,200,200,0.1)")

    return fig