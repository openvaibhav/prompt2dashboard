from __future__ import annotations
from pathlib import Path
import pandas as pd
import warnings
import re

def _looks_like_datetime_column(col_name: str):
    col = col_name.lower()

    datetime_keywords = [
        "time",
        "timestamp",
        "date",
        "datetime",
        "created",
        "published",
        "upload",
        "day",
        "month",
        "year"
    ]

    return any(k in col for k in datetime_keywords)

def _infer_type_label(series: pd.Series):

    dtype = series.dtype

    if pd.api.types.is_datetime64_any_dtype(dtype):
        return "datetime"
    if pd.api.types.is_bool_dtype(dtype):
        return "boolean"
    if pd.api.types.is_integer_dtype(dtype):
        return "integer"
    if pd.api.types.is_float_dtype(dtype):
        return "float"

    if pd.api.types.is_object_dtype(dtype) or isinstance(dtype, pd.StringDtype):

        if _looks_like_datetime_column(series.name):
            sample = series.dropna().head(5)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                parsed = pd.to_datetime(sample, errors="coerce")

            if parsed.notna().all():
                return "datetime"

        return "string"

    return str(dtype)

def _build_schema_description(table_name: str, df: pd.DataFrame):

    lines: list[str] = [
        f"Table: {table_name}",
        "Columns:",
    ]

    for col in df.columns:
        type_label = _infer_type_label(df[col])
        sample_values = df[col].dropna().head(3).tolist()

        lines.append(
            f"- {col} ({type_label}) | example values: {sample_values}"
        )

    return "\n".join(lines)


def safe_read_csv(file):

    encodings = ["utf-8", "latin-1", "cp1252"]

    for enc in encodings:
        try:
            if hasattr(file, "seek"):
                file.seek(0)
            return pd.read_csv(file, encoding=enc)
        except UnicodeDecodeError:
            continue

    raise ValueError("Could not decode CSV with common encodings")

def _find_header_line(file):
    file.seek(0)
    
    for i in range(30):
        line = file.readline()
        
        if isinstance(line, bytes):
            line = line.decode(errors="ignore")
        
        line = line.strip()
        
        if not line:
            continue
        
        tokens = [t.strip() for t in line.split(",") if t.strip()]
        
        if len(tokens) < 2:
            continue
        
        non_numeric = sum(
            1 for t in tokens
            if not re.match(r'^-?\d+(\.\d+)?$', t.strip('"').strip("'"))
        )
        
        if non_numeric / len(tokens) >= 0.6:
            file.seek(0)
            return i
    
    file.seek(0)
    return 0

def _clean_column_names(df: pd.DataFrame):
    clean_cols = []
    seen = set()

    for col in df.columns:
        col = str(col)
        col = re.sub(r"<.*?>", "", col)
        col = re.sub(r"[^a-zA-Z0-9_]", " ", col)
        tokens = col.lower().split()

        if not tokens:
            col = "column"
        elif len("_".join(tokens)) > 40:
            priority_keywords = [
                "timestamp", "date", "time", "datetime",
                "id", "name", "url", "type", "code",
                "category", "region", "language"
            ]
            col = tokens[-1]
            for keyword in priority_keywords:
                if any(keyword in t for t in tokens):
                    col = keyword
                    break
        else:
            col = "_".join(tokens)

        base = col
        i = 1
        while col in seen:
            col = f"{base}_{i}"
            i += 1

        seen.add(col)
        clean_cols.append(col)

    df.columns = clean_cols
    return df


def load_schema(csv_input, *, parse_dates=True, encoding="utf-8", **read_csv_kwargs):
    common_kwargs = {"low_memory": False, **read_csv_kwargs}

    if not isinstance(csv_input, (str, Path)):
        header_line = _find_header_line(csv_input)
        csv_input.seek(0)

        try:
            df = pd.read_csv(csv_input, skiprows=header_line, encoding="utf-8", **common_kwargs)
        except UnicodeDecodeError:
            csv_input.seek(0)
            df = pd.read_csv(csv_input, skiprows=header_line, encoding="latin-1", **common_kwargs)

        table_name = "uploaded_dataset"

    else:
        csv_path = Path(csv_input)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path.resolve()}")

        with open(csv_path, "rb") as f:
            header_line = _find_header_line(f)

        try:
            df = pd.read_csv(csv_path, skiprows=header_line, encoding="utf-8", **common_kwargs)
        except UnicodeDecodeError:
            df = pd.read_csv(csv_path, skiprows=header_line, encoding="latin-1", **common_kwargs)

        table_name = csv_path.stem.lower()

    df = _clean_column_names(df)
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    df.columns = [c.lower() for c in df.columns]

    for col in df.columns:
        if _looks_like_datetime_column(col) and df[col].dtype == object:
            try:
                df.loc[:, col] = pd.to_datetime(df[col], errors="coerce")
            except Exception:
                pass

    # Guards
    if df.empty or len(df.columns) == 0:
        raise ValueError("The uploaded CSV appears to be empty or has no readable columns.")
    if len(df.columns) == 1:
        raise ValueError("CSV must have at least 2 columns to analyse.")

    schema_description = _build_schema_description(table_name, df)

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    datetime_cols = df.select_dtypes(include="datetime").columns.tolist()
    categorical_cols = df.select_dtypes(exclude=["number", "datetime"]).columns.tolist()

    schema_description += "\n\nNumeric columns: " + ", ".join(numeric_cols)
    schema_description += "\nCategorical columns: " + ", ".join(categorical_cols)

    return df, schema_description, numeric_cols, categorical_cols