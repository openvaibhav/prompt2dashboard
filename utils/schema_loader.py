from __future__ import annotations
from pathlib import Path
import pandas as pd

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
        sample = series.dropna().head(100)
        if len(sample) > 0:
            try:
                pd.to_datetime(sample)
                return "datetime"
            except (ValueError, TypeError):
                pass
        return "string"

    return str(dtype)


def _derive_table_name(csv_path: str | Path):
    
    return Path(csv_path).stem.lower()


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
            return pd.read_csv(file, encoding=enc)
        except UnicodeDecodeError:
            continue

    raise ValueError("Could not decode CSV with common encodings")


def load_schema(csv_input, *, parse_dates=True, encoding="utf-8", **read_csv_kwargs):
    
    common_kwargs = {"low_memory": False, **read_csv_kwargs}

    if not isinstance(csv_input, (str, Path)):
        try:
            csv_input.seek(0)
            df = pd.read_csv(csv_input, encoding="utf-8", **common_kwargs)
        except UnicodeDecodeError:
            df = pd.read_csv(csv_input, encoding="latin-1", **common_kwargs)

        table_name = "uploaded_dataset"

    else:
        csv_path = Path(csv_input)

        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path.resolve()}")

        try:
            df = pd.read_csv(csv_path, encoding="utf-8", **common_kwargs)
        except UnicodeDecodeError:
            df = pd.read_csv(csv_path, encoding="latin-1", **common_kwargs)

        table_name = csv_path.stem.lower()
        
    schema_description = _build_schema_description(table_name, df)

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(exclude="number").columns.tolist()

    schema_description += "\n\nNumeric columns: " + ", ".join(numeric_cols)
    schema_description += "\nCategorical columns: " + ", ".join(categorical_cols)

    return df, schema_description