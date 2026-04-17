import io
import os
import zipfile
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Optional diagnostics heatmap
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_SEABORN = True
except Exception:
    HAS_SEABORN = False

st.set_page_config(layout="wide", page_title="Welding Signal QC Playground - Prototype 2")

# ============================================================
# ------------------------- CONSTANTS -------------------------
# ============================================================

CHANNEL_FIXED_MAP = {
    0: "Channel 1",
    1: "Channel 2",
    2: "Channel 3",
}

RAW_LINE_WIDTH = 0.8
RAW_OPACITY = 0.25
CENTER_LINE_WIDTH = 3.0
CL_LINE_WIDTH = 2.0
BEAD_SEPARATOR_COLOR = "rgba(255,255,0,0.25)"  # 75% transparent yellow
CL_COLOR = "#00FF00"

# ============================================================
# ---------------------- SESSION DEFAULTS ---------------------
# ============================================================

def init_session():
    defaults = {
        "zip_name": None,
        "raw_data": None,
        "sorted_files": None,
        "column_names": None,
        "segmentation_done": False,
        "segmentation_locked": False,
        "grouping_locked": False,
        "bead_map": None,
        "common_beads": None,
        "group_assignments": None,
        "segmentation_channel_name": None,
        "segmentation_channel_idx": 0,
        "segmentation_threshold": 0.5,
        "group_size": 1,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()

# ============================================================
# -------------------------- HELPERS --------------------------
# ============================================================

def get_theme_center_color() -> str:
    base = st.get_option("theme.base")
    return "black" if base == "light" else "white"

def short_label(csv_name: str) -> str:
    base = os.path.splitext(os.path.basename(csv_name))[0]
    return base[:6] if len(base) > 6 else base

def load_zip_to_data(uploaded_zip) -> Tuple[Dict[str, pd.DataFrame], List[str], List[str]]:
    data = {}
    with zipfile.ZipFile(uploaded_zip, "r") as zf:
        csv_names = sorted([n for n in zf.namelist() if n.lower().endswith(".csv")])
        for name in csv_names:
            with zf.open(name) as f:
                df = pd.read_csv(f)
                data[os.path.basename(name)] = df
    if not data:
        return {}, [], []
    first_df = data[csv_names[0].split("/")[-1]]
    column_names = first_df.columns.tolist()
    return data, sorted(list(data.keys())), column_names

def segment_beads(df: pd.DataFrame, column_idx: int, threshold: float) -> List[Tuple[int, int]]:
    start_indices, end_indices = [], []
    signal = df.iloc[:, column_idx].to_numpy()
    i = 0
    while i < len(signal):
        if signal[i] > threshold:
            start = i
            while i < len(signal) and signal[i] > threshold:
                i += 1
            end = i - 1
            start_indices.append(start)
            end_indices.append(end)
        else:
            i += 1
    return list(zip(start_indices, end_indices))

def build_bead_map(raw_data: Dict[str, pd.DataFrame], seg_col_idx: int, threshold: float):
    bead_map = {}
    for fname, df in raw_data.items():
        bead_ranges = segment_beads(df, seg_col_idx, threshold)
        bead_dict = {}
        for bead_no, (start, end) in enumerate(bead_ranges, start=1):
            bead_dict[bead_no] = df.iloc[start:end+1].reset_index(drop=True)
        bead_map[fname] = bead_dict
    return bead_map

def get_common_beads(bead_map: Dict[str, Dict[int, pd.DataFrame]]) -> List[int]:
    if not bead_map:
        return []
    bead_sets = []
    for fname, bead_dict in bead_map.items():
        bead_sets.append(set(bead_dict.keys()))
    if not bead_sets:
        return []
    return sorted(list(set.intersection(*bead_sets))) if bead_sets else []

def compute_group_assignments(sorted_files: List[str], group_size: int) -> Dict[str, int]:
    group_assignments = {}
    if group_size <= 0:
        group_size = 1
    for idx, fname in enumerate(sorted_files):
        group_no = idx // group_size + 1
        group_assignments[fname] = group_no
    return group_assignments

def moving_average(y: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return y.copy()
    if window > len(y):
        window = len(y)
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode="same")

def gaussian_smooth(y: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return y.copy()
    if window % 2 == 0:
        window += 1
    x = np.linspace(-2, 2, window)
    g = np.exp(-(x**2) / 2)
    g = g / g.sum()
    return np.convolve(y, g, mode="same")

def savgol_like_smooth(y: np.ndarray, window: int, polyorder: int = 2) -> np.ndarray:
    # Lightweight fallback without scipy.
    # For prototype only: use repeated moving average approximation.
    if window <= 1:
        return y.copy()
    tmp = moving_average(y, window)
    if polyorder >= 2:
        tmp = moving_average(tmp, max(3, window // 2 * 2 + 1))
    return tmp

def apply_smoothing(y: np.ndarray, method: str, window: int, polyorder: int = 2) -> np.ndarray:
    if method == "None":
        return y.copy()
    if method == "Moving Average":
        return moving_average(y, window)
    if method == "Gaussian":
        return gaussian_smooth(y, window)
    if method == "Savitzky-Golay":
        return savgol_like_smooth(y, window, polyorder)
    return y.copy()

def compute_center(arr: np.ndarray, method: str) -> np.ndarray:
    if arr.size == 0:
        return np.array([])
    if method == "Mean":
        return np.mean(arr, axis=0)
    if method == "Median":
        return np.median(arr, axis=0)
    if method == "Trimmed Mean":
        if arr.shape[0] < 3:
            return np.mean(arr, axis=0)
        sorted_arr = np.sort(arr, axis=0)
        return np.mean(sorted_arr[1:-1], axis=0)
    return np.median(arr, axis=0)

def aggregate_for_step_1d(y: np.ndarray, interval: int) -> np.ndarray:
    if interval <= 0:
        interval = 1
    return np.array([np.mean(y[i:i+interval]) for i in range(0, len(y), interval)])

def create_adjusted_lines(
    lines: Dict[str, np.ndarray],
    method: str,
    reference_group: Optional[int],
    group_assignments: Dict[str, int],
    scope: str
) -> Dict[str, np.ndarray]:
    if method == "None":
        return {k: v.copy() for k, v in lines.items()}

    adjusted = {}
    filenames = list(lines.keys())

    # choose reference using mean level
    if reference_group is None:
        reference_files = filenames
    else:
        reference_files = [f for f in filenames if group_assignments.get(f) == reference_group]
        if not reference_files:
            reference_files = filenames

    if scope == "Whole Channel":
        ref_vals = np.concatenate([lines[f] for f in reference_files])
        ref_mean = np.mean(ref_vals)
        ref_std = np.std(ref_vals) if np.std(ref_vals) > 1e-12 else 1.0

        for f, y in lines.items():
            y_mean = np.mean(y)
            y_std = np.std(y) if np.std(y) > 1e-12 else 1.0

            if method == "Offset Shift":
                adjusted[f] = y - (y_mean - ref_mean)

            elif method == "Ratio Scaling":
                ratio = ref_mean / y_mean if abs(y_mean) > 1e-12 else 1.0
                adjusted[f] = y * ratio

            elif method == "Affine":
                adjusted[f] = ((y - y_mean) / y_std) * ref_std + ref_mean

            elif method == "Z-score Normalization":
                adjusted[f] = (y - y_mean) / y_std

            elif method == "Min-Max Normalization":
                ymin, ymax = np.min(y), np.max(y)
                denom = ymax - ymin if abs(ymax - ymin) > 1e-12 else 1.0
                adjusted[f] = (y - ymin) / denom

            elif method == "Per-Signal Normalization":
                denom = np.linalg.norm(y)
                denom = denom if denom > 1e-12 else 1.0
                adjusted[f] = y / denom

            else:
                adjusted[f] = y.copy()
    else:
        # Per Bead scope is handled bead-by-bead before concatenation in this prototype
        adjusted = {k: v.copy() for k, v in lines.items()}

    return adjusted

def compute_control_limits(
    lines_matrix: np.ndarray,
    method: str,
    center_method: str,
    params: dict
):
    """
    Returns:
        center, ucl, lcl, point_status_matrix(optional), aux(optional)
    """
    if lines_matrix.size == 0:
        return np.array([]), np.array([]), np.array([]), None, {}

    center = compute_center(lines_matrix, center_method)
    aux = {}
    point_status_matrix = None

    if method == "Mean ± k·Std":
        k = params.get("k", 3.0)
        mu = np.mean(lines_matrix, axis=0)
        sigma = np.std(lines_matrix, axis=0, ddof=1) if lines_matrix.shape[0] > 1 else np.zeros(lines_matrix.shape[1])
        ucl = mu + k * sigma
        lcl = mu - k * sigma

    elif method == "Median ± k·MAD":
        k = params.get("k", 3.0)
        med = np.median(lines_matrix, axis=0)
        mad = np.median(np.abs(lines_matrix - med), axis=0)
        ucl = med + k * mad
        lcl = med - k * mad

    elif method == "Percentile Band":
        low = params.get("low_pct", 5.0)
        high = params.get("high_pct", 95.0)
        ucl = np.percentile(lines_matrix, high, axis=0)
        lcl = np.percentile(lines_matrix, low, axis=0)

    elif method == "IQR Band":
        k = params.get("k", 1.5)
        q1 = np.percentile(lines_matrix, 25, axis=0)
        q3 = np.percentile(lines_matrix, 75, axis=0)
        iqr = q3 - q1
        ucl = q3 + k * iqr
        lcl = q1 - k * iqr

    elif method == "Z-score Band":
        z = params.get("z", 3.0)
        mu = np.mean(lines_matrix, axis=0)
        sigma = np.std(lines_matrix, axis=0, ddof=1) if lines_matrix.shape[0] > 1 else np.zeros(lines_matrix.shape[1])
        ucl = mu + z * sigma
        lcl = mu - z * sigma

    elif method == "Z-score Per Point":
        z = params.get("z", 3.0)
        mu = np.mean(lines_matrix, axis=0)
        sigma = np.std(lines_matrix, axis=0, ddof=1) if lines_matrix.shape[0] > 1 else np.ones(lines_matrix.shape[1])
        sigma = np.where(sigma < 1e-12, 1.0, sigma)
        zmat = (lines_matrix - mu) / sigma
        point_status_matrix = np.abs(zmat) > z
        ucl = mu + z * sigma
        lcl = mu - z * sigma
        aux["zmat"] = zmat

    elif method == "Step-based Norm (M)":
        interval = params.get("step_interval", 20)
        norm_low = params.get("norm_low", -3.0)
        norm_high = params.get("norm_high", 4.0)

        step_arrays = np.array([aggregate_for_step_1d(y, interval) for y in lines_matrix])
        min_steps = min(len(a) for a in step_arrays)
        step_arrays = np.array([a[:min_steps] for a in step_arrays])

        min_ref = np.min(step_arrays, axis=0)
        max_ref = np.max(step_arrays, axis=0)
        denom = max_ref - min_ref
        denom = np.where(denom < 1e-12, 1.0, denom)
        norm_matrix = (step_arrays - min_ref) / denom

        center_step = np.median(norm_matrix, axis=0)
        ucl_step = np.full(min_steps, norm_high)
        lcl_step = np.full(min_steps, norm_low)

        aux["step_mode"] = "norm"
        aux["interval"] = interval
        aux["step_matrix"] = norm_matrix
        aux["step_center"] = center_step
        aux["step_ucl"] = ucl_step
        aux["step_lcl"] = lcl_step
        point_status_matrix = (norm_matrix < norm_low) | (norm_matrix > norm_high)

        # Keep line-space UCL/LCL empty for this method
        ucl = np.array([])
        lcl = np.array([])

    elif method == "Step-based Z (M)":
        interval = params.get("step_interval", 20)
        z_low = params.get("z_low", 3.0)
        z_high = params.get("z_high", 3.0)

        step_arrays = np.array([aggregate_for_step_1d(y, interval) for y in lines_matrix])
        min_steps = min(len(a) for a in step_arrays)
        step_arrays = np.array([a[:min_steps] for a in step_arrays])

        mu = np.median(step_arrays, axis=0)
        sigma = np.std(step_arrays, axis=0, ddof=1) if step_arrays.shape[0] > 1 else np.ones(step_arrays.shape[1])
        sigma = np.where(sigma < 1e-12, 1.0, sigma)
        zmat = (step_arrays - mu) / sigma

        center_step = np.median(zmat, axis=0)
        ucl_step = np.full(min_steps, z_high)
        lcl_step = np.full(min_steps, -z_low)

        aux["step_mode"] = "z"
        aux["interval"] = interval
        aux["step_matrix"] = zmat
        aux["step_center"] = center_step
        aux["step_ucl"] = ucl_step
        aux["step_lcl"] = lcl_step
        point_status_matrix = (zmat < -z_low) | (zmat > z_high)

        ucl = np.array([])
        lcl = np.array([])

    else:
        ucl = np.array([])
        lcl = np.array([])

    return center, ucl, lcl, point_status_matrix, aux

def build_multibead_dataset(
    bead_map: Dict[str, Dict[int, pd.DataFrame]],
    selected_beads: List[int],
    channel_idx: int,
    sorted_files: List[str],
    smoothing_method: str,
    smoothing_window: int,
    smoothing_polyorder: int,
    per_bead_adjustment_method: str,
    per_bead_adjustment_reference_group: Optional[int],
    group_assignments: Dict[str, int],
    adjustment_scope: str,
):
    """
    Builds concatenated signal across beads using min bead length per bead across all files.
    Returns:
      line_map_before: {filename: concatenated_y}
      line_map_after : {filename: concatenated_y}
      bead_boundaries: list of x positions where each bead ends
      bead_length_table: pd.DataFrame
      bead_stat_table: pd.DataFrame
    """
    line_map_before = {f: [] for f in sorted_files}
    line_map_after = {f: [] for f in sorted_files}
    bead_boundaries = []
    bead_length_rows = []
    bead_stat_rows = []

    offset = 0

    for bead_no in selected_beads:
        available = []
        for fname in sorted_files:
            if bead_no in bead_map[fname]:
                y = bead_map[fname][bead_no].iloc[:, channel_idx].to_numpy(dtype=float)
                available.append((fname, y))

        if not available:
            continue

        min_len = min(len(y) for _, y in available)
        bead_before = {}
        for fname, y in available:
            ys = y[:min_len]
            ys = apply_smoothing(ys, smoothing_method, smoothing_window, smoothing_polyorder)
            bead_before[fname] = ys

        # per-bead adjustment if requested
        if adjustment_scope == "Per Bead":
            bead_after = create_adjusted_lines(
                bead_before,
                per_bead_adjustment_method,
                per_bead_adjustment_reference_group,
                group_assignments,
                "Whole Channel"
            )
        else:
            bead_after = {k: v.copy() for k, v in bead_before.items()}

        for fname in sorted_files:
            if fname in bead_before:
                line_map_before[fname].extend(bead_before[fname].tolist())
                line_map_after[fname].extend(bead_after[fname].tolist())

                bead_length_rows.append({
                    "file": fname,
                    "bead": bead_no,
                    "length": min_len
                })
                bead_stat_rows.append({
                    "file": fname,
                    "bead": bead_no,
                    "mean": float(np.mean(bead_before[fname])),
                    "median": float(np.median(bead_before[fname])),
                })

        offset += min_len
        bead_boundaries.append(offset)

    # whole-channel adjustment after concatenation if requested
    if adjustment_scope == "Whole Channel":
        line_map_after = create_adjusted_lines(
            {k: np.array(v, dtype=float) for k, v in line_map_before.items()},
            per_bead_adjustment_method,
            per_bead_adjustment_reference_group,
            group_assignments,
            "Whole Channel"
        )
    else:
        line_map_after = {k: np.array(v, dtype=float) for k, v in line_map_after.items()}

    line_map_before = {k: np.array(v, dtype=float) for k, v in line_map_before.items()}

    bead_length_table = pd.DataFrame(bead_length_rows)
    bead_stat_table = pd.DataFrame(bead_stat_rows)

    return line_map_before, line_map_after, bead_boundaries, bead_length_table, bead_stat_table

def build_display_entities(
    line_map: Dict[str, np.ndarray],
    sorted_files: List[str],
    group_assignments: Dict[str, int],
    group_size: int,
    display_mode: str,
    center_method: str
):
    """
    Returns entities to plot:
      raw_entities: list[(legend_name, y)]
      grouped_center_entities: list[(legend_name, y)]
      matrix_for_cl: np.ndarray
    """
    raw_entities = []
    grouped_center_entities = []

    available_files = [f for f in sorted_files if f in line_map and len(line_map[f]) > 0]

    if group_size <= 1:
        for f in available_files:
            raw_entities.append((short_label(f), line_map[f]))
        matrix_for_cl = np.vstack([line_map[f] for f in available_files]) if available_files else np.empty((0, 0))
        grouped_center_entities = []
        return raw_entities, grouped_center_entities, matrix_for_cl

    # group_size > 1
    group_nos = sorted(set(group_assignments[f] for f in available_files))
    for f in available_files:
        raw_entities.append((short_label(f), line_map[f]))

    group_representatives = []
    for g in group_nos:
        members = [line_map[f] for f in available_files if group_assignments[f] == g]
        if not members:
            continue
        mat = np.vstack(members)
        center = compute_center(mat, center_method)
        grouped_center_entities.append((f"Group {g}", center))
        group_representatives.append(center)

    matrix_for_cl = np.vstack(group_representatives) if group_representatives else np.empty((0, 0))

    return raw_entities, grouped_center_entities, matrix_for_cl

def add_bead_separators(fig: go.Figure, bead_boundaries: List[int]):
    for x in bead_boundaries[:-1]:
        fig.add_vline(
            x=x - 0.5,
            line=dict(color=BEAD_SEPARATOR_COLOR, width=2, dash="dot")
        )

def add_regular_plot_traces(
    fig: go.Figure,
    raw_entities: List[Tuple[str, np.ndarray]],
    grouped_center_entities: List[Tuple[str, np.ndarray]],
    bead_boundaries: List[int],
    center: np.ndarray,
    ucl: np.ndarray,
    lcl: np.ndarray,
    group_size: int,
    display_mode: str,
    title: str
):
    center_color = get_theme_center_color()

    # raw
    if group_size <= 1:
        for legend_name, y in raw_entities:
            fig.add_trace(go.Scatter(
                x=np.arange(len(y)),
                y=y,
                mode="lines",
                name=legend_name,
                line=dict(width=RAW_LINE_WIDTH),
                opacity=RAW_OPACITY,
                legendgroup=legend_name,
                showlegend=True
            ))
    else:
        if display_mode in ["Show Ungrouped Signal", "Show All Signal"]:
            for legend_name, y in raw_entities:
                fig.add_trace(go.Scatter(
                    x=np.arange(len(y)),
                    y=y,
                    mode="lines",
                    name=legend_name,
                    line=dict(width=RAW_LINE_WIDTH),
                    opacity=RAW_OPACITY,
                    legendgroup=legend_name,
                    showlegend=(display_mode != "Show Grouped Signal")
                ))

        if display_mode in ["Show Grouped Signal", "Show All Signal"]:
            for legend_name, y in grouped_center_entities:
                fig.add_trace(go.Scatter(
                    x=np.arange(len(y)),
                    y=y,
                    mode="lines",
                    name=legend_name,
                    line=dict(width=1.2),
                    opacity=0.85,
                    legendgroup=legend_name,
                    showlegend=True
                ))

    # center
    if len(center) > 0:
        fig.add_trace(go.Scatter(
            x=np.arange(len(center)),
            y=center,
            mode="lines",
            name="Center",
            line=dict(color=center_color, width=CENTER_LINE_WIDTH),
            legendgroup="Center",
            showlegend=True
        ))

    # control limits
    if len(ucl) > 0:
        fig.add_trace(go.Scatter(
            x=np.arange(len(ucl)),
            y=ucl,
            mode="lines",
            name="UCL",
            line=dict(color=CL_COLOR, width=CL_LINE_WIDTH, dash="dash"),
            legendgroup="UCL",
            showlegend=True
        ))
    if len(lcl) > 0:
        fig.add_trace(go.Scatter(
            x=np.arange(len(lcl)),
            y=lcl,
            mode="lines",
            name="LCL",
            line=dict(color=CL_COLOR, width=CL_LINE_WIDTH, dash="dash"),
            legendgroup="LCL",
            showlegend=True
        ))

    add_bead_separators(fig, bead_boundaries)
    fig.update_layout(
        title=title,
        xaxis_title="Concatenated Index Across Beads",
        yaxis_title="Signal Value",
        legend=dict(orientation="h"),
        margin=dict(l=30, r=20, t=60, b=40),
    )

def add_step_plot_traces(
    fig: go.Figure,
    raw_entities: List[Tuple[str, np.ndarray]],
    grouped_center_entities: List[Tuple[str, np.ndarray]],
    aux: dict,
    point_status_matrix: Optional[np.ndarray],
    group_size: int,
    display_mode: str,
    title: str
):
    step_matrix = aux["step_matrix"]
    center_step = aux["step_center"]
    ucl_step = aux["step_ucl"]
    lcl_step = aux["step_lcl"]
    center_color = get_theme_center_color()

    # plot step-space grouped or raw
    if group_size <= 1:
        for i, (legend_name, _) in enumerate(raw_entities):
            y = step_matrix[i]
            fig.add_trace(go.Scatter(
                x=np.arange(len(y)),
                y=y,
                mode="lines",
                name=legend_name,
                line=dict(width=RAW_LINE_WIDTH),
                opacity=RAW_OPACITY,
                legendgroup=legend_name,
                showlegend=True
            ))
    else:
        if display_mode in ["Show Ungrouped Signal", "Show All Signal"]:
            # use first N raw entities matching step_matrix rows
            for i, (legend_name, _) in enumerate(raw_entities[:len(step_matrix)]):
                y = step_matrix[i]
                fig.add_trace(go.Scatter(
                    x=np.arange(len(y)),
                    y=y,
                    mode="lines",
                    name=legend_name,
                    line=dict(width=RAW_LINE_WIDTH),
                    opacity=RAW_OPACITY,
                    legendgroup=legend_name,
                    showlegend=(display_mode != "Show Grouped Signal")
                ))

        if display_mode in ["Show Grouped Signal", "Show All Signal"]:
            # build group means in step-space
            grouped = {}
            cursor = 0
            for legend_name, _ in raw_entities:
                group_name = legend_name
                grouped.setdefault(group_name, [])
            # Instead of raw filename grouping in legend, use grouped_center_entities names
            # Create from original row ordering
            # Simpler prototype: skip separate grouped step traces if raw shown; plot only center + UCL/LCL
            pass

    fig.add_trace(go.Scatter(
        x=np.arange(len(center_step)),
        y=center_step,
        mode="lines",
        name="Center",
        line=dict(color=center_color, width=CENTER_LINE_WIDTH),
        legendgroup="Center",
        showlegend=True
    ))
    fig.add_trace(go.Scatter(
        x=np.arange(len(ucl_step)),
        y=ucl_step,
        mode="lines",
        name="UCL",
        line=dict(color=CL_COLOR, width=CL_LINE_WIDTH, dash="dash"),
        legendgroup="UCL",
        showlegend=True
    ))
    fig.add_trace(go.Scatter(
        x=np.arange(len(lcl_step)),
        y=lcl_step,
        mode="lines",
        name="LCL",
        line=dict(color=CL_COLOR, width=CL_LINE_WIDTH, dash="dash"),
        legendgroup="LCL",
        showlegend=True
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Step Index",
        yaxis_title="Step-based Value",
        legend=dict(orientation="h"),
        margin=dict(l=30, r=20, t=60, b=40),
    )

def get_formula_text(center_method: str, cl_method: str, adjustment_method: str, params: dict) -> str:
    lines = []
    lines.append(f"Center method: {center_method}")

    if cl_method == "Mean ± k·Std":
        k = params.get("k", 3.0)
        lines.append(f"UCL = mean(x) + {k}·std(x)")
        lines.append(f"LCL = mean(x) - {k}·std(x)")

    elif cl_method == "Median ± k·MAD":
        k = params.get("k", 3.0)
        lines.append(f"UCL = median(x) + {k}·MAD(x)")
        lines.append(f"LCL = median(x) - {k}·MAD(x)")

    elif cl_method == "Percentile Band":
        low = params.get("low_pct", 5.0)
        high = params.get("high_pct", 95.0)
        lines.append(f"LCL = percentile(x, {low})")
        lines.append(f"UCL = percentile(x, {high})")

    elif cl_method == "IQR Band":
        k = params.get("k", 1.5)
        lines.append(f"UCL = Q3 + {k}·IQR")
        lines.append(f"LCL = Q1 - {k}·IQR")

    elif cl_method == "Z-score Band":
        z = params.get("z", 3.0)
        lines.append(f"UCL = mean(x) + {z}·std(x)")
        lines.append(f"LCL = mean(x) - {z}·std(x)")

    elif cl_method == "Z-score Per Point":
        z = params.get("z", 3.0)
        lines.append(f"z_i = (x_i - mean_i) / std_i")
        lines.append(f"Flag if |z_i| > {z}")

    elif cl_method == "Step-based Norm (M)":
        interval = params.get("step_interval", 20)
        low = params.get("norm_low", -3.0)
        high = params.get("norm_high", 4.0)
        lines.append(f"step_y = mean(x[i:i+{interval}])")
        lines.append(f"norm = (step_y - min_ref) / (max_ref - min_ref)")
        lines.append(f"Flag if norm < {low} or norm > {high}")

    elif cl_method == "Step-based Z (M)":
        interval = params.get("step_interval", 20)
        z_low = params.get("z_low", 3.0)
        z_high = params.get("z_high", 3.0)
        lines.append(f"step_y = mean(x[i:i+{interval}])")
        lines.append(f"z = (step_y - median_ref) / std_ref")
        lines.append(f"Flag if z < -{z_low} or z > {z_high}")

    lines.append(f"Adjustment: {adjustment_method}")
    return "\n".join(lines)

# ============================================================
# -------------------------- SIDEBAR --------------------------
# ============================================================

st.sidebar.header("Prototype 2")

uploaded_zip = st.sidebar.file_uploader("Upload ZIP of CSV files", type="zip")

# If new file uploaded, reset relevant session state
if uploaded_zip is not None:
    if st.session_state.zip_name != uploaded_zip.name:
        st.session_state.zip_name = uploaded_zip.name
        st.session_state.raw_data = None
        st.session_state.sorted_files = None
        st.session_state.column_names = None
        st.session_state.segmentation_done = False
        st.session_state.segmentation_locked = False
        st.session_state.grouping_locked = False
        st.session_state.bead_map = None
        st.session_state.common_beads = None
        st.session_state.group_assignments = None

        raw_data, sorted_files, column_names = load_zip_to_data(uploaded_zip)
        st.session_state.raw_data = raw_data
        st.session_state.sorted_files = sorted_files
        st.session_state.column_names = column_names

# ============================================================
# ----------------------- MAIN STATUS -------------------------
# ============================================================

st.title("Welding Signal QC Playground - Prototype 2")

if st.session_state.raw_data is None:
    st.info("Upload a ZIP file from the sidebar to begin.")
    st.stop()

raw_data = st.session_state.raw_data
sorted_files = st.session_state.sorted_files
column_names = st.session_state.column_names

st.write(f"Loaded files: **{len(sorted_files)}**")
st.write(f"Detected columns: **{len(column_names)}**")

# ============================================================
# ----------------------- SEGMENTATION ------------------------
# ============================================================

st.sidebar.subheader("1) Bead Segmentation")

seg_options = column_names[:min(3, len(column_names))]
default_idx = 0
if st.session_state.segmentation_channel_name in seg_options:
    default_idx = seg_options.index(st.session_state.segmentation_channel_name)

seg_channel_name = st.sidebar.selectbox(
    "Segmentation Channel",
    options=seg_options,
    index=default_idx
)
seg_channel_idx = column_names.index(seg_channel_name)
seg_threshold = st.sidebar.number_input(
    "Segmentation Threshold",
    value=float(st.session_state.segmentation_threshold),
    step=0.1
)

run_seg = st.sidebar.button("Run Bead Segmentation")
lock_seg = st.sidebar.button("Lock Segmentation")

if run_seg:
    bead_map = build_bead_map(raw_data, seg_channel_idx, seg_threshold)
    common_beads = get_common_beads(bead_map)

    st.session_state.bead_map = bead_map
    st.session_state.common_beads = common_beads
    st.session_state.segmentation_done = True
    st.session_state.segmentation_channel_name = seg_channel_name
    st.session_state.segmentation_channel_idx = seg_channel_idx
    st.session_state.segmentation_threshold = seg_threshold
    st.session_state.grouping_locked = False
    st.success("Bead segmentation completed.")

if lock_seg and st.session_state.segmentation_done:
    st.session_state.segmentation_locked = True
    st.sidebar.success("Segmentation locked.")

if not st.session_state.segmentation_done:
    st.warning("Run bead segmentation first.")
    st.stop()

bead_map = st.session_state.bead_map
common_beads = st.session_state.common_beads

if not common_beads:
    st.error("No common bead numbers found across all files after segmentation.")
    st.stop()

# ============================================================
# ------------------------ GROUPING ---------------------------
# ============================================================

st.sidebar.subheader("2) Grouping")

group_size = st.sidebar.number_input(
    "Group Size",
    min_value=1,
    value=int(st.session_state.group_size),
    step=1
)
apply_grouping = st.sidebar.button("Apply Grouping")
lock_grouping = st.sidebar.button("Lock Grouping")

if apply_grouping or st.session_state.group_assignments is None:
    st.session_state.group_size = group_size
    st.session_state.group_assignments = compute_group_assignments(sorted_files, group_size)

if lock_grouping:
    st.session_state.grouping_locked = True
    st.sidebar.success("Grouping locked.")

group_assignments = st.session_state.group_assignments
group_size = st.session_state.group_size
total_groups = max(group_assignments.values()) if group_assignments else 0

# ============================================================
# --------------------- COMMON CONTROLS -----------------------
# ============================================================

st.sidebar.subheader("3) Signal Processing")

smoothing_method = st.sidebar.selectbox(
    "Smoothing Method",
    ["None", "Moving Average", "Savitzky-Golay", "Gaussian"]
)
smoothing_window = st.sidebar.number_input("Smoothing Window / Step", min_value=1, value=11, step=2)
smoothing_polyorder = st.sidebar.number_input("Savitzky-Golay Polyorder", min_value=1, value=2, step=1)
show_smoothed_only = st.sidebar.checkbox("Display Smoothed Line Only", value=False)

st.sidebar.subheader("4) Bead and Display")
selected_beads = st.sidebar.multiselect(
    "Bead Numbers to Display",
    options=common_beads,
    default=common_beads
)

if not selected_beads:
    st.warning("Select at least one bead.")
    st.stop()

center_method = st.sidebar.selectbox(
    "Center Method",
    ["Median", "Mean", "Trimmed Mean"],
    index=0
)

if group_size > 1:
    display_mode = st.sidebar.radio(
        "Grouped Display Mode",
        ["Show Ungrouped Signal", "Show Grouped Signal", "Show All Signal"],
        index=2
    )
else:
    display_mode = "Show All Signal"

st.sidebar.subheader("5) Adjustment")
adjustment_method = st.sidebar.selectbox(
    "Adjustment Method",
    ["None", "Offset Shift", "Ratio Scaling", "Affine", "Z-score Normalization", "Min-Max Normalization", "Per-Signal Normalization"],
    index=0
)
adjustment_scope = st.sidebar.selectbox(
    "Adjustment Scope",
    ["Per Bead", "Whole Channel"],
    index=0
)
reference_group = st.sidebar.selectbox(
    "Adjustment Reference Group",
    options=["All Groups"] + [f"Group {i}" for i in range(1, total_groups + 1)],
    index=0
)
reference_group_no = None if reference_group == "All Groups" else int(reference_group.split()[-1])

st.sidebar.subheader("6) Control Limit / QC Method")
cl_method = st.sidebar.selectbox(
    "Control Limit Method",
    [
        "Mean ± k·Std",
        "Median ± k·MAD",
        "Percentile Band",
        "IQR Band",
        "Z-score Band",
        "Z-score Per Point",
        "Step-based Norm (M)",
        "Step-based Z (M)",
    ],
    index=0
)

cl_params = {}

if cl_method in ["Mean ± k·Std", "Median ± k·MAD", "IQR Band"]:
    cl_params["k"] = st.sidebar.number_input("k", min_value=0.1, value=3.0 if cl_method != "IQR Band" else 1.5, step=0.1)

if cl_method == "Percentile Band":
    cl_params["low_pct"] = st.sidebar.number_input("Lower Percentile", min_value=0.0, max_value=50.0, value=5.0, step=0.5)
    cl_params["high_pct"] = st.sidebar.number_input("Upper Percentile", min_value=50.0, max_value=100.0, value=95.0, step=0.5)

if cl_method in ["Z-score Band", "Z-score Per Point"]:
    cl_params["z"] = st.sidebar.number_input("± Z-score", min_value=0.1, value=3.0, step=0.1)

if cl_method == "Step-based Norm (M)":
    cl_params["step_interval"] = st.sidebar.number_input("Step Interval", min_value=1, value=20, step=1)
    cl_params["norm_low"] = st.sidebar.number_input("Norm_Low", value=-3.0, step=0.1)
    cl_params["norm_high"] = st.sidebar.number_input("Norm_High", value=4.0, step=0.1)

if cl_method == "Step-based Z (M)":
    cl_params["step_interval"] = st.sidebar.number_input("Step Interval", min_value=1, value=20, step=1)
    cl_params["z_low"] = st.sidebar.number_input("Z_Low", min_value=0.1, value=3.0, step=0.1)
    cl_params["z_high"] = st.sidebar.number_input("Z_High", min_value=0.1, value=3.0, step=0.1)

# ============================================================
# ---------------------- STATUS PANEL -------------------------
# ============================================================

st.markdown("### Current Status")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Files", len(sorted_files))
c2.metric("Common Beads", len(common_beads))
c3.metric("Selected Beads", len(selected_beads))
c4.metric("Group Size", group_size)
c5.metric("Groups", total_groups)

# ============================================================
# ------------------------- TABS ------------------------------
# ============================================================

tab1, tab2, tab3, tab4 = st.tabs(["Channel 1", "Channel 2", "Channel 3", "Diagnostics"])

for channel_idx, tab in zip([0, 1, 2], [tab1, tab2, tab3]):
    with tab:
        st.subheader(f"{CHANNEL_FIXED_MAP[channel_idx]}")

        if channel_idx >= len(column_names):
            st.info("This dataset has fewer columns than expected for this channel.")
            continue

        line_map_before, line_map_after, bead_boundaries, bead_length_table, bead_stat_table = build_multibead_dataset(
            bead_map=bead_map,
            selected_beads=selected_beads,
            channel_idx=channel_idx,
            sorted_files=sorted_files,
            smoothing_method=smoothing_method,
            smoothing_window=int(smoothing_window),
            smoothing_polyorder=int(smoothing_polyorder),
            per_bead_adjustment_method=adjustment_method,
            per_bead_adjustment_reference_group=reference_group_no,
            group_assignments=group_assignments,
            adjustment_scope=adjustment_scope,
        )

        # BEFORE
        raw_entities_before, grouped_entities_before, cl_matrix_before = build_display_entities(
            line_map=line_map_before,
            sorted_files=sorted_files,
            group_assignments=group_assignments,
            group_size=group_size,
            display_mode=display_mode,
            center_method=center_method,
        )

        center_before, ucl_before, lcl_before, point_status_before, aux_before = compute_control_limits(
            cl_matrix_before,
            cl_method,
            center_method,
            cl_params
        )

        fig_before = go.Figure()

        if cl_method in ["Step-based Norm (M)", "Step-based Z (M)"]:
            add_step_plot_traces(
                fig_before,
                raw_entities_before,
                grouped_entities_before,
                aux_before,
                point_status_before,
                group_size,
                display_mode,
                title=f"Before Adjustment - {CHANNEL_FIXED_MAP[channel_idx]}",
            )
        else:
            add_regular_plot_traces(
                fig_before,
                raw_entities_before,
                grouped_entities_before,
                bead_boundaries,
                center_before,
                ucl_before,
                lcl_before,
                group_size,
                display_mode,
                title=f"Before Adjustment - {CHANNEL_FIXED_MAP[channel_idx]}",
            )

        st.plotly_chart(fig_before, use_container_width=True)

        # AFTER
        raw_entities_after, grouped_entities_after, cl_matrix_after = build_display_entities(
            line_map=line_map_after,
            sorted_files=sorted_files,
            group_assignments=group_assignments,
            group_size=group_size,
            display_mode=display_mode,
            center_method=center_method,
        )

        center_after, ucl_after, lcl_after, point_status_after, aux_after = compute_control_limits(
            cl_matrix_after,
            cl_method,
            center_method,
            cl_params
        )

        fig_after = go.Figure()

        if cl_method in ["Step-based Norm (M)", "Step-based Z (M)"]:
            add_step_plot_traces(
                fig_after,
                raw_entities_after,
                grouped_entities_after,
                aux_after,
                point_status_after,
                group_size,
                display_mode,
                title=f"After Adjustment - {CHANNEL_FIXED_MAP[channel_idx]}",
            )
        else:
            add_regular_plot_traces(
                fig_after,
                raw_entities_after,
                grouped_entities_after,
                bead_boundaries,
                center_after,
                ucl_after,
                lcl_after,
                group_size,
                display_mode,
                title=f"After Adjustment - {CHANNEL_FIXED_MAP[channel_idx]}",
            )

        st.plotly_chart(fig_after, use_container_width=True)

with tab4:
    st.subheader("Diagnostics")

    # Build diagnostics from segmentation once
    bead_length_rows = []
    bead_stat_rows = []

    for fname in sorted_files:
        for bead_no in common_beads:
            if bead_no in bead_map[fname]:
                bead_df = bead_map[fname][bead_no]
                bead_length_rows.append({
                    "file": fname,
                    "bead": bead_no,
                    "length": len(bead_df)
                })

                for ch_idx in [0, 1, 2]:
                    if ch_idx < bead_df.shape[1]:
                        y = bead_df.iloc[:, ch_idx].to_numpy(dtype=float)
                        bead_stat_rows.append({
                            "file": fname,
                            "bead": bead_no,
                            "channel": CHANNEL_FIXED_MAP[ch_idx],
                            "mean": float(np.mean(y)),
                            "median": float(np.median(y)),
                        })

    df_length = pd.DataFrame(bead_length_rows)
    df_stats = pd.DataFrame(bead_stat_rows)

    total_rows = sum(len(df) for df in raw_data.values())
    avg_beads_per_file = df_length.groupby("file")["bead"].nunique().mean() if not df_length.empty else 0
    avg_bead_length = df_length["length"].mean() if not df_length.empty else 0

    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Total Rows", int(total_rows))
    a2.metric("Files", len(sorted_files))
    a3.metric("Common Beads", len(common_beads))
    a4.metric("Avg Bead Length", f"{avg_bead_length:.1f}")

    st.markdown("### Group Assignments")
    group_table = pd.DataFrame({
        "file": sorted_files,
        "group": [group_assignments[f] for f in sorted_files]
    })
    st.dataframe(group_table, use_container_width=True)

    st.markdown("### Bead Length Heatmap")
    if not df_length.empty:
        heat_len = df_length.pivot(index="file", columns="bead", values="length")
        if HAS_SEABORN:
            fig, ax = plt.subplots(figsize=(10, max(4, len(heat_len) * 0.35)))
            sns.heatmap(heat_len, cmap="viridis", ax=ax)
            ax.set_xlabel("Bead Number")
            ax.set_ylabel("File")
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.dataframe(heat_len, use_container_width=True)
    else:
        st.info("No bead length data available.")

    st.markdown("### Signal Statistic Heatmap")
    stat_mode = st.selectbox("Statistic for Heatmap", ["Mean", "Median"], index=1)
    stat_channel = st.selectbox("Channel for Heatmap", ["Channel 1", "Channel 2", "Channel 3"], index=0)
    if not df_stats.empty:
        col = "mean" if stat_mode == "Mean" else "median"
        dfh = df_stats[df_stats["channel"] == stat_channel]
        heat_stat = dfh.pivot(index="file", columns="bead", values=col)
        if HAS_SEABORN:
            fig, ax = plt.subplots(figsize=(10, max(4, len(heat_stat) * 0.35)))
            sns.heatmap(heat_stat, cmap="magma", ax=ax)
            ax.set_xlabel("Bead Number")
            ax.set_ylabel("File")
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.dataframe(heat_stat, use_container_width=True)
    else:
        st.info("No signal statistic data available.")

    st.markdown("### Formula Summary")
    formula_text = get_formula_text(center_method, cl_method, adjustment_method, cl_params)
    st.code(formula_text, language="text")

    st.markdown("### Raw Tables")
    if not df_length.empty:
        with st.expander("Bead Length Table", expanded=False):
            st.dataframe(df_length, use_container_width=True)
    if not df_stats.empty:
        with st.expander("Signal Statistics Table", expanded=False):
            st.dataframe(df_stats, use_container_width=True)
