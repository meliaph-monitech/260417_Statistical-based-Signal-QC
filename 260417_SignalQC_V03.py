import streamlit as st
import pandas as pd
import numpy as np
import zipfile
import plotly.graph_objects as go

st.set_page_config(layout="wide")

# ============================================================
# ---------------------- UTILITIES ----------------------------
# ============================================================

def parse_bead_input(text, max_bead):
    result = set()
    if not text.strip():
        return []

    tokens = text.split(",")

    for token in tokens:
        token = token.strip()

        if "-" in token:
            try:
                start, end = map(int, token.split("-"))
                if start > end:
                    start, end = end, start
                for i in range(start, end + 1):
                    if 1 <= i <= max_bead:
                        result.add(i)
            except:
                continue
        else:
            try:
                val = int(token)
                if 1 <= val <= max_bead:
                    result.add(val)
            except:
                continue

    return sorted(result)


def segment_beads(df, col_idx, threshold):
    signal = df.iloc[:, col_idx].to_numpy()
    segments = []
    i = 0
    while i < len(signal):
        if signal[i] > threshold:
            start = i
            while i < len(signal) and signal[i] > threshold:
                i += 1
            end = i - 1
            segments.append((start, end))
        else:
            i += 1
    return segments


def load_zip(uploaded_zip):
    data = {}
    with zipfile.ZipFile(uploaded_zip) as z:
        names = sorted([n for n in z.namelist() if n.endswith(".csv")])
        for name in names:
            with z.open(name) as f:
                df = pd.read_csv(f)
                data[name] = df
    return data, names


def smooth(y, window):
    return np.convolve(y, np.ones(window)/window, mode="same")


def compute_center(arr, method):
    if method == "Mean":
        return np.mean(arr, axis=0)
    elif method == "Median":
        return np.median(arr, axis=0)
    elif method == "Trimmed":
        return np.mean(np.sort(arr, axis=0)[1:-1], axis=0)
    return np.mean(arr, axis=0)


def compute_cl(arr, method, params):
    if arr.size == 0:
        return None, None

    if method == "Mean±Std":
        m = np.mean(arr, axis=0)
        s = np.std(arr, axis=0)
        k = params["k"]
        return m + k*s, m - k*s

    elif method == "Median±MAD":
        med = np.median(arr, axis=0)
        mad = np.median(np.abs(arr - med), axis=0)
        k = params["k"]
        return med + k*mad, med - k*mad

    elif method == "Percentile":
        return np.percentile(arr, params["high"], axis=0), \
               np.percentile(arr, params["low"], axis=0)

    elif method == "Z-score":
        m = np.mean(arr, axis=0)
        s = np.std(arr, axis=0)
        z = params["z"]
        return m + z*s, m - z*s

    return None, None


def adjust_offset(arr):
    ref = np.mean(arr[0])
    out = []
    for a in arr:
        out.append(a - (np.mean(a) - ref))
    return np.array(out)


# ============================================================
# ---------------------- SIDEBAR ------------------------------
# ============================================================

st.sidebar.header("Prototype 3")

uploaded = st.sidebar.file_uploader("Upload ZIP", type="zip")

if uploaded:

    data, names = load_zip(uploaded)
    first_df = list(data.values())[0]
    columns = first_df.columns.tolist()

    st.sidebar.subheader("Segmentation")

    col_name = st.sidebar.selectbox("Channel", columns[:3])
    col_idx = columns.index(col_name)

    threshold = st.sidebar.number_input("Threshold", value=0.5)

    if st.sidebar.button("Run Segmentation"):
        bead_map = {}
        for fname, df in data.items():
            segs = segment_beads(df, col_idx, threshold)
            bead_map[fname] = {}
            for i, (s, e) in enumerate(segs, start=1):
                bead_map[fname][i] = df.iloc[s:e+1].reset_index(drop=True)

        st.session_state.bead_map = bead_map
        st.session_state.files = sorted(names)

    if "bead_map" not in st.session_state:
        st.stop()

    bead_map = st.session_state.bead_map
    files = st.session_state.files

    # ========================================================
    # BEAD SELECTION (NEW)
    # ========================================================

    st.sidebar.subheader("Bead Selection")

    all_beads = set()
    for f in bead_map.values():
        all_beads.update(f.keys())
    all_beads = sorted(all_beads)

    show_all = st.sidebar.checkbox("Show All Beads", value=True)

    bead_input = st.sidebar.text_input(
        "Beads (e.g. 1-3,6,10-12)",
        value=""
    )

    if show_all:
        selected_beads = all_beads
    else:
        selected_beads = parse_bead_input(bead_input, max(all_beads))

    if not selected_beads:
        st.warning("No valid bead selected")
        st.stop()

    st.sidebar.write(f"Selected: {selected_beads}")

    # ========================================================
    # GROUPING
    # ========================================================

    st.sidebar.subheader("Grouping")
    group_size = st.sidebar.number_input("Group size", 1, 50, 1)

    if group_size > 1:
        mode = st.sidebar.radio(
            "Display Mode",
            ["Show Ungrouped Signal", "Show Grouped Signal", "Show All Signal"]
        )
    else:
        mode = "Show All Signal"

    # ========================================================
    # QC + ADJUST
    # ========================================================

    st.sidebar.subheader("QC Method")

    qc_method = st.sidebar.selectbox("Method", [
        "Mean±Std",
        "Median±MAD",
        "Percentile",
        "Z-score"
    ])

    params = {}

    if qc_method in ["Mean±Std", "Median±MAD"]:
        params["k"] = st.sidebar.number_input("k", 0.1, 10.0, 3.0)

    if qc_method == "Percentile":
        params["low"] = st.sidebar.number_input("Low %", 0, 50, 5)
        params["high"] = st.sidebar.number_input("High %", 50, 100, 95)

    if qc_method == "Z-score":
        params["z"] = st.sidebar.number_input("Z", 0.1, 10.0, 3.0)

    st.sidebar.subheader("Adjustment")

    adj_mode = st.sidebar.selectbox("Adjustment", ["None", "Offset"])

    # ========================================================
    # AXIS CONTROL
    # ========================================================

    st.sidebar.subheader("Axis Control")

    y_min = st.sidebar.number_input("Y min", value=0.0)
    y_max = st.sidebar.number_input("Y max", value=0.0)

    use_custom_y = st.sidebar.checkbox("Use custom Y range")

    # ========================================================
    # TABS
    # ========================================================

    tabs = st.tabs(["Channel 1", "Channel 2", "Channel 3"])

    theme = st.get_option("theme.base")
    center_color = "black" if theme == "light" else "white"

    for ch_idx, tab in enumerate(tabs):

        with tab:

            lines = []
            file_names = []

            offset = 0
            bead_boundaries = []

            for bead in selected_beads:

                bead_data = []
                valid_files = []

                for fname in files:
                    if bead in bead_map[fname]:
                        y = bead_map[fname][bead].iloc[:, ch_idx].to_numpy()
                        bead_data.append(y)
                        valid_files.append(fname)

                if not bead_data:
                    continue

                min_len = min(len(x) for x in bead_data)

                bead_data = [x[:min_len] for x in bead_data]

                for i, y in enumerate(bead_data):
                    x = np.arange(min_len) + offset
                    lines.append((valid_files[i], x, y))

                offset += min_len
                bead_boundaries.append(offset)

            # Convert to array per file
            file_dict = {}
            for fname, x, y in lines:
                if fname not in file_dict:
                    file_dict[fname] = []
                file_dict[fname].extend(y.tolist())

            arrs = np.array(list(file_dict.values()))
            names_list = list(file_dict.keys())

            # GROUP
            groups = []
            for i in range(0, len(arrs), group_size):
                groups.append(arrs[i:i+group_size])

            centers = [compute_center(g, "Median") for g in groups]

            # BEFORE
            fig1 = go.Figure()

            # Center + CL FIRST (legend order)
            ucl, lcl = compute_cl(arrs, qc_method, params)

            if ucl is not None:
                fig1.add_trace(go.Scatter(y=ucl,
                    name="UCL",
                    line=dict(color="#00FF00", dash="dash", width=2)))
                fig1.add_trace(go.Scatter(y=lcl,
                    name="LCL",
                    line=dict(color="#00FF00", dash="dash", width=2)))

            for c in centers:
                fig1.add_trace(go.Scatter(y=c,
                    name="Center",
                    line=dict(color=center_color, width=3)))

            # raw
            for i, (fname, _, _) in enumerate(lines):
                if mode in ["Show Ungrouped Signal", "Show All Signal"]:
                    fig1.add_trace(go.Scatter(
                        y=arrs[i],
                        name=fname[:6] if group_size==1 else f"Group {i//group_size+1}",
                        line=dict(width=1),
                        opacity=0.25
                    ))

            # bead separators
            for b in bead_boundaries:
                fig1.add_vline(x=b, line=dict(color="yellow", dash="dot"), opacity=0.25)

            fig1.update_layout(
                title="Before Adjustment",
                legend=dict(x=1.02, y=1),
            )

            if use_custom_y:
                fig1.update_yaxes(range=[y_min, y_max])

            st.plotly_chart(fig1, use_container_width=True)

            # AFTER
            if adj_mode == "Offset":
                arrs2 = adjust_offset(arrs)
            else:
                arrs2 = arrs.copy()

            fig2 = go.Figure()

            ucl2, lcl2 = compute_cl(arrs2, qc_method, params)

            if ucl2 is not None:
                fig2.add_trace(go.Scatter(y=ucl2,
                    name="UCL",
                    line=dict(color="#00FF00", dash="dash", width=2)))
                fig2.add_trace(go.Scatter(y=lcl2,
                    name="LCL",
                    line=dict(color="#00FF00", dash="dash", width=2)))

            for c in centers:
                fig2.add_trace(go.Scatter(y=c,
                    name="Center",
                    line=dict(color=center_color, width=3)))

            for i in range(len(arrs2)):
                fig2.add_trace(go.Scatter(
                    y=arrs2[i],
                    name=names_list[i][:6],
                    line=dict(width=1),
                    opacity=0.25
                ))

            for b in bead_boundaries:
                fig2.add_vline(x=b, line=dict(color="yellow", dash="dot"), opacity=0.25)

            fig2.update_layout(
                title="After Adjustment",
                legend=dict(x=1.02, y=1),
            )

            if use_custom_y:
                fig2.update_yaxes(range=[y_min, y_max])

            st.plotly_chart(fig2, use_container_width=True)
