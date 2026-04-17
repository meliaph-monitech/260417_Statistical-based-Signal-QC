import streamlit as st
import pandas as pd
import numpy as np
import zipfile
import io
import plotly.graph_objects as go

st.set_page_config(layout="wide")

# ============================================================
# -------------------- CORE UTILITIES -------------------------
# ============================================================

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
    return data


def extract_beads(data, seg_col, threshold):
    bead_map = {}
    for fname, df in data.items():
        segs = segment_beads(df, seg_col, threshold)
        bead_map[fname] = {}
        for i, (s, e) in enumerate(segs, start=1):
            bead_map[fname][i] = df.iloc[s:e+1].reset_index(drop=True)
    return bead_map


def align_beads(bead_map, bead_num, ch_idx):
    arrs = []
    names = []
    for fname, beads in bead_map.items():
        if bead_num in beads:
            y = beads[bead_num].iloc[:, ch_idx].to_numpy()
            arrs.append(y)
            names.append(fname)
    if not arrs:
        return None, None
    min_len = min(len(a) for a in arrs)
    arrs = [a[:min_len] for a in arrs]
    return np.array(arrs), names


def group_data(arrs, names, group_size):
    groups = []
    for i in range(0, len(arrs), group_size):
        groups.append(arrs[i:i+group_size])
    return groups


def compute_center(arr, method):
    if method == "Mean":
        return np.mean(arr, axis=0)
    elif method == "Median":
        return np.median(arr, axis=0)
    elif method == "Trimmed":
        return np.mean(np.sort(arr, axis=0)[1:-1], axis=0)
    return np.mean(arr, axis=0)


def compute_cl(arr, method, params):
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

    elif method == "Z-score (global)":
        m = np.mean(arr, axis=0)
        s = np.std(arr, axis=0)
        z = params["z"]
        return m + z*s, m - z*s

    return None, None


def smooth_signal(y, window):
    return np.convolve(y, np.ones(window)/window, mode="same")


def adjust_offset(arr):
    mean_ref = np.mean(arr[0])
    adj = []
    for a in arr:
        shift = np.mean(a) - mean_ref
        adj.append(a - shift)
    return np.array(adj)


def step_aggregate(arr, step):
    return np.array([np.mean(arr[:, i:i+step], axis=1)
                     for i in range(0, arr.shape[1], step)]).T


# ============================================================
# ---------------------- UI START -----------------------------
# ============================================================

st.title("Welding Signal QC Playground")

# ============================================================
# Upload
# ============================================================

uploaded = st.file_uploader("Upload ZIP", type="zip")

if uploaded:

    if "data" not in st.session_state:
        st.session_state.data = load_zip(uploaded)

    data = st.session_state.data
    st.write(f"Files: {len(data)}")

    # ========================================================
    # Segmentation
    # ========================================================

    st.sidebar.header("Segmentation")
    seg_col = st.sidebar.selectbox("Channel", [0,1,2])
    threshold = st.sidebar.number_input("Threshold", value=0.2)

    if st.sidebar.button("Run Segmentation"):
        st.session_state.bead_map = extract_beads(data, seg_col, threshold)
        st.session_state.segmented = True

    if "segmented" in st.session_state:

        bead_map = st.session_state.bead_map

        # get bead numbers
        bead_set = set()
        for f in bead_map.values():
            bead_set.update(f.keys())
        bead_list = sorted(list(bead_set))

        bead_sel = st.sidebar.selectbox("Bead", bead_list)

        # grouping
        st.sidebar.header("Grouping")
        group_size = st.sidebar.number_input("Group size", 1, 50, 5)

        # center
        center_method = st.sidebar.selectbox("Center",
            ["Mean", "Median", "Trimmed"])

        # smoothing
        st.sidebar.header("Smoothing")
        smooth_on = st.sidebar.checkbox("Enable smoothing")
        window = st.sidebar.slider("Window", 3, 51, 11)

        # QC
        st.sidebar.header("QC Method")
        qc_method = st.sidebar.selectbox("Method", [
            "Mean±Std",
            "Median±MAD",
            "Percentile",
            "Z-score (global)",
            "Z-score (pointwise)",
            "Step-based Norm (M)",
            "Step-based Z (M)"
        ])

        params = {}

        if qc_method in ["Mean±Std", "Median±MAD"]:
            params["k"] = st.sidebar.number_input("k", 0.1, 10.0, 3.0)

        if qc_method == "Percentile":
            params["low"] = st.sidebar.number_input("Low %", 0, 50, 5)
            params["high"] = st.sidebar.number_input("High %", 50, 100, 95)

        if "Z-score" in qc_method:
            params["z"] = st.sidebar.number_input("Z", 0.1, 10.0, 3.0)

        if "Step-based" in qc_method:
            params["step"] = st.sidebar.slider("Step", 5, 200, 20)
            params["low"] = st.sidebar.number_input("Low", -5.0, 5.0, -3.0)
            params["high"] = st.sidebar.number_input("High", -5.0, 10.0, 4.0)

        # adjustment
        st.sidebar.header("Adjustment")
        adj_mode = st.sidebar.selectbox("Adjustment",
            ["None", "Offset"])

        # ====================================================
        # CHANNEL TABS
        # ====================================================

        tabs = st.tabs(["Channel 0", "Channel 1", "Channel 2"])

        for ch_idx, tab in enumerate(tabs):

            with tab:

                arrs, names = align_beads(bead_map, bead_sel, ch_idx)

                if arrs is None:
                    st.warning("No data")
                    continue

                # smoothing
                if smooth_on:
                    arrs = np.array([smooth_signal(a, window) for a in arrs])

                groups = group_data(arrs, names, group_size)

                group_centers = np.array([compute_center(g, center_method)
                                          for g in groups])

                # =======================
                # BEFORE
                # =======================

                fig1 = go.Figure()

                for g in groups:
                    for line in g:
                        fig1.add_trace(go.Scatter(y=line,
                            line=dict(width=1), opacity=0.2))

                for c in group_centers:
                    fig1.add_trace(go.Scatter(y=c,
                        line=dict(width=3)))

                ucl, lcl = compute_cl(arrs, qc_method, params)

                if ucl is not None:
                    fig1.add_trace(go.Scatter(y=ucl,
                        line=dict(color="red", dash="dash")))
                    fig1.add_trace(go.Scatter(y=lcl,
                        line=dict(color="red", dash="dash")))

                fig1.update_layout(title="Before Adjustment")
                st.plotly_chart(fig1, use_container_width=True)

                # =======================
                # AFTER
                # =======================

                if adj_mode == "Offset":
                    arrs_adj = adjust_offset(arrs)
                else:
                    arrs_adj = arrs.copy()

                groups_adj = group_data(arrs_adj, names, group_size)

                group_centers_adj = np.array(
                    [compute_center(g, center_method)
                     for g in groups_adj])

                fig2 = go.Figure()

                for g in groups_adj:
                    for line in g:
                        fig2.add_trace(go.Scatter(y=line,
                            line=dict(width=1), opacity=0.2))

                for c in group_centers_adj:
                    fig2.add_trace(go.Scatter(y=c,
                        line=dict(width=3)))

                ucl2, lcl2 = compute_cl(arrs_adj, qc_method, params)

                if ucl2 is not None:
                    fig2.add_trace(go.Scatter(y=ucl2,
                        line=dict(color="red", dash="dash")))
                    fig2.add_trace(go.Scatter(y=lcl2,
                        line=dict(color="red", dash="dash")))

                fig2.update_layout(title="After Adjustment")
                st.plotly_chart(fig2, use_container_width=True)

        # ====================================================
        # TAB 4: DIAGNOSTICS
        # ====================================================

        st.header("Diagnostics")

        lengths = []
        for fname, beads in bead_map.items():
            for b, df in beads.items():
                lengths.append({
                    "file": fname,
                    "bead": b,
                    "length": len(df)
                })

        df_len = pd.DataFrame(lengths)
        pivot = df_len.pivot(index="file", columns="bead", values="length")

        st.subheader("Bead Length Heatmap")
        st.dataframe(pivot)
