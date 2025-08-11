import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
from sklearn.metrics.pairwise import euclidean_distances
import shap

# -----------------------------
# Movie Theater Theme Styling
# -----------------------------
st.markdown("""
    <style>
        body { background-color: #1c1c1c; color: #f5f5f5; }
        .stApp {
            background-image: linear-gradient(to bottom, #2d0c0c, #000000);
            color: #f5f5f5;
            font-family: 'Trebuchet MS', sans-serif;
        }
        h1 { text-align: center; color: #FFD700; font-size: 3em; margin-bottom: 0.2em; }
        h2, h3, .stSubheader { color: #ffcc00; }
        .stMetric { background-color: #2a2a2a !important; border: 1px solid #ffcc00;
                    border-radius: 10px; padding: 10px; color: #ffffff; }
        section[data-testid="stSidebar"] { background-color: #3d0f0f; }
        button[kind="primary"] { background-color: #ff4444 !important; color: white !important;
                                 border-radius: 12px; font-weight: bold; }
        .stDataFrame { border: 1px solid #FFD700; border-radius: 5px; }
        .stFileUploader { background-color: #1e1e1e; border: 2px dashed #ffcc00; padding: 10px; }
        .small-note { font-size: 0.9rem; opacity: 0.8; }
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# Load models & artifacts
# -----------------------------
@st.cache_resource
def load_model(file_path):
    return joblib.load(file_path)

rf_model = load_model("best_random_forest_class3.pkl")
xgb_model = load_model("best_xgboost_class3.pkl")

@st.cache_data
def load_distributor_freq_map():
    with open("distributor_freq_map.pkl", "rb") as f:
        return pickle.load(f)
distributor_freq_map = load_distributor_freq_map()

# -----------------------------
# Helpers
# -----------------------------
def align_and_average_probs(model_a, model_b, X):
    """
    Align per-class predict_proba across two models and average them.
    Returns (dfa, dfb, avg_df, ensemble_bins)
    """
    pa, ca = model_a.predict_proba(X), model_a.classes_
    pb, cb = model_b.predict_proba(X), model_b.classes_
    dfa = pd.DataFrame(pa, columns=ca)
    dfb = pd.DataFrame(pb, columns=cb)
    all_classes = sorted(set(dfa.columns).union(set(dfb.columns)))
    dfa = dfa.reindex(columns=all_classes, fill_value=0)
    dfb = dfb.reindex(columns=all_classes, fill_value=0)
    avg_df = (dfa + dfb) / 2.0
    ensemble_bins = avg_df.idxmax(axis=1).astype(int).to_numpy()
    return dfa, dfb, avg_df, ensemble_bins

def preprocess_uploaded_df(df_raw, distributor_freq_map, feature_names):
    df = df_raw.copy()
    df['User Rating Count'] = df['User Rating Count'].replace(',', '', regex=True).astype(float)

    # Release date / days since
    df['release_date'] = pd.to_datetime(df['Title US Release date'], format='%m/%d/%Y', errors='coerce')
    df['release_missing'] = df['release_date'].isna().astype(int)
    df['days_since_release'] = (pd.to_datetime("today") - df['release_date']).dt.days
    df = df.drop(columns=['Title US Release date', 'release_date'], errors='ignore')

    # Distributor frequency
    df['Distributor_Name_Freq'] = df['Sony Title distributor name (per IMDB)'].map(distributor_freq_map).fillna(0)

    # IP type (single-select)
    ip_types = ['Full Episode', 'Full Pitch', 'Full Feature']
    for ip in ip_types:
        df[f'IP_{ip}'] = (df['IP_Type'] == ip).astype(int)

    # Title type (single-select)
    title_types = ['TV series', 'feature', 'TV movie', 'video', 'videoÅ¡documentary', 'TV mini-series', 'short']
    for title in title_types:
        df[f'Title_{title}'] = (df['Title Type'] == title).astype(int)

    # Genre one-hots (multi-label text)
    genre_set = [
        'Action', 'Adventure', 'Animation', 'Biography', 'Comedy', 'Crime', 'Documentary',
        'Drama', 'Family', 'Fantasy', 'Film-Noir', 'Game-Show', 'History', 'Horror',
        'Music', 'Musical', 'Mystery', 'Reality-TV', 'Romance', 'Sci-Fi', 'Short',
        'Sport', 'Thriller', 'War', 'Western'
    ]
    for genre in genre_set:
        df[genre] = df['Title Genre'].fillna('').str.contains(genre, case=False).astype(int)

    # Numeric types
    df['video_duration_sec'] = df['video_duration_sec'].astype(float)
    df['User Rating'] = df['User Rating'].astype(float)

    # Ensure all features exist
    for col in feature_names:
        if col not in df:
            df[col] = 0

    return df[feature_names].copy()

def compute_shap_topk_for_class(model, X, class_index=3, topk=10):
    """
    Compute SHAP values for a given class (index) for tree models.
    Returns:
      shap_df: DataFrame with columns ['feature','value','shap'] sorted by |shap|
               for the first row in X (single explanation focus).
      shap_arrays: The raw SHAP arrays (may be list per class or 3D array).
      base_value: Expected value for the selected class (if available).
    """
    # TreeExplainer for tree-based models
    try:
        explainer = shap.TreeExplainer(model, model_output="probability", feature_perturbation="tree_path_dependent")
    except Exception:
        # Fallback (some RF versions): omit args
        explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X)  # could be list (per class) or array
    base_value = None

    # Determine per-class SHAP matrix for the requested class
    if isinstance(shap_values, list):
        # List[ndarray (n_samples, n_features)] — one per class
        if class_index >= len(shap_values):
            # if class missing, return zeros
            vals = np.zeros((X.shape[0], X.shape[1]))
            base_value = getattr(explainer, "expected_value", [0]*len(shap_values))
            base_value = base_value[class_index] if class_index < len(base_value) else 0
        else:
            vals = shap_values[class_index]
            # expected_value is list-like per class
            try:
                base_value = explainer.expected_value[class_index]
            except Exception:
                base_value = None
    else:
        # ndarray (n_samples, n_features, n_classes) or (n_samples, n_features)
        if shap_values.ndim == 3:
            if class_index >= shap_values.shape[2]:
                vals = np.zeros((X.shape[0], X.shape[1]))
            else:
                vals = shap_values[:, :, class_index]
            base_value = getattr(explainer, "expected_value", None)
            if isinstance(base_value, (list, np.ndarray)) and len(np.shape(base_value)) > 0:
                try:
                    base_value = base_value[class_index]
                except Exception:
                    pass
        else:
            vals = shap_values  # treat as single output
            base_value = getattr(explainer, "expected_value", None)

    # Build a top-k table for the FIRST row (single prediction detail)
    row_vals = vals[0]
    row_abs = np.abs(row_vals)
    top_idx = np.argsort(-row_abs)[:topk]
    shap_df = pd.DataFrame({
        "feature": X.columns[top_idx],
        "value": X.iloc[0, top_idx].values,
        "shap": row_vals[top_idx]
    }).sort_values("shap", key=lambda s: np.abs(s), ascending=False).reset_index(drop=True)

    return shap_df, shap_values, base_value

# -----------------------------
# Feature list (must match training)
# -----------------------------
feature_names = [
    'video_duration_sec', 'User Rating', 'User Rating Count', 'days_since_release',
    'Action', 'Adventure', 'Animation', 'Biography', 'Comedy', 'Crime', 'Documentary',
    'Drama', 'Family', 'Fantasy', 'Film-Noir', 'Game-Show', 'History', 'Horror',
    'Music', 'Musical', 'Mystery', 'Reality-TV', 'Romance', 'Sci-Fi', 'Short',
    'Sport', 'Thriller', 'War', 'Western', 'Distributor_Name_Freq', 'IP_Full Episode',
    'IP_Full Feature', 'IP_Full Pitch', 'Title_TV mini-series', 'Title_TV movie',
    'Title_TV series', 'Title_feature', 'Title_short', 'Title_video', 'Title_videoÅ¡documentary'
]

# -----------------------------
# Sidebar — inputs & toggles
# -----------------------------
st.sidebar.header("Numerical Inputs")
input_data = {
    'video_duration_sec': st.sidebar.number_input("Video Duration (sec)", value=300),
    'User Rating': st.sidebar.slider("User Rating", 0.0, 10.0, 7.0),
    'User Rating Count': st.sidebar.number_input("User Rating Count", value=100),
    'days_since_release': st.sidebar.number_input("Days Since Release", value=100),
}

st.sidebar.subheader("Decision Threshold")
hv_threshold = st.sidebar.slider("High-value flag (Avg Bin 4 %)", 0, 100, 60, 1)

st.sidebar.subheader("Distributor")
distributor_list = sorted(distributor_freq_map.keys())
selected_distributor = st.sidebar.selectbox("Select Distributor", distributor_list)
input_data['Distributor_Name_Freq'] = distributor_freq_map.get(selected_distributor, 0)

st.sidebar.subheader("IP Type")
ip_options = ['Full Episode', 'Full Pitch', 'Full Feature']
selected_ip = st.sidebar.selectbox("Select IP Type", ip_options)
for ip in ip_options:
    input_data[f"IP_{ip}"] = int(ip == selected_ip)

st.sidebar.subheader("Title Type")
title_options = ['TV series', 'feature', 'TV movie', 'video', 'videoÅ¡documentary', 'TV mini-series', 'short']
selected_title = st.sidebar.selectbox("Select Title Type", title_options)
for title in title_options:
    input_data[f"Title_{title}"] = int(title == selected_title)

st.sidebar.subheader("Genres")
binary_flags = [
    'Action', 'Adventure', 'Animation', 'Biography', 'Comedy', 'Crime', 'Documentary',
    'Drama', 'Family', 'Fantasy', 'Film-Noir', 'Game-Show', 'History', 'Horror',
    'Music', 'Musical', 'Mystery', 'Reality-TV', 'Romance', 'Sci-Fi', 'Short',
    'Sport', 'Thriller', 'War', 'Western'
]
for flag in binary_flags:
    input_data[flag] = int(st.sidebar.checkbox(flag, value=False))

# Fill any missing features and order them
for col in feature_names:
    if col not in input_data:
        input_data[col] = 0
X_input = pd.DataFrame([input_data])[feature_names]

# -----------------------------
# Header & nav
# -----------------------------
st.title("🎬 High Value Content Classifer")
st.page_link("pages/arima_dashboard.py", label="📈 ARIMA Dashboard", icon="📘")
st.page_link("pages/Sentiment_app.py", label="🗣️ Sentiment Analysis", icon="💬")
# st.page_link("pages/Survival_Analysis_Sony.py", label="⏳ Survival Analysis", icon="⏳")

# -----------------------------
# Single prediction
# -----------------------------
if st.button("Predict High Value Content Likelihood"):
    # Per-model hard bins
    rf_bin = int(rf_model.predict(X_input)[0])
    xgb_bin = int(xgb_model.predict(X_input)[0])

    # Align & average per-class probabilities → ensemble
    rf_df, xgb_df, avg_df, ens_bins = align_and_average_probs(rf_model, xgb_model, X_input)
    ens_bin = int(ens_bins[0])

    # Bin 4 (class=3) probabilities (%), class-safe get()
    rf_proba_bin4  = float(rf_df.get(3, pd.Series([0])).iloc[0]) * 100
    xgb_proba_bin4 = float(xgb_df.get(3, pd.Series([0])).iloc[0]) * 100
    avg_proba_bin4 = float(avg_df.get(3, pd.Series([0])).iloc[0]) * 100

    st.subheader("Prediction Results")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("RF Bin‑4 %", f"{rf_proba_bin4:.2f}%")
        st.metric("RF Pred. Bin", str(rf_bin))
    with c2:
        st.metric("XGB Bin‑4 %", f"{xgb_proba_bin4:.2f}%")
        st.metric("XGB Pred. Bin", str(xgb_bin))
    with c3:
        st.metric("Ensemble Bin‑4 %", f"{avg_proba_bin4:.2f}%")
        st.metric("Ensemble Pred. Bin", str(ens_bin))
    with c4:
        hv = (avg_proba_bin4 >= hv_threshold) or (ens_bin == 3)
        st.metric("Decision", "High Value ✅" if hv else "Not High Value ❌")

    # --- SHAP explanations (top‑k) for Bin 4 (class=3)
    st.markdown("### 🔍 SHAP Explanation (Bin 4 probability)")
    topk = st.slider("Top features to show", 3, 20, 10, 1)
    with st.spinner("Computing SHAP (RF)…"):
        rf_shap_df, _, rf_base = compute_shap_topk_for_class(rf_model, X_input, class_index=3, topk=topk)
    with st.spinner("Computing SHAP (XGB)…"):
        xgb_shap_df, _, xgb_base = compute_shap_topk_for_class(xgb_model, X_input, class_index=3, topk=topk)

    st.write("**Random Forest — Top contributors**")
    st.dataframe(rf_shap_df)
    st.caption(f"RF base value (expected Bin‑4 prob): {rf_base:.4f}" if rf_base is not None else "RF base value unavailable.")

    st.write("**XGBoost — Top contributors**")
    st.dataframe(xgb_shap_df)
    st.caption(f"XGB base value (expected Bin‑4 prob): {xgb_base:.4f}" if xgb_base is not None else "XGB base value unavailable.")

# -----------------------------
# Batch prediction
# -----------------------------
st.markdown("---")
st.header("📄 Batch Prediction from File")

with st.expander("Try Example Data Instead"):
    example_path = "example_data.csv"
    try:
        df_example = pd.read_csv(example_path)
        st.subheader("📋 Example CSV Data")
        st.dataframe(df_example.head())

        df_ex_t = preprocess_uploaded_df(df_example, distributor_freq_map, feature_names)

        # Ensemble path
        rf_df_ex, xgb_df_ex, avg_df_ex, ens_bins_ex = align_and_average_probs(rf_model, xgb_model, df_ex_t)
        n_ex = len(df_ex_t)
        rf_probs_ex  = (rf_df_ex.get(3, pd.Series(np.zeros(n_ex))) * 100).to_numpy()
        xgb_probs_ex = (xgb_df_ex.get(3, pd.Series(np.zeros(n_ex))) * 100).to_numpy()
        avg_probs_ex = (avg_df_ex.get(3, pd.Series(np.zeros(n_ex))) * 100).to_numpy()
        rf_bins_ex   = rf_model.predict(df_ex_t).astype(int)
        xgb_bins_ex  = xgb_model.predict(df_ex_t).astype(int)

        df_example['RF_Prob_Bin4 (%)']  = rf_probs_ex
        df_example['XGB_Prob_Bin4 (%)'] = xgb_probs_ex
        df_example['Ensemble_Bin4 (%)'] = avg_probs_ex
        df_example['RF_Pred_Bin']       = rf_bins_ex
        df_example['XGB_Pred_Bin']      = xgb_bins_ex
        df_example['Ensemble_Pred_Bin'] = ens_bins_ex
        df_example['High_Value_Flag']   = (df_example['Ensemble_Bin4 (%)'] >= hv_threshold) | (df_example['Ensemble_Pred_Bin'] == 3)

        # Manual centroid cluster (as before)
        cluster_centroids = [
            {'User_Rating': 6.55, 'Genre_Action': 0.99, 'Genre_Adventure': 0.99, 'Genre_Crime': 0.99, 'Genre_Drama': 0.5, 'Genre_Mystery': 0.01, 'Genre_Sci-Fi': 0.01, 'Genre_Thriller': 1},
            {'User_Rating': 6.46, 'Genre_Action': 0.74, 'Genre_Adventure': 0.74, 'Genre_Crime': 0.74, 'Genre_Drama': 0.63, 'Genre_Mystery': 0.26, 'Genre_Sci-Fi': 0.26, 'Genre_Thriller': 1},
            {'User_Rating': 6.54, 'Genre_Action': 0.97, 'Genre_Adventure': 0.97, 'Genre_Crime': 0.97, 'Genre_Drama': 0.52, 'Genre_Mystery': 0.03, 'Genre_Sci-Fi': 0.03, 'Genre_Thriller': 1},
            {'User_Rating': 6.2,  'Genre_Action': 0,    'Genre_Adventure': 0,    'Genre_Crime': 0,    'Genre_Drama': 1,  'Genre_Mystery': 1,   'Genre_Sci-Fi': 1,   'Genre_Thriller': 1},
        ]
        manual_features = ['User Rating', 'Action', 'Adventure', 'Crime', 'Drama', 'Mystery', 'Sci-Fi', 'Thriller']
        df_cluster_in = df_ex_t[manual_features].copy()
        df_cluster_in.columns = [f"Genre_{c}" if c != 'User Rating' else c for c in manual_features]
        centroids_df_ex = pd.DataFrame(cluster_centroids)
        distances_ex = euclidean_distances(df_cluster_in, centroids_df_ex)
        df_example['Manual Cluster'] = np.argmin(distances_ex, axis=1)

        st.subheader("Predictions for Example Data")
        st.dataframe(df_example.head())
    except FileNotFoundError:
        pass

uploaded_file = st.file_uploader("Upload CSV or Excel file", type=['csv', 'xlsx'])

if uploaded_file:
    df_input = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
    st.subheader("Preview of Uploaded Data")
    st.dataframe(df_input.head())

    df_t = preprocess_uploaded_df(df_input, distributor_freq_map, feature_names)

    # Ensemble probabilities
    rf_df, xgb_df, avg_df, ensemble_bins = align_and_average_probs(rf_model, xgb_model, df_t)
    n = len(df_t)
    rf_probs  = (rf_df.get(3, pd.Series(np.zeros(n))) * 100).to_numpy()
    xgb_probs = (xgb_df.get(3, pd.Series(np.zeros(n))) * 100).to_numpy()
    avg_probs = (avg_df.get(3, pd.Series(np.zeros(n))) * 100).to_numpy()

    rf_bins  = rf_model.predict(df_t).astype(int)
    xgb_bins = xgb_model.predict(df_t).astype(int)

    # Append outputs
    df_input['RF_Prob_Bin4 (%)']  = rf_probs
    df_input['XGB_Prob_Bin4 (%)'] = xgb_probs
    df_input['Ensemble_Bin4 (%)'] = avg_probs
    df_input['RF_Pred_Bin']       = rf_bins
    df_input['XGB_Pred_Bin']      = xgb_bins
    df_input['Ensemble_Pred_Bin'] = ensemble_bins
    df_input['High_Value_Flag']   = (df_input['Ensemble_Bin4 (%)'] >= hv_threshold) | (df_input['Ensemble_Pred_Bin'] == 3)

    # Optional batch SHAP (fast cap)
    st.markdown("### 🔍 Batch SHAP (optional)")
    do_batch_shap = st.checkbox("Compute SHAP for first N rows (Bin 4 probability)", value=False)
    if do_batch_shap:
        N = st.number_input("Rows to explain (top‑N from the uploaded file)", min_value=1, max_value=int(min(200, len(df_t))), value=int(min(25, len(df_t))))
        topk = st.slider("Top features per row", 3, 15, 8, 1, key="batch_topk")
        with st.spinner("Computing batch SHAP (RF)…"):
            rf_shap_df1, rf_shap_vals1, _ = compute_shap_topk_for_class(rf_model, df_t.iloc[:1], class_index=3, topk=topk)  # warm-up
            try:
                rf_expl = shap.TreeExplainer(rf_model, model_output="probability", feature_perturbation="tree_path_dependent")
            except Exception:
                rf_expl = shap.TreeExplainer(rf_model)
            rf_vals = rf_expl.shap_values(df_t.iloc[:N])
            # Get class slice
            rf_vals_cls = rf_vals[3] if isinstance(rf_vals, list) else (rf_vals[:, :, 3] if rf_vals.ndim == 3 else rf_vals)

        with st.spinner("Computing batch SHAP (XGB)…"):
            xgb_shap_df1, xgb_shap_vals1, _ = compute_shap_topk_for_class(xgb_model, df_t.iloc[:1], class_index=3, topk=topk)  # warm-up
            try:
                xgb_expl = shap.TreeExplainer(xgb_model, model_output="probability", feature_perturbation="tree_path_dependent")
            except Exception:
                xgb_expl = shap.TreeExplainer(xgb_model)
            xgb_vals = xgb_expl.shap_values(df_t.iloc[:N])
            xgb_vals_cls = xgb_vals[3] if isinstance(xgb_vals, list) else (xgb_vals[:, :, 3] if xgb_vals.ndim == 3 else xgb_vals)

        # Build concise “Top_Features” string per row (feature:shap)
        feats = np.array(df_t.columns)
        top_strings = []
        for i in range(min(N, len(df_t))):
            # average SHAP across models for ranking
            row_rf = rf_vals_cls[i]
            row_xgb = xgb_vals_cls[i]
            row_avg = (row_rf + row_xgb) / 2.0
            order = np.argsort(-np.abs(row_avg))[:topk]
            s = "; ".join([f"{feats[j]}:{row_avg[j]:+.3f}" for j in order])
            top_strings.append(s)

        df_input.loc[:N-1, "Top_Features(AvgSHAP)"] = top_strings
        st.caption("Top features are ranked by average |SHAP| across RF+XGB for Bin‑4 probability.")

    # Manual centroid cluster assignment (as before)
    cluster_centroids = [
        {'User_Rating': 6.55, 'Genre_Action': 0.99, 'Genre_Adventure': 0.99, 'Genre_Crime': 0.99, 'Genre_Drama': 0.5, 'Genre_Mystery': 0.01, 'Genre_Sci-Fi': 0.01, 'Genre_Thriller': 1},
        {'User_Rating': 6.46, 'Genre_Action': 0.74, 'Genre_Adventure': 0.74, 'Genre_Crime': 0.74, 'Genre_Drama': 0.63, 'Genre_Mystery': 0.26, 'Genre_Sci-Fi': 0.26, 'Genre_Thriller': 1},
        {'User_Rating': 6.54, 'Genre_Action': 0.97, 'Genre_Adventure': 0.97, 'Genre_Crime': 0.97, 'Genre_Drama': 0.52, 'Genre_Mystery': 0.03, 'Genre_Sci-Fi': 0.03, 'Genre_Thriller': 1},
        {'User_Rating': 6.2,  'Genre_Action': 0,    'Genre_Adventure': 0,    'Genre_Crime': 0,    'Genre_Drama': 1,  'Genre_Mystery': 1,   'Genre_Sci-Fi': 1,   'Genre_Thriller': 1},
    ]
    manual_features = ['User Rating', 'Action', 'Adventure', 'Crime', 'Drama', 'Mystery', 'Sci-Fi', 'Thriller']
    df_cluster_input = df_t[manual_features].copy()
    df_cluster_input.columns = [f"Genre_{col}" if col != 'User Rating' else col for col in manual_features]
    centroids_df = pd.DataFrame(cluster_centroids)
    distances = euclidean_distances(df_cluster_input, centroids_df)
    df_input['Manual Cluster'] = np.argmin(distances, axis=1)

    st.subheader("Prediction Results")
    st.dataframe(df_input.head())

    @st.cache_data
    def convert_df_to_excel(df):
        from io import BytesIO
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Predictions')
        return output.getvalue()

    excel_data = convert_df_to_excel(df_input)
    st.download_button(
        label="📥 Download Predictions as Excel",
        data=excel_data,
        file_name='predicted_content_results.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )
