import streamlit as st

# --- Movie Theater Theme Styling ---
st.markdown("""
    <style>
        /* Background & main text styling */
        body {
            background-color: #1c1c1c;
            color: #f5f5f5;
        }
        .stApp {
            background-image: linear-gradient(to bottom, #2d0c0c, #000000);
            color: #f5f5f5;
            font-family: 'Trebuchet MS', sans-serif;
        }

        /* Title customization */
        h1 {
            text-align: center;
            color: #FFD700; /* Gold */
            font-size: 3em;
            margin-bottom: 0.2em;
        }

        h2, h3, .stSubheader {
            color: #ffcc00;
        }

        /* Metric styling */
        .stMetric {
            background-color: #2a2a2a !important;
            border: 1px solid #ffcc00;
            border-radius: 10px;
            padding: 10px;
            color: #ffffff;
        }

        /* Sidebar styling */
        section[data-testid="stSidebar"] {
            background-color: #3d0f0f;
        }

        /* Buttons */
        button[kind="primary"] {
            background-color: #ff4444 !important;
            color: white !important;
            border-radius: 12px;
            font-weight: bold;
        }

        /* Dataframe styling */
        .stDataFrame {
            border: 1px solid #FFD700;
            border-radius: 5px;
        }

        /* File uploader */
        .stFileUploader {
            background-color: #1e1e1e;
            border: 2px dashed #ffcc00;
            padding: 10px;
        }
    </style>
""", unsafe_allow_html=True)

import pandas as pd
import joblib
import pickle

# --- Load models ---
@st.cache_resource
def load_model(file_path):
    return joblib.load(file_path)

rf_model = load_model("best_random_forest_class3.pkl")
xgb_model = load_model("best_xgboost_class3.pkl")

# --- Load distributor frequency encoding ---
@st.cache_data
def load_distributor_freq_map():
    with open("distributor_freq_map.pkl", "rb") as f:
        return pickle.load(f)

distributor_freq_map = load_distributor_freq_map()

# --- Load clustering model ---
@st.cache_resource
def load_cluster_model():
    return joblib.load("cluster_pipeline.pkl")

cluster_pipeline = load_cluster_model()

# --- Sheet Transformation ---
def preprocess_uploaded_df(df_raw, distributor_freq_map, feature_names):
    df = df_raw.copy()

    # --- Clean & Convert ---
    df['User Rating Count'] = df['User Rating Count'].replace(',', '', regex=True).astype(float)

    # Days since release
    df['release_date'] = pd.to_datetime(df['Title US Release date'], format='%m/%d/%Y', errors='coerce')
    df['release_missing'] = df['release_date'].isna().astype(int)
    df['days_since_release'] = (pd.to_datetime("today") - df['release_date']).dt.days

    df = df.drop(columns=['Title US Release date', 'release_date'], errors='ignore')


    # Distributor frequency
    df['Distributor_Name_Freq'] = df['Sony Title distributor name (per IMDB)'].map(distributor_freq_map).fillna(0)

    # --- IP Type One-hot ---
    ip_types = ['Full Episode', 'Full Pitch', 'Full Feature']
    for ip in ip_types:
        df[f'IP_{ip}'] = (df['IP_Type'] == ip).astype(int)

    # --- Title Type One-hot ---
    title_types = [
        'TV series', 'feature', 'TV movie', 'video', 'videoÅ¡documentary',
        'TV mini-series', 'short'
    ]
    for title in title_types:
        df[f'Title_{title}'] = (df['Title Type'] == title).astype(int)

    # --- Genre one-hot (multi-label) ---
    genre_set = [
        'Action', 'Adventure', 'Animation', 'Biography', 'Comedy',
        'Crime', 'Documentary', 'Drama', 'Family', 'Fantasy', 'Film-Noir',
        'Game-Show', 'History', 'Horror', 'Music', 'Musical', 'Mystery',
        'Reality-TV', 'Romance', 'Sci-Fi', 'Short', 'Sport', 'Thriller',
        'War', 'Western'
    ]
    for genre in genre_set:
        df[genre] = df['Title Genre'].fillna('').str.contains(genre, case=False).astype(int)

    # --- Assemble final input ---
    df['video_duration_sec'] = df['video_duration_sec'].astype(float)
    df['User Rating'] = df['User Rating'].astype(float)

    for col in feature_names:
        if col not in df:
            df[col] = 0

    df_final = df[feature_names].copy()
    return df_final


# --- Feature list (must match training model) ---
feature_names = [
    'video_duration_sec', 'User Rating', 'User Rating Count',
    'days_since_release', 'Action', 'Adventure', 'Animation', 'Biography', 'Comedy',
    'Crime', 'Documentary', 'Drama', 'Family', 'Fantasy', 'Film-Noir', 'Game-Show',
    'History', 'Horror', 'Music', 'Musical', 'Mystery', 'Reality-TV', 'Romance',
    'Sci-Fi', 'Short', 'Sport', 'Thriller', 'War', 'Western', 'Distributor_Name_Freq',
    'IP_Full Episode', 'IP_Full Feature', 'IP_Full Pitch', 'Title_TV mini-series',
    'Title_TV movie', 'Title_TV series', 'Title_feature', 'Title_short', 'Title_video',
    'Title_videoÅ¡documentary'
]

# --- Sidebar Inputs ---
st.sidebar.header("Numerical Inputs")
input_data = {
    'video_duration_sec': st.sidebar.number_input("Video Duration (sec)", value=300),
    'User Rating': st.sidebar.slider("User Rating", 0.0, 10.0, 7.0),
    'User Rating Count': st.sidebar.number_input("User Rating Count", value=100),
    'days_since_release': st.sidebar.number_input("Days Since Release", value=100),
}

# --- Distributor dropdown (searchable) ---
st.sidebar.subheader("Distributor")
distributor_list = sorted(distributor_freq_map.keys())
selected_distributor = st.sidebar.selectbox("Select Distributor", distributor_list)
input_data['Distributor_Name_Freq'] = distributor_freq_map.get(selected_distributor, 0)

# --- IP Type (only one selected) ---
st.sidebar.subheader("IP Type")
ip_options = ['Full Episode', 'Full Pitch', 'Full Feature']
selected_ip = st.sidebar.selectbox("Select IP Type", ip_options)
for ip in ip_options:
    input_data[f"IP_{ip}"] = int(ip == selected_ip)

# --- Title Type (only one selected) ---
st.sidebar.subheader("Title Type")
title_options = [
    'TV series', 'feature', 'TV movie', 'video',
    'videoÅ¡documentary', 'TV mini-series', 'short'
]
selected_title = st.sidebar.selectbox("Select Title Type", title_options)
for title in title_options:
    input_data[f"Title_{title}"] = int(title == selected_title)

# --- Genre and other binary flags ---
st.sidebar.subheader("Genres")
binary_flags = [
    'Action', 'Adventure', 'Animation', 'Biography', 'Comedy',
    'Crime', 'Documentary', 'Drama', 'Family', 'Fantasy', 'Film-Noir',
    'Game-Show', 'History', 'Horror', 'Music', 'Musical', 'Mystery',
    'Reality-TV', 'Romance', 'Sci-Fi', 'Short', 'Sport', 'Thriller',
    'War', 'Western'
]
for flag in binary_flags:
    input_data[flag] = int(st.sidebar.checkbox(flag, value=False))

# --- Fill in any missing feature columns with 0 ---
for col in feature_names:
    if col not in input_data:
        input_data[col] = 0

# --- Ensure correct column order ---
X_input = pd.DataFrame([input_data])[feature_names]

# --- App Title ---
st.title("🎬 High Value Content Classifer")
#st.markdown("<h5 style='text-align: center; color: #f5f5f5;'>Will your content be the next blockbuster?</h5>", unsafe_allow_html=True)
st.write("Navigate to other pages:")
st.page_link("pages/arima_dashboard.py", label="📄 Overview", icon="📘")


# --- Run Prediction ---
if st.button("Predict High Value Content Likelihood"):
    rf_proba = rf_model.predict_proba(X_input)[0][3] * 100
    xgb_proba = xgb_model.predict_proba(X_input)[0][3] * 100
    avg_proba = (rf_proba + xgb_proba) / 2

    st.subheader("Prediction Results")
    st.metric("Random Forest: Bin 4 Probability", f"{rf_proba:.2f}%")
    st.metric("XGBoost: Bin 4 Probability", f"{xgb_proba:.2f}%")
    st.metric("Average Bin 4 Probability", f"{avg_proba:.2f}%")

    st.success("Prediction completed successfully.")

# --- Mass Excel Prediction ---
st.markdown("---")
st.header("📄 Batch Prediction from File")

uploaded_file = st.file_uploader("Upload CSV or Excel file", type=['csv', 'xlsx'])

if uploaded_file:
    # Read uploaded file
    if uploaded_file.name.endswith(".csv"):
        df_input = pd.read_csv(uploaded_file)
    else:
        df_input = pd.read_excel(uploaded_file)

    st.subheader("Preview of Uploaded Data")
    st.dataframe(df_input.head())

    # Preprocess raw input into model-ready format
    df_input_transformed = preprocess_uploaded_df(df_input, distributor_freq_map, feature_names)

    # Predict content bin
    rf_probs = rf_model.predict_proba(df_input_transformed)[:, 3] * 100
    xgb_probs = xgb_model.predict_proba(df_input_transformed)[:, 3] * 100
    avg_probs = (rf_probs + xgb_probs) / 2

    # --- Handle missing values before clustering ---
    fill_value = -1  # Use -1 to indicate "missing" more clearly than 0
    if df_input_transformed.isnull().any().any():
        st.warning(f"Detected missing values. Filling with {fill_value} for clustering.")
        missing_cols = df_input_transformed.columns[df_input_transformed.isnull().any()]
        st.write("⚠️ Missing columns filled:", list(missing_cols))
        df_input_transformed = df_input_transformed.fillna(fill_value)

    # --- Predict cluster for each row ---
    clusters = cluster_pipeline.predict(df_input_transformed)
    


    # Append to uploaded data
    df_input['RF_Prob_Bin4 (%)'] = rf_probs
    df_input['XGB_Prob_Bin4 (%)'] = xgb_probs
    df_input['Avg_Prob_Bin4 (%)'] = avg_probs
    df_input['Predicted Cluster'] = clusters


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
