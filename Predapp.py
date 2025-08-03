import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
from sklearn.metrics.pairwise import euclidean_distances

# --- Movie Theater Theme Styling ---
st.markdown("""
    <style>
        body {
            background-color: #1c1c1c;
            color: #f5f5f5;
        }
        .stApp {
            background-image: linear-gradient(to bottom, #2d0c0c, #000000);
            color: #f5f5f5;
            font-family: 'Trebuchet MS', sans-serif;
        }
        h1 {
            text-align: center;
            color: #FFD700;
            font-size: 3em;
            margin-bottom: 0.2em;
        }
        h2, h3, .stSubheader {
            color: #ffcc00;
        }
        .stMetric {
            background-color: #2a2a2a !important;
            border: 1px solid #ffcc00;
            border-radius: 10px;
            padding: 10px;
            color: #ffffff;
        }
        section[data-testid="stSidebar"] {
            background-color: #3d0f0f;
        }
        button[kind="primary"] {
            background-color: #ff4444 !important;
            color: white !important;
            border-radius: 12px;
            font-weight: bold;
        }
        .stDataFrame {
            border: 1px solid #FFD700;
            border-radius: 5px;
        }
        .stFileUploader {
            background-color: #1e1e1e;
            border: 2px dashed #ffcc00;
            padding: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# --- Load models ---
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

# --- Sheet Transformation ---
def preprocess_uploaded_df(df_raw, distributor_freq_map, feature_names):
    df = df_raw.copy()
    df['User Rating Count'] = df['User Rating Count'].replace(',', '', regex=True).astype(float)
    df['release_date'] = pd.to_datetime(df['Title US Release date'], format='%m/%d/%Y', errors='coerce')
    df['release_missing'] = df['release_date'].isna().astype(int)
    df['days_since_release'] = (pd.to_datetime("today") - df['release_date']).dt.days
    df = df.drop(columns=['Title US Release date', 'release_date'], errors='ignore')
    df['Distributor_Name_Freq'] = df['Sony Title distributor name (per IMDB)'].map(distributor_freq_map).fillna(0)

    ip_types = ['Full Episode', 'Full Pitch', 'Full Feature']
    for ip in ip_types:
        df[f'IP_{ip}'] = (df['IP_Type'] == ip).astype(int)

    title_types = [
        'TV series', 'feature', 'TV movie', 'video', 'videoÅ¡documentary',
        'TV mini-series', 'short'
    ]
    for title in title_types:
        df[f'Title_{title}'] = (df['Title Type'] == title).astype(int)

    genre_set = [
        'Action', 'Adventure', 'Animation', 'Biography', 'Comedy', 'Crime', 'Documentary',
        'Drama', 'Family', 'Fantasy', 'Film-Noir', 'Game-Show', 'History', 'Horror',
        'Music', 'Musical', 'Mystery', 'Reality-TV', 'Romance', 'Sci-Fi', 'Short',
        'Sport', 'Thriller', 'War', 'Western'
    ]
    for genre in genre_set:
        df[genre] = df['Title Genre'].fillna('').str.contains(genre, case=False).astype(int)

    df['video_duration_sec'] = df['video_duration_sec'].astype(float)
    df['User Rating'] = df['User Rating'].astype(float)

    for col in feature_names:
        if col not in df:
            df[col] = 0

    df_final = df[feature_names].copy()
    return df_final

feature_names = [
    'video_duration_sec', 'User Rating', 'User Rating Count', 'days_since_release',
    'Action', 'Adventure', 'Animation', 'Biography', 'Comedy', 'Crime', 'Documentary',
    'Drama', 'Family', 'Fantasy', 'Film-Noir', 'Game-Show', 'History', 'Horror',
    'Music', 'Musical', 'Mystery', 'Reality-TV', 'Romance', 'Sci-Fi', 'Short',
    'Sport', 'Thriller', 'War', 'Western', 'Distributor_Name_Freq', 'IP_Full Episode',
    'IP_Full Feature', 'IP_Full Pitch', 'Title_TV mini-series', 'Title_TV movie',
    'Title_TV series', 'Title_feature', 'Title_short', 'Title_video', 'Title_videoÅ¡documentary'
]

st.sidebar.header("Numerical Inputs")
input_data = {
    'video_duration_sec': st.sidebar.number_input("Video Duration (sec)", value=300),
    'User Rating': st.sidebar.slider("User Rating", 0.0, 10.0, 7.0),
    'User Rating Count': st.sidebar.number_input("User Rating Count", value=100),
    'days_since_release': st.sidebar.number_input("Days Since Release", value=100),
}

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

for col in feature_names:
    if col not in input_data:
        input_data[col] = 0

X_input = pd.DataFrame([input_data])[feature_names]

st.title("\U0001F3AC High Value Content Classifer")
st.page_link("pages/arima_dashboard.py", label="\U0001F4C8 ARIMA Dashboard", icon="\U0001F4D8")

if st.button("Predict High Value Content Likelihood"):
    rf_proba = rf_model.predict_proba(X_input)[0][3] * 100
    xgb_proba = xgb_model.predict_proba(X_input)[0][3] * 100
    avg_proba = (rf_proba + xgb_proba) / 2

    st.subheader("Prediction Results")
    st.metric("Random Forest: Bin 4 Probability", f"{rf_proba:.2f}%")
    st.metric("XGBoost: Bin 4 Probability", f"{xgb_proba:.2f}%")
    st.metric("Average Bin 4 Probability", f"{avg_proba:.2f}%")
    st.success("Prediction completed successfully.")

st.markdown("---")
st.header("\U0001F4C4 Batch Prediction from File")

uploaded_file = st.file_uploader("Upload CSV or Excel file", type=['csv', 'xlsx'])

if uploaded_file:
    df_input = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
    st.subheader("Preview of Uploaded Data")
    st.dataframe(df_input.head())

    df_input_transformed = preprocess_uploaded_df(df_input, distributor_freq_map, feature_names)
    rf_probs = rf_model.predict_proba(df_input_transformed)[:, 3] * 100
    xgb_probs = xgb_model.predict_proba(df_input_transformed)[:, 3] * 100
    avg_probs = (rf_probs + xgb_probs) / 2

    df_input['RF_Prob_Bin4 (%)'] = rf_probs
    df_input['XGB_Prob_Bin4 (%)'] = xgb_probs
    df_input['Avg_Prob_Bin4 (%)'] = avg_probs

    # Manual cluster assignment
    cluster_centroids = [
        {'User_Rating': 6.55, 'Genre_Action': 0.99, 'Genre_Adventure': 0.99, 'Genre_Crime': 0.99, 'Genre_Drama': 0.5, 'Genre_Mystery': 0.01, 'Genre_Sci-Fi': 0.01, 'Genre_Thriller': 1},
        {'User_Rating': 6.46, 'Genre_Action': 0.74, 'Genre_Adventure': 0.74, 'Genre_Crime': 0.74, 'Genre_Drama': 0.63, 'Genre_Mystery': 0.26, 'Genre_Sci-Fi': 0.26, 'Genre_Thriller': 1},
        {'User_Rating': 6.54, 'Genre_Action': 0.97, 'Genre_Adventure': 0.97, 'Genre_Crime': 0.97, 'Genre_Drama': 0.52, 'Genre_Mystery': 0.03, 'Genre_Sci-Fi': 0.03, 'Genre_Thriller': 1},
        {'User_Rating': 6.2, 'Genre_Action': 0, 'Genre_Adventure': 0, 'Genre_Crime': 0, 'Genre_Drama': 1, 'Genre_Mystery': 1, 'Genre_Sci-Fi': 1, 'Genre_Thriller': 1},
    ]

    manual_features = ['User Rating', 'Action', 'Adventure', 'Crime', 'Drama', 'Mystery', 'Sci-Fi', 'Thriller']
    df_cluster_input = df_input_transformed[manual_features].copy()
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
        label="\U0001F4E5 Download Predictions as Excel",
        data=excel_data,
        file_name='predicted_content_results.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )
