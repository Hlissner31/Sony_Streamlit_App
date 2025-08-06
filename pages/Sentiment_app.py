import streamlit as st
import pandas as pd
import numpy as np
import joblib
import unicodedata
import re

# ---------- Feature Engineering Function ----------
def clean_and_extract_features(title):
    # Normalize text
    title_clean = unicodedata.normalize('NFKD', title).encode('ascii', 'ignore').decode('utf-8')
    title_clean = title_clean.replace('!', ' ').replace('?', ' ')
    title_clean = title_clean.lower()
    title_clean = re.sub(r'(season \d+|episode \d+|classic tv rewind|pilot)', '', title_clean)
    title_clean = re.sub(r'[^\w\s]', '', title_clean)
    title_clean = re.sub(r'\s+', ' ', title_clean).strip()

    # Features
    sentiment_score = 1.0 if "!" in title else 0.0
    title_length = len(title)
    num_words = len(title.split())
    num_caps_words = sum(word.isupper() for word in title.split())
    num_exclamations = title.count("!")
    has_exclamation = int("!" in title)
    has_question = int("?" in title)
    title_contains_clickbait = int(bool(re.search(r"\b(you won't believe|shocking|revealed|top \d+)\b", title.lower())))

    return pd.DataFrame([{
        'sentiment_score': sentiment_score,
        'title_length': title_length,
        'num_words': num_words,
        'has_exclamation': has_exclamation,
        'num_caps_words': num_caps_words,
        'num_exclamations': num_exclamations,
        'has_question': has_question,
        'title_contains_clickbait': title_contains_clickbait
    }])

# ---------- Load Trained Models ----------
model_views = joblib.load("model_views.pkl")
model_revenue = joblib.load("model_revenue.pkl")

# ---------- Streamlit App ----------
st.set_page_config(page_title="YouTube Title Predictor", page_icon="📺")
st.title("📺 YouTube Title Impact Predictor")
st.markdown("Enter a YouTube video title to estimate how it might perform.")

title_input = st.text_input("Enter Video Title", "")

if title_input:
    features = clean_and_extract_features(title_input)

    # Ensure column order matches training
    expected_cols = ['sentiment_score', 'title_length', 'num_words', 'has_exclamation',
                     'num_caps_words', 'num_exclamations', 'has_question', 'title_contains_clickbait']
    features = features.reindex(columns=expected_cols)

    # Make predictions
    log_views_pred = model_views.predict(features)[0]
    log_revenue_pred = model_revenue.predict(features)[0]
    predicted_views = int(np.exp(log_views_pred))
    predicted_revenue = round(np.exp(log_revenue_pred), 2)

    # Display Results
    st.success("✅ Predictions")
    st.metric(label="Estimated Views", value=f"{predicted_views:,}")
    st.metric(label="Estimated Net Revenue ($)", value=f"${predicted_revenue}")

    # Interpretive Feedback
    st.markdown("### 🎯 Interpretive Feedback")
    if predicted_views > 10000:
        st.info("This title is likely to perform **very well** based on historical patterns.")
    elif predicted_views > 1000:
        st.info("This title has **moderate performance potential**.")
    else:
        st.warning("This title may have **low performance** unless paired with strong content.")

    if features['sentiment_score'][0] < 0.5:
        st.markdown("🔹 Consider making the title more exciting (e.g., using `!` or strong emotional words).")
    if features['num_caps_words'][0] > 4:
        st.markdown("⚠️ Too many capitalized words may look like clickbait.")
    if features['title_contains_clickbait'][0]:
        st.markdown("⚠️ This title has clickbait-like phrasing. Use carefully to avoid audience mistrust.")

    # Feature Importance (Top Drivers)
    st.markdown("### 🧠 Top Drivers of Predicted Revenue")
    try:
        feature_importances = model_revenue.feature_importances_
        importance_df = pd.DataFrame({
            'Feature': expected_cols,
            'Importance': feature_importances
        }).sort_values(by="Importance", ascending=False)

        st.bar_chart(importance_df.set_index("Feature"))
    except:
        st.warning("Feature importance not available for this model.")

    # Suggested Tweaks
    st.markdown("### ✍️ Suggested Title Tweaks")
    suggestions = []
    if features['title_length'][0] < 40:
        suggestions.append("📏 Try increasing title length to ~45–70 characters.")
    if features['num_words'][0] < 5:
        suggestions.append("📝 Add more descriptive or keyword-rich words.")
    if features['has_question'][0] == 0:
        suggestions.append("❓ Consider using a question format to increase curiosity.")
    if not suggestions:
        st.success("Your title already meets several optimization criteria!")
    else:
        for s in suggestions:
            st.markdown(f"- {s}")

    # Display Extracted Features
    st.subheader("🧪 Extracted Features")
    st.dataframe(features)
