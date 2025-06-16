import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load('youtube_virality_model.pkl')

st.title("🎥 YouTube Shorts Virality Predictor")

# Input form
video_length = st.number_input("Video Length (seconds)", min_value=1, max_value=60, value=15)
hashtags = st.number_input("Number of Hashtags", min_value=0, max_value=10, value=3)
category = st.selectbox("Category", ["Comedy", "Tech", "Lifestyle", "Gaming", "Music"])
desc_length = st.number_input("Description Length (characters)", min_value=0, max_value=300, value=100)
title_length = st.number_input("Enter title length (number of characters)", min_value=1, max_value=100)
upload_time = st.slider("Upload Time (24hr format)", min_value=0, max_value=23, value=15)


# Map category to one-hot encoding
categories = ["Comedy", "Tech", "Lifestyle", "Gaming", "Music"]
cat_features = [1 if category == c else 0 for c in categories]

# Make prediction
if st.button("Predict Virality"):
    X_input = [video_length, hashtags, desc_length, title_length, upload_time] + cat_features
    X_input = np.array(X_input).reshape(1, -1)
    result = model.predict(X_input)[0]
    
    if result == 1:
        st.success("✅ Your YouTube Short is likely to go VIRAL!")
    else:
        st.info("ℹ️ Your YouTube Short may not go viral. Try optimizing features!")
