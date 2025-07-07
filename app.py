import streamlit as st
import numpy as np
import cv2
import os
import pickle
import uuid
import yt_dlp
from keras.models import load_model

# === Load Resources ===

@st.cache_resource
def load_model_cached():
    return load_model("models/LRCN_MODEL_Date_Time_2025_07_07_14_05_42_Loss_0.370_Accuracy_0.905.keras")

@st.cache_resource
def load_class_list():
    with open("models/class_list1.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_mean_std():
    with open("models/mean.pkl", "rb") as f1, open("models/std_array.pkl", "rb") as f2:
        return pickle.load(f1), pickle.load(f2)

# === Video Processing ===

def preprocess_frames(video_path, sequence_length=20):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    skip = max(total_frames // sequence_length, 1)

    frames = []
    for i in range(sequence_length):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * skip)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (64, 64))
        frame = frame.astype(np.float32) / 255.0
        frames.append(frame)

    cap.release()
    return np.array(frames)

def predict_action(video_path):
    frames = preprocess_frames(video_path)
    if frames.shape[0] < 20:
        return "Not enough frames", 0.0

    mean, std = load_mean_std()
    frames = (frames - mean) / std
    input_data = np.expand_dims(frames, axis=0)

    model = load_model_cached()
    predictions = model.predict(input_data)[0]
    class_list = load_class_list()
    predicted_index = np.argmax(predictions)
    confidence = predictions[predicted_index]

    return class_list[predicted_index], confidence

def download_youtube_video(url, output_folder="test_video"):
    os.makedirs(output_folder, exist_ok=True)
    video_id = str(uuid.uuid4())
    output_path = os.path.join(output_folder, f"{video_id}.mp4")
    ydl_opts = {
        'format': '18',  # 360p mp4
        'outtmpl': output_path,
        'quiet': True,
        'no_warnings': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    return output_path

# === Streamlit UI ===

st.title("🎥 Human Activity Recognition from YouTube")

youtube_url = st.text_input("Enter a YouTube video URL")

if st.button("Download and Predict"):
    if not youtube_url:
        st.error("Please enter a valid YouTube URL.")
    else:
        with st.spinner("Downloading video..."):
            try:
                video_path = download_youtube_video(youtube_url)
                st.success("✅ Downloaded video!")
                st.video(video_path)

                with st.spinner("Predicting action..."):
                    label, confidence = predict_action(video_path)
                    st.success(f"🏷️ Predicted Action: **{label}** ({confidence:.2f} confidence)")

            except Exception as e:
                st.error(f"Error downloading or processing video: {e}")
