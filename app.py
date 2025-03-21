import json
import numpy as np
import librosa
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit
from transformers import pipeline
import eventlet

# Initialize Flask App
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet")

# Emotion Labels (Ensure correct mapping)
EMOTION_LABELS = ["neutral", "happy", "sad", "angry", "fearful", "disgust", "surprised"]

# Load the trained emotion detection model
try:
    model = tf.keras.models.load_model("mental_health_emotion_model.keras")
    print("✅ Emotion detection model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading emotion detection model: {e}")
    model = None

# Load NLP model for sentiment analysis
try:
    sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")
    print("✅ Sentiment analysis model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading sentiment analysis model: {e}")
    sentiment_pipeline = None

# Function to process audio input
def extract_features(audio_path, max_pad_length=174):
    try:
        audio, sample_rate = librosa.load(audio_path, sr=22050)
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        pad_width = max_pad_length - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, max(0, pad_width))), mode='constant')
        return mfccs.T  # Transposing to (timesteps, features)
    except Exception as e:
        print(f"❌ Error processing {audio_path}: {e}")
        return None

# Handle incoming WebSocket messages
@socketio.on("message")
def handle_message(data):
    try:
        print(f"🔹 Received WebSocket message: {data}")

        # Ensure data is a valid dictionary
        if isinstance(data, str):
            try:
                data = json.loads(data)  # Convert JSON string to dictionary
            except json.JSONDecodeError:
                emit("response", {"error": "Invalid JSON format"})
                return

        if not isinstance(data, dict):
            emit("response", {"error": "Invalid message format. Expected JSON."})
            return

        text = data.get("text", "")
        audio_path = data.get("audio_path", None)
        response = ""

        # Process audio emotion detection
        if audio_path and model:
            features = extract_features(audio_path)
            if features is not None:
                features = np.expand_dims(features, axis=0)
                prediction = model.predict(features)

                if isinstance(prediction, np.ndarray) and prediction.shape[1] == len(EMOTION_LABELS):
                    emotion_index = np.argmax(prediction)
                    emotion = EMOTION_LABELS[emotion_index]
                    response += f"Detected emotion: {emotion}. "
                else:
                    response += "Error in emotion detection. "
            else:
                response += "Error processing audio. "

        # Process text sentiment analysis
        if text and sentiment_pipeline:
            sentiment_result = sentiment_pipeline(text)[0]
            sentiment = sentiment_result["label"].lower()
            response += f"Detected sentiment: {sentiment}. "

        # Generate chatbot response
        if "sad" in response or "negative" in response:
            response += "I'm here for you. Would you like to talk about what's troubling you?"
        elif "happy" in response or "positive" in response:
            response += "That's wonderful to hear! Keep spreading positivity."
        else:
            response += "Tell me more about how you're feeling."

        # Send response via WebSocket
        emit("response", {"message": response})
        print(f"✅ Response sent: {response}")

    except Exception as e:
        print(f"❌ WebSocket error: {e}")
        emit("response", {"error": f"An error occurred: {e}"})

# Basic HTTP Route
@app.route("/")
def index():
    return "Real-time Mental Health Assistant Running!"

# Run the Flask SocketIO Server
if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000, debug=True)
