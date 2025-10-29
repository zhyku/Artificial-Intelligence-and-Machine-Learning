import os
import uuid
import time
import io
from typing import Optional, List
from contextlib import asynccontextmanager # Import for lifespan
# import asyncio # No longer needed if asyncio.to_thread for Firestore is removed

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import numpy
import numpy as np

# NO Firebase Admin SDK imports needed if not interacting with Firestore from backend
# import firebase_admin
# from firebase_admin import credentials, firestore, auth

# Import your feature extraction logic
from feature_extractor import extract_features, SAMPLE_RATE # Assuming feature_extractor.py is in the same directory

# Global variable to hold the loaded model
emotion_model = None

# --- FastAPI App Initialization ---
# Define lifespan context manager for startup/shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    global emotion_model # Declare global for the model

    # --- Emotion Model Loading ---
    MODEL_PATH = "ser_model_sklearn.pkl" # Assuming this is your model file
    print(f"Loading emotion model from: {MODEL_PATH}")
    try:
        import joblib # Assuming you use joblib to load your sklearn model
        emotion_model = joblib.load(MODEL_PATH)
        print("Emotion model loaded successfully!")
    except FileNotFoundError:
        print(f"ERROR: Model file not found at {MODEL_PATH}. Please ensure the model is in the correct path.")
        emotion_model = None
    except Exception as e:
        print(f"ERROR loading emotion model: {e}")
        emotion_model = None

    yield # Application startup is complete, yield control to FastAPI

    # --- SHUTDOWN LOGIC (runs when app stops) ---
    print("Application shutting down...")
    # Any cleanup for your model or other resources can go here
    print("Application shut down complete.")


app = FastAPI(
    title="Emotion Detection Backend",
    description="API for real-time emotion detection from audio streams (without Firebase Logging).",
    version="1.0.0",
    lifespan=lifespan # Assign the lifespan context manager
)

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to your frontend's domain(s) in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Model for Emotion Log (NO LONGER USED AS FIRESTORE INTERACTION REMOVED) ---
# This class is commented out because it's only used for Firestore data types,
# which are now removed. If you plan to store logs in a different way, you might
# define a similar model for that purpose.
# class EmotionLogSchema(BaseModel):
#     detected_emotion: str
#     confidence_score: float
#     timestamp: int
#     agent_id: str
#     call_id: str
#     id: Optional[str] = None
#     all_emotions_confidence: Optional[dict] = None


# --- Emotion Labels Mapping ---
# Ensure these match the order of your model's output classes
EMOTION_LABELS = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

# --- Emotion Prediction Function ---
def predict_emotion_from_features(features: np.ndarray) -> (str, float, dict):
    """
    Predicts emotion from extracted audio features using the loaded model.
    Returns detected emotion, confidence score, and all emotion probabilities.
    """
    if emotion_model is None:
        print("Emotion model is not loaded during prediction attempt.")
        # Return a consistent 'unknown' state with dummy probabilities
        return "unknown", 0.0, {label: 0.0 for label in EMOTION_LABELS}

    try:
        # Ensure features are correctly shaped for the model
        # Assuming 'features' from extract_features is (n_frames, n_features_per_frame)
        # and your sklearn model expects a single sample (1, n_features)
        processed_features = np.mean(features, axis=0).reshape(1, -1)

        # Make prediction
        # For sklearn models, use predict_proba for probabilities
        probabilities = emotion_model.predict_proba(processed_features)[0]
        predicted_class_index = np.argmax(probabilities)
        detected_emotion = EMOTION_LABELS[predicted_class_index]
        confidence_score = probabilities[predicted_class_index]

        # Prepare all emotions confidence for frontend
        all_emotions_confidence = {label: float(prob) for label, prob in zip(EMOTION_LABELS, probabilities)}

        return detected_emotion, float(confidence_score), all_emotions_confidence

    except Exception as e:
        print(f"Error during emotion prediction: {e}")
        return "error", 0.0, {label: 0.0 for label in EMOTION_LABELS}


# --- API Endpoints ---

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Speech Emotion Recognition API (without Firebase Logging)"}


@app.post("/analyze/stream/")
async def analyze_audio_stream(audio_file: UploadFile = File(...)):
    """
    Receives an audio stream, extracts features, and predicts emotion.
    NOTE: This version does NOT save logs to Firestore.
    """
    if emotion_model is None:
        raise HTTPException(status_code=503, detail="Emotion model not loaded. Service unavailable.")

    print(f"Received /analyze/stream/ request. Audio data length: {audio_file.size} bytes")
    print(f"Content-Type header received: {audio_file.content_type}")

    try:
        # Read the audio data into bytes
        audio_bytes = await audio_file.read()
        
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="Empty audio data received.")

        # Extract features from the audio bytes
        features = extract_features(audio_bytes, sample_rate=SAMPLE_RATE) # Pass SAMPLE_RATE explicitly

        if features is None or features.size == 0:
            raise HTTPException(status_code=400, detail="Feature extraction failed or resulted in empty features.")

        # Predict emotion using your model
        detected_emotion, confidence_score, all_emotions_confidence = predict_emotion_from_features(features)

        # No Firestore logging in this version

        return {
            "detected_emotion": detected_emotion,
            "confidence_score": confidence_score,
            "all_emotions_confidence": all_emotions_confidence # Return all probabilities to frontend
        }

    except Exception as e:
        print(f"Error during audio processing or prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")


# The /logs endpoint is removed as it relied entirely on Firebase Firestore
# @app.get("/logs", response_model=List[EmotionLogSchema])
# async def get_emotion_logs(limit: int = 20):
#     """
#     Retrieves the most recent emotion logs from Firestore.
#     This endpoint is removed because Firebase integration is disabled.
#     """
#     pass


# --- Main execution block for Uvicorn ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)