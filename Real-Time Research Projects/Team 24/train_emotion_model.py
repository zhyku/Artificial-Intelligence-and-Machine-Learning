import os
import numpy as np
import pandas as pd
import joblib # To save/load the scikit-learn model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC # Or MLPClassifier, RandomForestClassifier, etc.
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import time # For timing operations

# Assuming your feature_extractor.py is in the same directory
from feature_extractor import extract_features, SAMPLE_RATE, N_MFCC

# --- CONFIGURATION ---
# Path to your dataset directory
# >>>>>> YOUR DATASET PATH HAS BEEN UPDATED HERE <<<<<<<
DATASET_PATH = r'C:\Users\SANGEM PAVAN GOUD\OneDrive\Desktop\Speech Emotion Project\speech-emotion-recognition-ravdess-data'

# Output file name for your trained model
MODEL_OUTPUT_FILENAME = 'ser_model_sklearn.pkl'
# Optional: Save the scaler if you want to apply the same scaling during prediction
SCALER_OUTPUT_FILENAME = 'scaler.pkl'

# --- 1. Collect and Process Data ---
# This part is highly dependent on your dataset's structure.
# This example is adapted for the RAVDESS dataset structure.
# You will need to modify this loop if your dataset has a different file/folder structure or labeling scheme.

all_features = [] # To store the feature vectors
all_labels = []   # To store the corresponding emotion labels
processed_files_count = 0
skipped_files_count = 0
start_time = time.time()

print(f"Starting feature extraction from dataset at: {DATASET_PATH}")
print("This may take some time depending on your dataset size...")

# Define emotion mapping for RAVDESS (emotion codes from filename)
ravdess_emotion_map = {
    '01': 'neutral', # neutral
    '02': 'calm',    # calm (often grouped with neutral or removed)
    '03': 'happy',   # happy
    '04': 'sad',     # sad
    '05': 'angry',   # angry
    '06': 'fearful', # fearful
    '07': 'disgusted', # disgusted
    '08': 'surprised' # surprised
}

# --- Iterate through dataset folders/files ---
# This loop assumes a structure like: DATASET_PATH/Actor_01/audio_file.wav
# Adjust 'os.listdir' and file parsing based on your dataset.
try:
    for actor_folder in sorted(os.listdir(DATASET_PATH)):
        actor_path = os.path.join(DATASET_PATH, actor_folder)
        if not os.path.isdir(actor_path):
            continue # Skip if not a directory

        for audio_file_name in os.listdir(actor_path):
            if audio_file_name.endswith('.wav'): # Process only WAV files
                audio_file_path = os.path.join(actor_path, audio_file_name)
                
                # --- Extract emotion label from filename (RAVDESS-specific) ---
                # Filename example: 03-01-01-01-01-01-10.wav
                # parts[2] is the emotion code (e.g., '03' for happy)
                parts = audio_file_name.split('-')
                
                if len(parts) >= 3 and parts[2] in ravdess_emotion_map:
                    emotion_code = parts[2]
                    label = ravdess_emotion_map[emotion_code]
                    
                    # You might want to filter out 'calm' or group it with 'neutral'
                    # For simplicity here, we'll keep all mapped emotions
                    
                    # --- Extract features ---
                    features = extract_features(audio_file_path, sample_rate=SAMPLE_RATE, n_mfcc=N_MFCC)

                    if features is not None and features.size > 0:
                        # For traditional ML models (like SVM, MLP), you often average features
                        # across the frames to get a single feature vector per audio clip.
                        mean_features = np.mean(features, axis=0)
                        all_features.append(mean_features)
                        all_labels.append(label)
                        processed_files_count += 1
                    else:
                        print(f"Warning: Failed to extract features from {audio_file_name} or features were empty.")
                        skipped_files_count += 1
                else:
                    print(f"Skipping {audio_file_name}: Emotion code not found or not mapped in RAVDESS_emotion_map.")
                    skipped_files_count += 1
            else:
                skipped_files_count += 1 # Count non-.wav files or other files

except FileNotFoundError:
    print(f"\nERROR: Dataset path not found: '{DATASET_PATH}'")
    print("Please ensure the 'DATASET_PATH' variable is correct and the directory exists.")
    exit()
except Exception as e:
    print(f"\nAn unexpected error occurred during data collection: {e}")
    exit()

end_time = time.time()
print(f"\nFinished feature extraction. Processed {processed_files_count} files, skipped {skipped_files_count} files.")
print(f"Total time for feature extraction: {end_time - start_time:.2f} seconds.")

if not all_features:
    print("ERROR: No features extracted. Please ensure your DATASET_PATH is correct and contains WAV files with recognizable emotion labels.")
    print("Exiting. Model cannot be trained without data.")
    exit()

# Convert lists to numpy arrays
X = np.array(all_features)
y = np.array(all_labels)

print(f"Total samples for training: {X.shape[0]}")
print(f"Number of features per sample: {X.shape[1]}")
print(f"Unique emotions detected: {np.unique(y)}")

# --- 2. Preprocessing and Splitting Data ---
print("\nSplitting data into training and testing sets...")
# stratify=y ensures that the proportion of classes is the same in both train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

# Scale features (important for many ML models like SVM, Neural Networks)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Data scaled and split.")

# --- 3. Model Training ---
print("\nTraining the Support Vector Classifier model (this may take time)...")
# Choose your classifier. SVC is a good general-purpose choice.
# probability=True is essential if you want predict_proba (confidence scores) in your main.py
model = SVC(kernel='rbf', C=10, probability=True, random_state=42)
model.fit(X_train_scaled, y_train)
print("Model training complete.")

# --- 4. Evaluate Model ---
print("\n--- Model Evaluation ---")
y_pred = model.predict(X_test_scaled)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}") # Higher precision for accuracy

print("\nClassification Report (Precision, Recall, F1-score for each emotion):")
# Add zero_division=0 to handle cases where a class might not appear in test set for specific metrics
print(classification_report(y_test, y_pred, zero_division=0))

# Optional: Plot Confusion Matrix for better visualization of performance
try:
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix(y_test, y_pred, labels=model.classes_),
                annot=True, fmt='d', cmap='Blues',
                xticklabels=model.classes_, yticklabels=model.classes_)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()
except Exception as e:
    print(f"Could not plot confusion matrix: {e}")

# --- 5. Save the Trained Model and Scaler ---
print(f"\nSaving the trained model to {MODEL_OUTPUT_FILENAME}...")
joblib.dump(model, MODEL_OUTPUT_FILENAME)
print(f"Model saved successfully as '{MODEL_OUTPUT_FILENAME}'.")

print(f"Saving the StandardScaler to {SCALER_OUTPUT_FILENAME}...")
joblib.dump(scaler, SCALER_OUTPUT_FILENAME)
print(f"Scaler saved successfully as '{SCALER_OUTPUT_FILENAME}'.")

print("\nTraining script finished. Place 'ser_model_sklearn.pkl' and 'scaler.pkl' in your main.py directory.")