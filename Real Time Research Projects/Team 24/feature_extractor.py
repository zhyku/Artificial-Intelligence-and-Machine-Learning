import librosa
import numpy as np
import io # Import io for BytesIO

# Define the sample rate for processing (common for speech)
SAMPLE_RATE = 16000 # Hz

# Define N_MFCC as a global constant so it can be imported
N_MFCC = 40 # This should match the n_mfcc used during your model's training

def extract_features(audio_path_or_bytes, sample_rate=SAMPLE_RATE, n_mfcc=N_MFCC, hop_length=512):
    """
    Extracts MFCCs, pitch (f0), and energy features from an audio source.

    Args:
        audio_path_or_bytes (str or bytes): Path to the audio file or raw audio bytes.
        sample_rate (int): The sample rate to resample the audio to.
        n_mfcc (int): Number of MFCCs to extract.
        hop_length (int): The number of samples between successive frames.

    Returns:
        np.array: A 2D array of concatenated features (MFCCs, f0, energy) per frame.
                  Returns None if audio processing fails.
    """
    try:
        # Load audio from path or bytes
        if isinstance(audio_path_or_bytes, str):
            # For file paths, librosa can directly load
            y, sr = librosa.load(audio_path_or_bytes, sr=sample_rate, mono=True)
        elif isinstance(audio_path_or_bytes, bytes):
            # For raw bytes, librosa.load can also take a file-like object
            # Use BytesIO to make bytes data behave like a file
            y, sr = librosa.load(io.BytesIO(audio_path_or_bytes), sr=sample_rate, mono=True)
        else:
            raise TypeError("Input must be a file path (str) or audio bytes.")

        # Ensure audio is not empty after loading/conversion
        if len(y) == 0:
            print("Warning: Received empty audio or audio content is empty after loading.")
            return np.array([]) # Return empty numpy array to indicate no features

        # Ensure consistent sample rate if not already done by librosa.load
        if sr != sample_rate:
            y = librosa.resample(y=y, orig_sr=sr, target_sr=sample_rate)

        # 1. MFCCs (Mel-frequency cepstral coefficients)
        # Transpose to get (n_mfcc, n_frames) -> then .T to get (n_frames, n_mfcc)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length).T

        # 2. Pitch (f0) using pyin for robustness
        # pyin can sometimes return fewer frames or NaNs, so handle padding/nan_to_num
        f0, voiced_flag, voiced_probs = librosa.pyin(y=y, sr=sr, fmin=librosa.note_to_hz('C2'),
                                                     fmax=librosa.note_to_hz('C5'),
                                                     frame_length=hop_length * 2,
                                                     hop_length=hop_length)
        f0 = np.nan_to_num(f0) # Replace NaNs with 0 (unvoiced)

        # 3. Energy (Root Mean Square - RMS)
        rms = librosa.feature.rms(y=y, frame_length=hop_length * 2, hop_length=hop_length).T

        # Ensure all features have the same number of frames before concatenating
        min_frames = min(mfccs.shape[0], f0.shape[0], rms.shape[0])

        # If any feature has 0 frames, return empty array to prevent dimension mismatch
        if min_frames == 0:
            return np.array([])

        mfccs = mfccs[:min_frames, :]
        f0 = f0[:min_frames]
        rms = rms[:min_frames, :]

        # Concatenate features: mfccs (frames, n_mfcc), f0 (frames, 1), rms (frames, 1)
        features = np.concatenate((mfccs, f0.reshape(-1, 1), rms), axis=1)

        return features

    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

# --- Example Usage (requires an audio file, e.g., 'sample.wav') ---
if __name__ == "__main__":
    # Create a dummy audio file for testing if you don't have one
    # You might need to install 'soundfile': pip install soundfile
    import soundfile as sf

    dummy_audio_path = "dummy_audio.wav"
    try:
        # Generate 5 seconds of sine wave at 440Hz
        t = np.linspace(0, 5, int(SAMPLE_RATE * 5), endpoint=False)
        dummy_audio_data = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
        sf.write(dummy_audio_path, dummy_audio_data, SAMPLE_RATE)
        print(f"Generated dummy audio file: {dummy_audio_path}")

        features = extract_features(dummy_audio_path)
        if features is not None and features.size > 0:
            print(f"Features shape: {features.shape}")
            # Expected shape: (number_of_frames, n_mfcc + 1 (for f0) + 1 (for rms))
            # Example: (157, 42) for 5 seconds of 16kHz audio with hop_length=512
            print(f"Number of MFCCs: {features.shape[1] - 2}") # Subtract 2 for f0 and rms
        else:
            print("Feature extraction failed or resulted in empty features.")

        # Example with raw bytes (simulating a stream chunk from frontend)
        with open(dummy_audio_path, 'rb') as f:
            # Read 2 seconds of audio data (assuming 16-bit, mono)
            audio_bytes_chunk = f.read(SAMPLE_RATE * 2 * 2) # 2 seconds * 2 bytes/sample (16-bit) * 1 channel (mono)

            print(f"\nTesting with bytes (simulated audio chunk): {len(audio_bytes_chunk)} bytes")
            features_from_bytes = extract_features(audio_bytes_chunk, sample_rate=SAMPLE_RATE)
            if features_from_bytes is not None and features_from_bytes.size > 0:
                print(f"Features shape from bytes: {features_from_bytes.shape}")
            else:
                print("Feature extraction from bytes failed or resulted in empty features.")

    except ImportError:
        print("\nInstall 'soundfile' to generate dummy audio: pip install soundfile")
    except FileNotFoundError:
        print(f"Error: {dummy_audio_path} not found. Please create an audio file or install soundfile.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")