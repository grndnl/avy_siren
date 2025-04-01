import os
import librosa
import noisereduce as nr
import numpy as np
import soundfile as sf

# Config
AUDIO_FOLDER = "test_audio"  # Folder containing WAV files
NOISE_CLIP_FILE = "cropped_noise_clip.wav"  # Your clean hum profile
SAMPLERATE = 16000  # Match this to your files
CHUNK_DURATION = 1.0  # seconds
ENERGY_THRESHOLD = 1000  # Tune this based on your data

# Load noise reference
noise_clip, _ = librosa.load(NOISE_CLIP_FILE, sr=SAMPLERATE)

# Function: Cannon Detection by Low-Freq Energy
def detect_cannon(chunk, sr):
    fft = np.fft.rfft(chunk)
    freqs = np.fft.rfftfreq(len(chunk), 1/sr)
    low_band = (freqs > 30) & (freqs < 80)
    energy = np.sum(np.abs(fft[low_band]))
    return energy > ENERGY_THRESHOLD

# Function: Process one file
def process_file(file_path):
    print(f"Processing: {file_path}")
    y, sr = librosa.load(file_path, sr=SAMPLERATE)
    chunk_size = int(CHUNK_DURATION * sr)
    detections = []

    for i in range(0, len(y), chunk_size):
        chunk = y[i:i+chunk_size]
        if len(chunk) < chunk_size:
            break

        cleaned = nr.reduce_noise(y=chunk, y_noise=noise_clip, sr=sr, prop_decrease=0.25, stationary=False)
        if detect_cannon(cleaned, sr):
            timestamp = round(i / sr, 2)
            detections.append(timestamp)

    return detections

# Main: Process All Files
results = {}

for filename in os.listdir(AUDIO_FOLDER):
    if filename.endswith(".wav"):
        path = os.path.join(AUDIO_FOLDER, filename)
        timestamps = process_file(path)
        results[filename] = timestamps

# Output results
for fname, times in results.items():
    print(f"\n📄 {fname}")
    for t in times:
        print(f"  💥 Cannon at {t} sec")

# Optional: Save results for comparison
import csv
with open("detections.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "timestamp_sec"])
    for fname, times in results.items():
        for t in times:
            writer.writerow([fname, t])
