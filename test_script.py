import os
import librosa
import noisereduce as nr
import numpy as np
import soundfile as sf

# Config
AUDIO_FOLDER = "recordings"  # Folder containing WAV files
NOISE_CLIP_FILE = "cropped_noise_clip.wav"  # Your clean hum profile
SAMPLERATE = 44100  # Match this to your files
CHUNK_DURATION = 0.5  # seconds
ENERGY_THRESHOLD = 50  # Tune this based on your data
ENERGY_THRESHOLD_dB = 10  # Tune this based on your data

# Load noise reference
# noise_clip, _ = librosa.load(NOISE_CLIP_FILE, sr=SAMPLERATE)

# Function: Cannon Detection by Low-Freq Energy
def detect_cannon(chunk, sr):
    fft = np.fft.rfft(chunk)
    freqs = np.fft.rfftfreq(len(chunk), 1/sr)
    low_band = (freqs > 30) & (freqs < 80)
    energy = np.sum(np.abs(fft[low_band]))
    return energy > ENERGY_THRESHOLD

def detect_cannon_dB(chunk, sr, db_threshold=-30):
    fft = np.fft.rfft(chunk)
    freqs = np.fft.rfftfreq(len(chunk), 1/sr)
    magnitude = np.abs(fft)

    # Convert to dB
    magnitude_db = 20 * np.log10(magnitude + 1e-6)

    # Focus on cannon band (30–80 Hz)
    low_band = (freqs > 20) & (freqs < 100)
    avg_db = np.mean(magnitude_db[low_band])
    print(f"Avg dB (30–80 Hz): {avg_db:.2f}")

    return avg_db > db_threshold  # e.g., -30 dB

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

        # chunk = nr.reduce_noise(y=chunk, y_noise=noise_clip, sr=sr, prop_decrease=0.25, stationary=False)
        if detect_cannon(chunk, sr):
        # if detect_cannon_dB(chunk, sr, ENERGY_THRESHOLD_dB):
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
    print(f"  Total: {len(times)} detections")

print(f"total detections: {sum([len(times) for times in results.values()])}")

# Optional: Save results for comparison
import csv
with open("detections.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["file_name", "seconds"])
    for fname, times in results.items():
        for t in times:
            writer.writerow([fname, t])
