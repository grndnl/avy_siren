import os
import librosa
import numpy as np
import csv
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

# Config
AUDIO_FOLDER = "recordings"
GROUND_TRUTH_FILE = "positives.csv"
OUTPUT_FILE = "detections.csv"
SAMPLERATE = 44100
CHUNK_DURATION = 0.5  # seconds
ENERGY_THRESHOLD = 33.1
MATCH_TOLERANCE = 0  # seconds to consider a true match

# === Detection Functions ===

def detect_cannon(chunk, sr):
    fft = np.fft.rfft(chunk)
    freqs = np.fft.rfftfreq(len(chunk), 1/sr)
    low_band = (freqs > 0) & (freqs < 64)
    energy = np.sum(np.abs(fft[low_band]))
    return energy > ENERGY_THRESHOLD

# === Process One File ===

def process_file(file_path):
    print(f"Processing: {file_path}")
    y, sr = librosa.load(file_path, sr=SAMPLERATE)
    chunk_size = int(CHUNK_DURATION * sr)
    detections = []

    for i in range(0, len(y), chunk_size):
        chunk = y[i:i+chunk_size]
        if len(chunk) < chunk_size:
            break

        if detect_cannon(chunk, sr):
            timestamp = round(i / sr, 2)
            detections.append(timestamp)

    return detections

# === Load Ground Truth ===

gt = pd.read_csv(GROUND_TRUTH_FILE)

# === Main Detection Loop ===

results = {}

for filename in os.listdir(AUDIO_FOLDER):
    if filename.endswith(".wav"):
        path = os.path.join(AUDIO_FOLDER, filename)
        detected = process_file(path)
        results[filename] = detected

# === Save Detection Results ===

with open(OUTPUT_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["file_name", "seconds"])
    for fname, times in results.items():
        for t in times:
            writer.writerow([fname, t])

print(f"\n✅ Saved detection results to {OUTPUT_FILE}")

# === Evaluation ===

tp = fp = fn = 0
all_detections = []

for fname, detected in results.items():
    detected = list(set(detected))  # Remove duplicates
    truth = gt[gt["file_name"] == fname]["seconds"].tolist()
    truth_matched = [False] * len(truth)

    for d in detected:
        matched = False
        for i, t in enumerate(truth):
            if abs(d - t) <= MATCH_TOLERANCE and not truth_matched[i]:
                tp += 1
                truth_matched[i] = True
                matched = True
                break
        if not matched:
            fp += 1

    fn += truth_matched.count(False)

# === Metrics ===

precision = tp / (tp + fp + 1e-6)
recall = tp / (tp + fn + 1e-6)
f1 = 2 * (precision * recall) / (precision + recall + 1e-6)

print(f"\n📊 Detection Evaluation (tolerance = {MATCH_TOLERANCE}s):")
print(f"  True Positives:  {tp}")
print(f"  False Positives: {fp}")
print(f"  False Negatives: {fn}")
print(f"  Precision:       {precision:.2f}")
print(f"  Recall:          {recall:.2f}")
print(f"  F1 Score:        {f1:.2f}")
