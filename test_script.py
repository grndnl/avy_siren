import os
import librosa
import numpy as np
import csv
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Config
AUDIO_FOLDER = "recordings"
GROUND_TRUTH_FILE = "positives.csv"
OUTPUT_FILE = "detections.csv"
SAMPLERATE = 44100
CHUNK_DURATION = 0.5  # seconds
THRESHOLD = 30
MATCH_TOLERANCE = 0  # seconds to consider a true match
LOW_FREQ_BAND = [0, 128]

# === Detection Functions ===

def detect_cannon(chunk, sr):
    fft = np.fft.rfft(chunk)
    magnitude = np.abs(fft)
    freqs = np.fft.rfftfreq(len(chunk), d=1/sr)
    low_band = (freqs > LOW_FREQ_BAND[0]) & (freqs < LOW_FREQ_BAND[1])
    avg_mag = np.mean(magnitude[low_band])
    return avg_mag > THRESHOLD

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

# === Generate Classification Labels ===

y_true = []
y_pred = []

for filename in os.listdir(AUDIO_FOLDER):
    if not filename.endswith(".wav"):
        continue

    path = os.path.join(AUDIO_FOLDER, filename)
    y, sr = librosa.load(path, sr=SAMPLERATE)
    chunk_size = int(CHUNK_DURATION * sr)
    num_chunks = len(y) // chunk_size

    detections = set(results.get(filename, []))
    ground_truth = set(gt[gt["file_name"] == filename]["seconds"].tolist())

    for i in range(num_chunks):
        timestamp = round(i * CHUNK_DURATION, 2)

        # Ground truth label
        is_true = any(abs(timestamp - t) <= MATCH_TOLERANCE for t in ground_truth)
        y_true.append(1 if is_true else 0)

        # Prediction label
        is_pred = any(abs(timestamp - d) <= MATCH_TOLERANCE for d in detections)
        y_pred.append(1 if is_pred else 0)

# === Classification Report ===

print("\n🧾 Classification Report:")
print(classification_report(y_true, y_pred, target_names=["No Cannon", "Cannon"]))

# === Confusion Matrix ===

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Cannon", "Cannon"])
disp.plot(cmap='Blues', values_format='d')
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close()

print("\n🖼️ Confusion matrix saved to 'confusion_matrix.png'")
