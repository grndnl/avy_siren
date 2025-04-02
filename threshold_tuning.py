import os
import pandas as pd
import numpy as np
import librosa
from sklearn.metrics import precision_score, recall_score, f1_score
from itertools import product


# === Config ===
AUDIO_FOLDER = "recordings"  # Folder with .wav files
GROUND_TRUTH_CSV = "positives.csv"
SAMPLERATE = 44100
CHUNK_DURATION = 0.5  # seconds
FREQ_RANGE = (30, 80)
TOLERANCE = 0.5  # seconds for matching detected vs. true
THRESHOLDS = np.linspace(20, 100, 50)  # Magnitude thresholds to test

# Frequency ranges to try: [(low1, high1), (low2, high2), ...]
FREQ_RANGES = [
    (0, 256),
    (0, 128),
    (0, 64),
    (0, 32),
    (20, 60),
    (30, 60),
    (30, 80),
    (20, 80),
    (30, 70),
]

# === Load ground truth ===
gt = pd.read_csv(GROUND_TRUTH_CSV)

def load_audio(filename):
    return librosa.load(os.path.join(AUDIO_FOLDER, filename), sr=SAMPLERATE)[0]

def detect_cannon(chunk, freq_range, threshold, sr=SAMPLERATE):
    fft = np.fft.rfft(chunk)
    freqs = np.fft.rfftfreq(len(chunk), 1/sr)
    magnitude = np.abs(fft)
    band = (freqs > freq_range[0]) & (freqs < freq_range[1])
    avg_mag = np.mean(magnitude[band])
    return avg_mag > threshold

def match_timestamps(detected, truth, tolerance):
    matched = []
    for t in truth:
        for d in detected:
            if abs(d - t) <= tolerance:
                matched.append(d)
                break
    tp = len(matched)
    fp = len(detected) - tp
    fn = len(truth) - tp
    return tp, fp, fn

# === Main sweep loop ===
results = []

for freq_range, threshold in product(FREQ_RANGES, THRESHOLDS):
    tp_total = fp_total = fn_total = 0

    for fname in gt["file_name"].unique():
        y = load_audio(fname)
        truth_times = gt[gt["file_name"] == fname]["seconds"].values
        chunk_size = int(CHUNK_DURATION * SAMPLERATE)
        detected_times = []

        for i in range(0, len(y), chunk_size):
            chunk = y[i:i+chunk_size]
            if len(chunk) < chunk_size:
                break
            if detect_cannon(chunk, freq_range, threshold):
                timestamp = round(i / SAMPLERATE, 2)
                detected_times.append(timestamp)

        tp, fp, fn = match_timestamps(detected_times, truth_times, TOLERANCE)
        tp_total += tp
        fp_total += fp
        fn_total += fn

    precision = tp_total / (tp_total + fp_total + 1e-6)
    recall = tp_total / (tp_total + fn_total + 1e-6)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)

    results.append({
        "low_freq": freq_range[0],
        "high_freq": freq_range[1],
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1
    })

# === Select the best result (based on recall or f1)
df = pd.DataFrame(results)
best = df.sort_values(by="f1", ascending=False).iloc[0]

print("\n🎯 Best Config (based on recall):")
print(f"  Frequency band: {best.low_freq}–{best.high_freq} Hz")
print(f"  Threshold:      {best.threshold:.1f}")
print(f"  Precision:      {best.precision:.2f}")
print(f"  Recall:         {best.recall:.2f}")
print(f"  F1 Score:       {best.f1:.2f}")

# === Save results to CSV
df.to_csv("freq_band_tuning_results.csv", index=False)
print("\n📄 Saved all results to freq_band_tuning_results.csv")
