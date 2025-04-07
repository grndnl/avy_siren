import librosa
import numpy as np
import matplotlib.pyplot as plt

# === Config ===
# AUDIO_FILE = "recordings/audio_2025-04-01_07-26-49.wav"   # ← Update this
# AUDIO_FILE = "recordings/audio_2025-04-01_07-27-49.wav"   # ← Update this
AUDIO_FILE = "recordings/audio_2025-04-01_12-48-49.wav"
SAMPLERATE = 44100
CHUNK_DURATION = 0.5  # seconds
LOW_FREQ_BAND = (30, 80)
THRESHOLD = 50

# === Load audio ===
y, sr = librosa.load(AUDIO_FILE, sr=SAMPLERATE)
print(f"Loaded {AUDIO_FILE} with {len(y)} samples at {sr} Hz")
chunk_size = int(CHUNK_DURATION * sr)

# === Process in chunks ===
timestamps = []
avg_magnitudes = []

for i in range(0, len(y), chunk_size):
    chunk = y[i:i+chunk_size]
    if len(chunk) < chunk_size:
        break

    fft = np.fft.rfft(chunk)
    freqs = np.fft.rfftfreq(len(chunk), d=1/sr)
    magnitude = np.abs(fft)

    # Average magnitude in 30–80 Hz band
    low_band = (freqs > LOW_FREQ_BAND[0]) & (freqs < LOW_FREQ_BAND[1])
    avg_mag = np.mean(magnitude[low_band])

    avg_magnitudes.append(avg_mag)
    timestamps.append(i / sr)

# === Plot ===
plt.figure(figsize=(12, 5))
plt.plot(timestamps, avg_magnitudes, marker='o')
plt.axhline(y=THRESHOLD, color='r', linestyle='--', label=f'Example threshold: {THRESHOLD}')
plt.title(f"Raw FFT Magnitude (30–80 Hz): {AUDIO_FILE}")
plt.xlabel("Time (s)")
plt.ylabel("Avg Magnitude (30–80 Hz)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("magnitude_plot.png", dpi=150)
print("Saved plot to magnitude_plot.png")
