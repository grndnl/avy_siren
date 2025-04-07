import librosa
import numpy as np
import matplotlib.pyplot as plt
import os

# === Config ===
# AUDIO_FILE = "recordings/audio_2025-04-01_07-26-49.wav"
AUDIO_FILE = "recordings/audio_2025-04-01_12-48-49.wav"
START_TIME = 0.0     # seconds (cannon should start here)
DURATION = 59.0       # seconds (chunk length)
SAMPLERATE = 44100
OUTPUT_FILE = "fft_cannon_plot.png"


# === Load audio ===
y, sr = librosa.load(AUDIO_FILE, sr=SAMPLERATE)
start_sample = int(START_TIME * sr)
end_sample = int((START_TIME + DURATION) * sr)
chunk = y[start_sample:end_sample]

# === FFT ===
fft = np.fft.rfft(chunk)
freqs = np.fft.rfftfreq(len(chunk), d=1/sr)
magnitude = np.abs(fft)

# === Plot and Save ===
plt.figure(figsize=(10, 5))
plt.plot(freqs, magnitude)
plt.xlim(0, 200)  # Focus on cannon-relevant frequencies
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.title(f"FFT of Audio Chunk: {START_TIME:.1f}–{START_TIME + DURATION:.1f}s")
plt.grid(True)
plt.tight_layout()

plt.savefig(OUTPUT_FILE, dpi=150)
print(f"Saved FFT plot to {os.path.abspath(OUTPUT_FILE)}")
