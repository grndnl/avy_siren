import noisereduce as nr
import librosa
import soundfile as sf
import time
import matplotlib.pyplot as plt
import librosa.display
import numpy as np

def save_spectogram(sr, audio, name):
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Reduced Audio Spectrogram')
    plt.tight_layout()
    plt.savefig(f"spectrogram_{name}.png", dpi=150)



# Load files
start_time = time.time()
signal, sr = librosa.load("recordings/audio_2025-04-01_07-26-49.wav", sr=None)
noise, _ = librosa.load("recordings/audio_2025-04-01_07-27-49.wav", sr=sr)

# Apply noise reduction
# reduced = nr.reduce_noise(y=signal, y_noise=noise, sr=sr)
reduced = nr.reduce_noise(
    y=signal,
    y_noise=noise,
    sr=sr,
    prop_decrease=0.25,           # Gentle reduction, preserve signal
    stationary=False,
    time_mask_smooth_ms=800,      # Match cannon duration smoothing
    # freq_mask_smooth_hz=43,       # Let low frequencies breathe
    n_fft=2048,                   # Better low-frequency resolution
    win_length=2048,
    hop_length=512
)


# Save the result
sf.write("recordings/audio_2025-04-01_07-26-49_clean.wav", reduced, sr)

# Print processing time
end_time = time.time()
print(f"Processing time: {end_time - start_time:.2f} seconds")

# visualize the spectrogram
save_spectogram(sr, reduced, "clean")
save_spectogram(sr, signal, "original")
