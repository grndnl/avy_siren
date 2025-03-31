import os
import datetime
import subprocess
import time

# Directory to save recordings
SAVE_DIR = "/home/grndnl/recordings"
os.makedirs(SAVE_DIR, exist_ok=True)

# Recording parameters
DURATION = 60  # seconds
DEVICE = "plughw:2,0"  # adjust based on your 'arecord -l' output


def get_filename():
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return os.path.join(SAVE_DIR, f"audio_{timestamp}.wav")


def record_chunk():
    filename = get_filename()
    print(f"Recording: {filename}")

    command = [
        "arecord",
        "-D", DEVICE,
        "-f", "cd",
        "-t", "wav",
        "-d", str(DURATION),
        "-q",  # quiet
        filename
    ]
    subprocess.run(command)


if __name__ == "__main__":
    print("Starting 1-minute chunked recording...")
    try:
        while True:
            record_chunk()
    except KeyboardInterrupt:
        print("\nRecording stopped.")
