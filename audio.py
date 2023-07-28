import time

import numpy as np
import pyaudio

FORMAT = pyaudio.paFloat32  # sample format
CHANNELS = 1  # mono audio
RATE = 16000  # sample rate (Hz)
CHUNK = 16000  # frames per buffer

audio_buffer = None


def get_audio_buffer_from_mic(SECONDS: int, new_buffer_callback) -> np.ndarray:
    global audio_buffer
    audio_buffer = np.zeros(RATE * SECONDS, dtype=np.float32)

    def callback(in_data, frame_count, time_info, status):
        global audio_buffer
        # Convert data to numpy array
        np_data = np.frombuffer(in_data, dtype=np.float32)

        # shift buffer back by frame_count
        audio_buffer = np.roll(audio_buffer, -frame_count)

        # Zero out last frame_count elements to prevent overlap
        audio_buffer[-frame_count:] = 0

        # insert data into end of buffer
        audio_buffer[-frame_count:] = np_data

        new_buffer_callback(audio_buffer, np_data)

        return (in_data, pyaudio.paContinue)

    audio = pyaudio.PyAudio()

    # Open a streaming object with callback
    stream = audio.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        output=False,
                        input=True,
                        frames_per_buffer=CHUNK,
                        stream_callback=callback)

    while stream.is_active():
        time.sleep(0.1)

    # Close the stream
    stream.stop_stream()
    stream.close()

    # Terminate the PortAudio interface
    audio.terminate()
