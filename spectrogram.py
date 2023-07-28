import os
import random

import numpy as np
import torch
import torchaudio
from matplotlib import pyplot as plt
from PIL import Image
from torchvision import transforms

mel_spectrogram = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000,
    n_fft=4096,
    win_length=2000,
    hop_length=100,
    window_fn=torch.hamming_window,
    n_mels=256,
    center=True,
    pad_mode='reflect',
    power=2.0,
    norm='slaney',
)

amplitude_to_db = torchaudio.transforms.AmplitudeToDB(
    stype='power', top_db=100)

transform = transforms.Compose([transforms.Resize((256, 161)),
                                transforms.ToTensor()])

to_grayscale = transforms.Grayscale()


def get_spectrogram(data: np.ndarray, RATE: int):
    norm_data = data / np.max(np.abs(data))

    # Get last second of data
    norm_cur_data = norm_data[-RATE:]
    cur_data = data[-RATE:]

    tensor = torch.from_numpy(cur_data).float()
    mel_spec = mel_spectrogram(tensor)
    spec_db = amplitude_to_db(mel_spec)

    norm_tensor = torch.from_numpy(norm_cur_data).float()
    norm_mel_spec = mel_spectrogram(norm_tensor)
    norm_spec_db = amplitude_to_db(norm_mel_spec)
    img = None

    # -80dB can be replaced by sound level indicating silence
    if np.mean(norm_spec_db.numpy()) < -30:
        pass
    else:
        # write to greyscale image
        # make temporary file
        file_id = random.randint(0, 1000000000)
        plt.imsave(f'tmp-{file_id}.png', norm_spec_db.numpy(), cmap='gray',
                   origin='lower', vmin=-100, vmax=0)
        img = Image.open(f'tmp-{file_id}.png')
        img = transform(to_grayscale(img))
        # remove temporary file
        os.remove(f'tmp-{file_id}.png')
    return spec_db, img
