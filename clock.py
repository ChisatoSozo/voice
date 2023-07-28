import os
import threading
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from audio import get_audio_buffer_from_mic
from colormaps import alpha_to_blue_colormap, alpha_to_red_colormap
from grad_cam import make_guided_backprop
from resnet import make_model
from spectrogram import get_spectrogram

SECONDS = 10

shared_array = np.zeros((3, 256, 161 * SECONDS), dtype=np.float32)
male_gender_readings = np.zeros((SECONDS), dtype=np.float32)
female_gender_readings = np.zeros((SECONDS), dtype=np.float32)


def run_spectrogram():
    model = make_model()
    guided_backprop = make_guided_backprop(model)

    def callback(total_data, new_data):
        global shared_array, male_gender_readings, female_gender_readings
        spec_db, img = get_spectrogram(total_data, len(new_data))

        # push spec_db to the end of shared_array

        new_shared_array = np.zeros((3, 256, 161), dtype=np.float32)
        new_shared_array[0, :, :] = spec_db[:, :]

        if img is not None:
            gender = model(img.unsqueeze(0)).squeeze(0)
            male_grad_cam = guided_backprop(img, 1)
            female_grad_cam = guided_backprop(img, 0)
            male_gender_readings = np.roll(male_gender_readings, -1)
            male_gender_readings[-1] = gender[1].item() / 10
            female_gender_readings = np.roll(female_gender_readings, -1)
            female_gender_readings[-1] = gender[0].item() / 10
            new_shared_array[1, :, :] = female_grad_cam[:,
                                                        :] * female_gender_readings[-1]
            new_shared_array[2, :, :] = male_grad_cam[:, :] * \
                male_gender_readings[-1]

        else:
            male_gender_readings = np.roll(male_gender_readings, -1)
            male_gender_readings[-1] = np.sum(male_gender_readings) / \
                len(male_gender_readings)
            female_gender_readings = np.roll(female_gender_readings, -1)
            female_gender_readings[-1] = np.sum(
                female_gender_readings) / len(female_gender_readings)

        # append new_shared_array to shared_array, making the new shape (3, 256, 161 * (SECONDS + 1))
        shared_array = np.concatenate((shared_array, new_shared_array), axis=2)

    get_audio_buffer_from_mic(SECONDS, callback)


start_time = time.time()


def run_plot():
    global shared_array, start_time

    num_frames = 100
    interval = 1

    shared_array_1_blur = cv2.GaussianBlur(shared_array[1], (5, 5), 0)
    shared_array_2_blur = cv2.GaussianBlur(shared_array[2], (5, 5), 0)

    def update(frame):
        global shared_array, start_time
        has_new_data = shared_array.shape[2] > 161 * SECONDS
        if has_new_data:
            has_new_data = False
            # remove the first 161 columns of shared_array
            shared_array = shared_array[:, :, 161:]
            start_time = time.time()

        diff = time.time() - start_time
        start = int(diff * 161)
        end = start + int(161 * (SECONDS - 1.2))

        im_dat = shared_array[0, :, start:end]
        # flip im_dat vertically
        im_dat = np.flip(im_dat, axis=0)

        im.set_data(im_dat)
        shared_array_1_blur = cv2.GaussianBlur(shared_array[1], (5, 5), 0) * 3
        shared_array_2_blur = cv2.GaussianBlur(shared_array[2], (5, 5), 0) * 3
        im2.set_data(shared_array_1_blur[:, start:end])
        im3.set_data(shared_array_2_blur[:, start:end])
        male_gender_readings_avg = np.sum(
            male_gender_readings) / len(male_gender_readings)
        female_gender_readings_avg = np.sum(
            female_gender_readings) / len(female_gender_readings)
        for (bar, height) in zip(bar_chart, [male_gender_readings[-1], female_gender_readings[-1],
                                             male_gender_readings_avg, female_gender_readings_avg]):
            bar.set_height(height)
        #wait for the gui to update
        return [im, im2, im3, bar_chart[0], bar_chart[1], bar_chart[2], bar_chart[3]]
    

    def on_close(event):
        plt.close('all')
        # terminate the program
        os._exit(0)

    fig, (ax1, ax2) = plt.subplots(2)
    ax1.axis('off')
    fig.canvas.mpl_connect('close_event', on_close)
    red = alpha_to_red_colormap()
    blue = alpha_to_blue_colormap()
    # blur shared_array[1]

    im = ax1.imshow(shared_array[0], cmap='gray',
                    interpolation='nearest', animated=True, vmin=-100, vmax=0)
    im2 = ax1.imshow(shared_array_1_blur, cmap=red,
                     interpolation='nearest', animated=True, vmin=0, vmax=1)
    im3 = ax1.imshow(shared_array_2_blur, cmap=blue,
                     interpolation='nearest', animated=True, vmin=0, vmax=1)
    # bar graph of male and female scores
    male_gender_readings_avg = np.sum(
        male_gender_readings) / len(male_gender_readings)
    female_gender_readings_avg = np.sum(
        female_gender_readings) / len(female_gender_readings)
    bar_chart = ax2.bar(range(4), [-1, 1,
                                   -1, 1], color=['blue', 'pink', 'blue', 'pink'])
    ax2.bar_label(bar_chart, ["Male", "Female",
                  "Avg Male", "Avg Female"], padding=12)
    ax2.xaxis.set_visible(False)

    # Create the animation with repeat set to True
    ani = FuncAnimation(fig, update, frames=num_frames,
                        interval=interval, repeat=True, blit=True)

    plt.show()


# Create thread objects
spectrogram_thread = threading.Thread(target=run_spectrogram)


spectrogram_thread.start()
run_plot()
# Wait for both threads to complete (optional)
spectrogram_thread.join()

