import os
import pickle
import time

import numpy as np
from PIL import ImageGrab

from util.get_keys import key_check

from environment import Environment


# TODO read Deep Reinforcement Learning from Human Preferences https://arxiv.org/pdf/1706.03741.pdf
# TODO Look into better video capture/ MMS


def save(frames_img, frames_other, field, trial_num):

    data_dir = 'data/{}/'.format(field)

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    save_path_npy = data_dir + 'trial_{}.npy'
    save_path_other = data_dir + 'trial_{}.pkl'

    while os.path.exists(save_path_npy.format(trial_num)) or os.path.exists(save_path_other.format(trial_num)):
        trial_num += 1

    save_path_npy = save_path_npy.format(trial_num)
    save_path_other = save_path_other.format(trial_num)
    np.save(save_path_npy, np.array(frames_img))

    with open(save_path_other, 'wb+') as f:
        pickle.dump(frames_other, f)

    print("Trail {} saved, {} frames".format(trial_num, len(frames_other)))


def main():
    field = 'champions_carcam'
    env = Environment(crop_style=None, normalize=False, read_only=True, buffer_size=1, field=field, frame_time=1/8)
    env.reset()

    save_trials = False

    frames_img = []
    frames_other = []

    recording = False
    start_time = time.time()
    trial_num = 0
    num_frames = 0
    frames = []

    key_check()
    print('Ready')

    while True:

        keys = key_check()

        if '8' in keys and not recording:
            # Start recording
            print('Starting Recording for trial {}'.format(trial_num))
            recording = True
            start_time = time.time()
            frames_img = []
            frames_other = []
        elif '9' in keys and recording:
            # End recording
            print('\rStopped Recording at {0:.2f}s'.format(time.time() - start_time))
            recording = False
            trial_num += 1
            frames.append(len(frames_other))
            num_frames += len(frames_other)
            if save_trials:
                print('Saving')
                save(frames_img, frames_other, field, trial_num)
                print('Done')
        elif '0' in keys:
            print(frames)
            print('Ending process')
            exit()
        elif '5' in keys:
            print('{} frames average over {} trials'.format(num_frames / trial_num, trial_num))

        if not recording:
            time.sleep(0.5)
            continue

        print('\rRecording: {0:.2f}s'.format(time.time() - start_time), end='')

        frame, _, _, info = env.step()
        frames_img.append(frame)
        frames_other.append(keys)


if __name__ == '__main__':
    main()
