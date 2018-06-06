import os
import pickle

import numpy as np

from other.old_agent import run_agent
from supervised.classifiers import get_model
from util.preprocessor import car_ball
from util.data import key_vec


# Find ROI for ball
# Find ROI for car/dust
# Shuffle the data
# Compress the data
# Convert the key pressing to binary outputs.


def get_dataset():
    save_path_npy = 'data/trial_{}.npy'
    save_path_other = 'data/trial_{}.pkl'

    trials_imgs = []
    trials_metadata = []

    trial_num = 0
    while os.path.exists(save_path_npy.format(trial_num)) and os.path.exists(save_path_other.format(trial_num)):
        imgs = np.load(save_path_npy.format(trial_num))
        trials_imgs.append(imgs)
        num_frames = len(imgs)

        # List of frame time, key
        with open(save_path_other.format(trial_num), 'rb') as f:
            trials_metadata += pickle.load(f)
        print('Retrieved trial {}: {} frames'.format(trial_num, num_frames))

        trial_num += 1

    x = car_ball(trials_imgs)
    y = key_vec(trials_metadata)

    return x, y


print('Retrieving data')
x, y = get_dataset()  # Car, ball, keys

print('Getting Model')
model = get_model()

print('Training Model')
model.fit(x, y, epochs=10)

path = 'model_data/model_{}.h5'
model_num = 0
while os.path.exists(path.format(model_num)):
    model_num += 1
model.save(path.format(model_num))

print('Model {} Trained'.format(model_num))

print("Running agent")
run_agent()
