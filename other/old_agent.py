import os
import time

import numpy as np
from PIL import ImageGrab
from keras.models import load_model

from util.button_presser import PressKey, ReleaseKey, W, A, D, SHIFT
from util.get_keys import key_check


def get_model(model_num=None):
    path = 'model_data/model_{}.h5'
    if model_num is not None:
        return load_model('model_data/model_{}.h5'.format(model_num))

    model_num = 0
    while os.path.exists(path.format(model_num)):
        model_num += 1
    model_num -= 1  # Last one doesn't exist, so go back

    return load_model('model_data/model_{}.h5'.format(model_num))


def run_agent():

    model = get_model()

    key_check()  # Clears buffer

    frame_start = None
    playing = False

    print('Ready')
    while True:

        keys = key_check()

        if '0' in keys:
            print('Ending process')
            break

        if not playing and '8' in keys:
            print('Starting Playing')
            print()
            playing = True
            frame_start = time.time()
            recording_start = time.time()

        elif playing and '9' in keys:
            ReleaseKey(W)
            ReleaseKey(A)
            ReleaseKey(D)
            ReleaseKey(SHIFT)
            print("Stopping Playing at {0:.2f}s".format(time.time() - recording_start))
            playing = False

        if not playing:
            time.sleep(0.5)
            continue

        print('\rPlaying: {0:.2f}s'.format(time.time() - recording_start), end='\n')

        t1 = time.time()
        img = ImageGrab.grab(bbox=(640, 300, 640+640, 300+480))
        img_np = np.array(img)

        y_start_car = 65
        y_width_car = 90
        x_start_car = 260
        x_width_car = 120

        y_start_ball = 150
        y_width_ball = 160
        x_start_ball = 220
        x_width_ball = 125

        crop_car = img_np[480 - y_start_car - y_width_car:480 - y_start_car,
                          x_start_car:x_start_car + x_width_car]

        crop_ball = img_np[480 - y_start_ball - y_width_ball:480 - y_start_ball,
                           x_start_ball:x_start_ball + x_width_ball]

        keys = model.predict([crop_car.reshape(1, y_width_car, x_width_car, 3),
                              crop_ball.reshape(1, y_width_ball, x_width_ball, 3)])[0]
        print(keys)
        key = np.argmax(keys)

        PressKey(W)
        PressKey(SHIFT)

        if key == 0:
            ReleaseKey(A)
            ReleaseKey(D)
        elif key == 1:
            PressKey(A)
            ReleaseKey(D)
        else:
            PressKey(D)
            ReleaseKey(A)

        frame_time = time.time() - frame_start
        frame_start = time.time()

if __name__ == '__main__':
    run_agent()
