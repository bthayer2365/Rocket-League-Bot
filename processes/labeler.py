import time
import pickle

import numpy as np
import cv2

from util.get_keys import key_check


def create_labels(trial_num, start_frame=None):
    trial = np.load('data/champions/trial_{}.npy'.format(trial_num))
    print("Data loaded")

    labels = []

    if start_frame is None:
        try:
            with open('data/champions/trial_{}_labels.pkl'.format(trial_num), 'rb') as label_file:
                labels = pickle.load(label_file)
        except FileNotFoundError:
            pass
        start_frame = len(labels)
        print('Starting at frame {}'.format(start_frame))

    key = 0
    for frame_num, frame in enumerate(trial[start_frame:], start_frame):

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.imshow('cropped', img)
        cv2.waitKey(0)

        keys = []
        while not (len(keys) == 1 and ('0' == keys[0] or '1' == keys[0] or '5' == keys[0])):
            keys = key_check()
            time.sleep(0.1)

        cv2.destroyAllWindows()

        key = int(keys[0])

        if key is 5:
            break

        labels.append(key)
        print('{} {}/{}'.format(key, frame_num, len(trial)))

    with open('data/champions/trial_{}_labels.pkl'.format(trial_num), 'wb+') as label_file:
        pickle.dump(labels, file=label_file)
        print('Labels saved for trial {}'.format(trial_num))

    if key is 5:
        exit()


def main():
    for i in [1, 4, 5]:

        create_labels(i, None)


if __name__ == '__main__':
    main()
