import time

import numpy as np
import cv2

from util.get_keys import key_check
from util.preprocessor import Preprocessor
from util.data import get_labels


trial = np.load('data/champions/trial_0.npy')  # TODO Do this in a method
labels = get_labels(0)


def show(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imshow('car', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def save(i, ball, car, gray=False):
    path = 'data/screenshots/finals/'
    if not gray:
        ball = cv2.cvtColor(ball, cv2.COLOR_BGR2RGB)
        car = cv2.cvtColor(car, cv2.COLOR_BGR2RGB)
    cv2.imwrite(path + '{}_ball.jpg'.format(i), ball)  # TODO Add gray
    cv2.imwrite(path + '{}_car.jpg'.format(i), car)
    print("Img {} saved".format(i))


def save_full(i, img):
    path = 'data/screenshots/'
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(path + '{}_full.jpg'.format(i), img)
    print("Img {} saved".format(i))


def main():
    prep = Preprocessor(crop_style=1, should_normalize=False)
    for i, frame in enumerate(trial):
        print(i)
        ball, car = prep.get_ball_and_car(frame)
        show(car)
        cv2.waitKey(0)

        keys = key_check()
        while not (len(keys) == 1 and ('0' in keys or '1' in keys)):
            time.sleep(0.1)
            keys = key_check()

        if '1' in keys:
            save(i, ball, car)

        cv2.destroyAllWindows()


def save_specific(to_save, full=False, gray=False):
    prep = Preprocessor(crop_style=1, gray=gray, should_normalize=False)
    for i in to_save:
        if full:
            save_full(i, trial[i])
        else:
            ball, car = prep.process_frame(trial[i])
            save(i, ball, car, gray)


def save_on_change():  # TODO pass in trial
    to_save = []
    last_label = 0
    for i, label in enumerate(labels):
        if last_label == 1 and label == 0:
            to_save.append(i)
        last_label = label
    save_specific(to_save)


if __name__ == '__main__':
    # main()
    # save_specific([141], full=False, gray=True)
    save_on_change()
