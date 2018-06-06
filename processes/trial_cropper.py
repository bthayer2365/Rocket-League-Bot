import pickle

import numpy as np

from util.crop import ball_crop_batch, car_crop_batch, ball_dims, car_dims
from util.preprocessor import ball_mean, ball_std, car_mean, car_std


def crop(trial_num):
    trial = np.load('data/champions/trial_{}.npy'.format(trial_num))

    ball_crop = ball_crop_batch(trial)
    car_crop = car_crop_batch(trial)

    np.save('data/champions/trial_{}_ball.npy'.format(trial_num), ball_crop)
    np.save('data/champions/trial_{}_car.npy'.format(trial_num), car_crop)


def normalize(trial_num, ball_mean, ball_std, car_mean, car_std):
    ball = np.load('data/champions/trial_{}_ball.npy'.format(trial_num))
    car = np.load('data/champions/trial_{}_car.npy'.format(trial_num))

    ball_normalized = (ball - ball_mean) / ball_std
    car_normalized = (car - car_mean) / car_std

    np.save('data/champions/trial_{}_ball_normalized.npy'.format(trial_num), ball_normalized)
    np.save('data/champions/trial_{}_car_normalized.npy'.format(trial_num), car_normalized)


def split_labels(trial_num):
    with open('data/champions/trial_{}_labels.pkl'.format(trial_num), 'rb') as label_file:
        labels = np.array(pickle.load(label_file)).astype(np.bool)

    ball = np.load('data/champions/trial_{}_ball_normalized.npy'.format(trial_num))
    car = np.load('data/champions/trial_{}_car_normalized.npy'.format(trial_num))

    ball_ones = ball[labels]
    car_ones = car[labels]

    ball_zeros = ball[np.invert(labels)]
    car_zeros = car[np.invert(labels)]

    np.save('data/champions/trial_{}_ball_normalized_label_1.npy'.format(trial_num), ball_ones)
    np.save('data/champions/trial_{}_car_normalized_label_1.npy'.format(trial_num), car_ones)

    np.save('data/champions/trial_{}_ball_normalized_label_0.npy'.format(trial_num), ball_zeros)
    np.save('data/champions/trial_{}_car_normalized_label_0.npy'.format(trial_num), car_zeros)


def merge_labels():
    ball_ones_by_trial = []
    car_ones_by_trial = []

    ball_zeros_by_trial = []
    car_zeros_by_trial = []

    for trial_num in range(7):
        print('Trial {} loaded'.format(trial_num))
        ball_1 = np.load('data/champions/trial_{}_ball_normalized_label_1.npy'.format(trial_num))
        car_1 = np.load('data/champions/trial_{}_car_normalized_label_1.npy'.format(trial_num))

        ball_0 = np.load('data/champions/trial_{}_ball_normalized_label_0.npy'.format(trial_num))
        car_0 = np.load('data/champions/trial_{}_car_normalized_label_0.npy'.format(trial_num))

        ball_ones_by_trial.append(ball_1)
        car_ones_by_trial.append(car_1)

        ball_zeros_by_trial.append(ball_0)
        car_zeros_by_trial.append(car_0)

    all_ball_ones = np.concatenate(ball_ones_by_trial)
    all_car_ones = np.concatenate(car_ones_by_trial)

    all_ball_zeros = np.concatenate(ball_zeros_by_trial)
    all_car_zeros = np.concatenate(car_zeros_by_trial)

    np.save('data/ball_1s.npy', all_ball_ones)
    np.save('data/car_1s.npy', all_car_ones)
    np.save('data/ball_0s.npy', all_ball_zeros)
    np.save('data/car_0s.npy', all_car_zeros)


def run_job():

    # for trial_num in range(7):
    #     crop(trial_num)
    #     print('Crop {} saved'.format(trial_num))
    #
    # print('Stats gathered')
    #
    # for trial_num in range(7):
    #     normalize(trial_num, ball_mean, ball_std, car_mean, car_std)
    #     print('Trial {} normaliized'.format(trial_num))

    for trial_num in range(0, 7):
        split_labels(trial_num)
        print('Labels for trial {} split'.format(trial_num))

    print('Merging labels')
    merge_labels()

    print('Complete')


def main():
    run_job()

    ball_0s = np.load('data/ball_0s.npy')
    ball_1s = np.load('data/ball_1s.npy')
    car_0s = np.load('data/car_0s.npy')
    car_1s = np.load('data/car_1s.npy')

    print(ball_1s.shape)
    print(ball_0s.shape)
    print(car_1s.shape)
    print(car_0s.shape)

    assert ball_0s.shape[0] == car_0s.shape[0]
    assert ball_1s.shape[0] == car_1s.shape[0]

    num_0s = ball_0s.shape[0]
    num_1s = ball_1s.shape[0]

    print(num_0s)
    print(num_1s)
    print(num_0s/(num_1s + num_0s))
    print(num_1s/(num_1s + num_0s))

if __name__ == '__main__':
    main()
