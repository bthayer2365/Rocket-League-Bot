import os
import pickle

import numpy as np

import cv2

from util.dimensions import get_dims, get_bboxes


def crop_batch(x_start, y_start, x_end, y_end, imgs):
    return imgs[:, y_start:y_end, x_start:x_end]


def crop(x_start, y_start, x_end, y_end, img):
    return img[y_start:y_end, x_start:x_end]


def to_gray_batch(imgs):
    grays = np.zeros(imgs.shape[:-1])
    for i, img in enumerate(imgs):
        grays[i] = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return grays


class Preprocessor:
    def __init__(self, crop_style=0, gray=False, should_normalize=True, field='champions'):
        self.crop_style = crop_style
        self.ball_dims, self.ball_start, self.car_dims, self.car_start = get_dims(crop_style)
        self.ball_bbox, self.car_bbox = get_bboxes(crop_style)

        self.gray = gray
        self.should_normalize = should_normalize

        self.field = field
        if field == 'champions':
            self.num_trials = 7
        else:
            self.num_trials = 2

        self.ball_mean, self.ball_std, self.car_mean, self.car_std = self.get_stats()

    def get_stats(self, load=True, save=True):
        file_path = 'data/{}/stats_{}_{} ({}).pkl'.format(
            self.field, self.crop_style, 'G' if self.gray else 'C', self.num_trials)

        if load and os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                return pickle.load(f)

        print('Generating stats')

        ball_stat_dims = (self.num_trials,) + self.ball_dims
        car_stat_dims = (self.num_trials,) + self.car_dims

        if not self.gray:
            ball_stat_dims += (3, )
            car_stat_dims += (3, )

        ball_means = np.zeros(ball_stat_dims, dtype=np.float32)
        ball_vars = np.zeros(ball_stat_dims, dtype=np.float32)

        car_means = np.zeros(car_stat_dims, dtype=np.float32)
        car_vars = np.zeros(car_stat_dims, dtype=np.float32)

        weights = np.zeros((self.num_trials,), dtype=np.float32)

        for trial_num in range(self.num_trials):
            trial = np.load('data/{}/trial_{}.npy'.format(self.field, trial_num))
            ball_crop, car_crop = self.get_ball_and_car_batch(trial)

            if self.gray:
                ball_crop = to_gray_batch(ball_crop)
                car_crop = to_gray_batch(car_crop)

            # Weight trials so that all frames have equal weight
            weights[trial_num] = ball_crop.shape[0]
            assert ball_crop.shape[0] == car_crop.shape[0]

            ball_mean = np.mean(ball_crop, axis=0)
            ball_var = np.var(ball_crop, axis=0)

            car_mean = np.mean(car_crop, axis=0)
            car_var = np.var(car_crop, axis=0)

            del ball_crop
            del car_crop

            ball_means[trial_num] = ball_mean
            ball_vars[trial_num] = ball_var

            car_means[trial_num] = car_mean
            car_vars[trial_num] = car_var

            print('Trial {} complete'.format(trial_num))

        weights /= np.sum(weights)

        ball_mean = np.average(ball_means, 0, weights)
        ball_var = np.average(ball_vars, 0, weights)
        ball_std = np.sqrt(ball_var)

        car_mean = np.average(car_means, 0, weights)
        car_var = np.average(car_vars, 0, weights)
        car_std = np.sqrt(car_var)

        if save:
            with open(file_path, 'wb+') as f:
                pickle.dump((ball_mean, ball_std, car_mean, car_std), f)

        return ball_mean, ball_std, car_mean, car_std

    def reset_stats(self):
        self.ball_mean, self.ball_std, self.car_mean, self.car_std = self.get_stats()

    def get_ball(self, img):
        return crop(*self.ball_bbox, img)

    def get_car(self, img):
        return crop(*self.car_bbox, img)

    def get_ball_and_car(self, img):
        return self.get_ball(img), self.get_car(img)

    def get_ball_batch(self, imgs):
        return crop_batch(*self.ball_bbox, imgs)

    def get_car_batch(self, imgs):
        return crop_batch(*self.car_bbox, imgs)

    def get_ball_and_car_batch(self, imgs):
        return self.get_ball_batch(imgs), self.get_car_batch(imgs)

    def normalize_ball(self, ball):
        return (ball - self.ball_mean)/self.ball_std

    def normalize_car(self, car):
        return (car - self.car_mean)/self.car_std

    def normalize(self, ball, car):
        return [self.normalize_ball(ball), self.normalize_car(car)]

    def unnormalize_ball(self, ball):
        return ball * self.ball_std + self.ball_mean

    def unnormalize_car(self, car):
        return car * self.car_std + self.car_mean

    def unnormalize(self, ball, car):
        return [self.unnormalize_ball(ball), self.unnormalize_car(car)]

    def process_frame(self, img):
        ball, car = self.get_ball_and_car(img)

        if self.gray:
            ball = cv2.cvtColor(ball, cv2.COLOR_BGR2GRAY)
            car = cv2.cvtColor(car, cv2.COLOR_BGR2GRAY)

        if self.should_normalize:
            return self.normalize(ball, car)
        else:
            return ball, car

    def process_frames(self, imgs):
        balls, cars = self.get_ball_and_car_batch(imgs)

        if self.gray:
            balls = to_gray_batch(balls)
            cars = to_gray_batch(cars)
        if self.should_normalize:
            return self.normalize(balls, cars)
        else:
            return balls, cars


def main():
    Preprocessor()
    Preprocessor(gray=True)


if __name__ == '__main__':
    main()
