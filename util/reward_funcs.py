import numpy as np

from util.preprocessor import Preprocessor

ball_mean = None
ball_mse_mean = None
ball_mse_std = None


def ball_mse(imgs):
    # imgs = [ball_img, car_img]
    img = imgs[0]
    mse = np.mean(np.square(ball_mean - img))

    mse_norm = (mse - ball_mse_mean)/ball_mse_std

    # Reward a lower error
    return -mse_norm


def init():
    prep = Preprocessor()
    trial = np.load('data/trial_{}.npy'.format(1))
    ball_crop = prep.get_ball_batch(trial).astype(np.float32)

    global ball_mean, ball_mse_mean, ball_mse_std

    ball_mean = np.mean(ball_crop, axis=0)
    mses = np.mean(np.square(ball_crop - ball_mean), axis=(1, 2, 3))
    ball_mse_mean = mses.mean()
    ball_mse_std = mses.std()

init()
