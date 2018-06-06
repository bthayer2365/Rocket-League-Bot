import numpy as np

import cv2

from util.preprocessor import Preprocessor


def show(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imshow('thing', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def whole_image(trial):
    trial_mean = trial.mean(axis=0).astype(np.uint8)

    # mean mean squared error
    mses = np.mean(np.square(trial - trial_mean), axis=(1, 2, 3))
    print(np.mean(mses))
    print(np.std(mses))
    print(mses.min())
    print(mses.max())

    minframe = mses.argmin()
    maxframe = mses.argmax()

    show(trial[minframe])
    show(trial_mean)

    show(trial[maxframe])
    show(trial_mean)

    # best_to_worst = np.argsort(mses)
    #
    # for i, index in enumerate(best_to_worst):
    #     print(i)
    #     show(trial[index])
    #


def get_ball_trial_mean(trial):

    y_start_ball = 230
    y_width_ball = 80
    x_start_ball = 252
    x_width_ball = 96

    crop_ball = trial[:,
                      480 - y_start_ball - y_width_ball:480 - y_start_ball,
                      x_start_ball:x_start_ball + x_width_ball]

    trial_mean = crop_ball.mean(axis=0).astype(np.uint8)
    return trial_mean


def batch_gray(trial):
    gray = np.zeros(trial.shape[:-1])
    for i, frame in enumerate(trial):
        gray[i] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return gray


def ball_crop(trial):
    #
    # y_start_ball = 150
    # y_width_ball = 160
    # x_start_ball = 220
    # x_width_ball = 128

    y_start_ball = 230
    y_width_ball = 80
    x_start_ball = 252
    x_width_ball = 96

    crop_ball = trial[:,
                      480 - y_start_ball - y_width_ball:480 - y_start_ball,
                      x_start_ball:x_start_ball + x_width_ball]

    gray_crop = batch_gray(crop_ball).astype(np.uint8)
    
    gray_mean = gray_crop.mean(axis=0).astype(np.uint8)

    trial_mean = crop_ball.mean(axis=0).astype(np.uint8)
    
    gray_mses = np.mean(np.square(gray_crop - gray_mean), axis=(1, 2))
    mses = np.mean(np.square(crop_ball - trial_mean), axis=(1, 2, 3))
    print(np.mean(mses))
    print(np.std(mses))
    print(mses.min())
    print(mses.max())
    
    print()
    print('Grays')
    print(np.mean(gray_mses))
    print(np.std(gray_mses))
    print(gray_mses.min())
    print(gray_mses.max())
    
    minframe = mses.argmin()
    maxframe = mses.argmax()

    show(crop_ball[minframe])
    show(trial_mean)

    show(crop_ball[maxframe])
    show(trial_mean)

    best_to_worst = np.argsort(mses)

    # for i, index in enumerate(best_to_worst):
    #     if i % 1 == 0:
    #         print(i, mses[index])
    #         show(trial[index])
    #         # print(i, mses[i])
    #         # show(trial[i])

    def show_gray(img):
        cv2.imshow('cropped', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    gray_minframe = gray_mses.argmin()
    gray_maxframe = gray_mses.argmax()
    show_gray(gray_crop[gray_minframe].reshape(gray_mean.shape))
    show_gray(gray_mean)
    show_gray(gray_crop[gray_maxframe].reshape(gray_mean.shape))
    show_gray(gray_mean)


if __name__ == '__main__':
    prep = Preprocessor(crop_style=1)
    # 13 trials
    for trial_num in range(7):
        print('Displaying trial {}'.format(trial_num))
        trial = np.load('data/champions/trial_{}.npy'.format(trial_num))
        show(trial[0])
        ball, car = prep.get_ball_and_car(trial[0])
        show(ball)
        show(car)


    # ball_crop(trial)
