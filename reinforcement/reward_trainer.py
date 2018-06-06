# get data
# scramble data
# combine data
# train

import numpy as np

from reinforcement import reward_models

from keras.callbacks import ModelCheckpoint

import gc


def main():
    ball_0s = np.load('data/ball_0s.npy').astype(np.float16)
    car_0s = np.load('data/car_0s.npy').astype(np.float16)
    ball_1s = np.load('data/ball_1s.npy').astype(np.float16)
    car_1s = np.load('data/car_1s.npy').astype(np.float16)

    num_0s = ball_0s.shape[0]
    num_1s = ball_1s.shape[0]

    total = num_0s + num_1s
    print('Data loaded, {} entries'.format(total))

    ball = np.concatenate([ball_0s, ball_1s])
    car = np.concatenate([car_0s, car_1s])
    labels = np.concatenate([np.zeros(num_0s), np.ones(num_1s)])

    del ball_0s, car_0s, ball_1s, car_1s
    gc.collect()

    c = np.c_[ball.reshape(len(ball), -1), car.reshape(len(car), -1), labels.reshape(len(labels), -1)]
    np.random.shuffle(c)
    ball_shuffled = c[:, :ball.size//len(ball)].reshape(ball.shape)
    car_shuffled = c[:, ball.size//len(ball):ball.size//len(ball) + car.size//len(car)].reshape(car.shape)
    labels_shuffled = c[:, ball.size//len(ball) + car.size//len(car):].reshape(labels.shape)
    del ball, car, labels, c
    gc.collect()

    class_weights = {
        0: num_1s/total,
        1: num_0s/total
    }

    for model_letter in 'BCF':

        model = eval('reward_model.get_model_{}()'.format(model_letter))
        filepath = "model_data/reward_function/" + model_letter + "-{val_acc:.3f}.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

        model.fit([ball_shuffled, car_shuffled], [labels_shuffled],
                  class_weight=class_weights, epochs=50, validation_split=0.1, callbacks=[checkpoint])
        print('Model {} fit'.format(model_letter))

if __name__ == '__main__':
    print('Starting')
    main()
