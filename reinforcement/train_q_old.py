import pickle

import gc

import numpy as np

from util.reward_funcs import ball_mean, ball_mse_std, ball_mse_mean
from util.crop import ball_crop_batch, car_crop_batch

from util.preprocessor import normalize
from util.data import key_vec

from reinforcement.models import get_model_A
from keras.models import load_model


def get_trial_data(trial_num, discount_factor=0.8, cutoff=30):
    trial = np.load('data/champions/trial_{}.npy'.format(trial_num))

    ball_crop = ball_crop_batch(trial)
    car_crop = car_crop_batch(trial)
    ball_crop, car_crop = normalize(ball_crop, car_crop)
    mses = np.mean(np.square(ball_crop - ball_mean), axis=(1, 2, 3))
    rewards = - (mses - ball_mse_mean) / ball_mse_std

    rewards_sum = np.zeros_like(rewards)

    r = rewards[-1] / (1 - discount_factor)  # assume things continued the way they were going
    for i in reversed(range(len(mses))):
        r = rewards[i] + r * discount_factor
        rewards_sum[i] = r

    X = ball_crop[:-cutoff], car_crop[:-cutoff]
    y = rewards_sum[:-cutoff]

    with open('data/champions/trial_{}.pkl'.format(trial_num), 'rb') as f:
        trial_keys = pickle.load(f)
        
    p = key_vec(trial_keys)[:-cutoff]

    return X, y, p


def get_data():
    X1s, X2s, ys, ps = [], [], [], []

    X, y, p = None, None, None

    for trial_num in range(6):
        X, y, p = get_trial_data(trial_num+1)  # p is the action that needs to be updated
        X1s.append(X[0])
        X2s.append(X[1])
        ys.append(np.expand_dims(y, -1))
        ps.append(p)

        print('Got trial {} data'.format(trial_num))
    del X, y, p
    gc.collect()

    # X1 = np.concatenate(X1s)
    # X2 = np.concatenate(X2s)
    # del X1s, X2s
    # gc.collect()
    #
    # y = np.concatenate(ys)
    # p = np.concatenate(ps)
    #
    # y = np.expand_dims(y, -1)
    
    return X1s, X2s, ys, ps


def shuffle_together(arrs):
    state = np.random.get_state()
    for arr in arrs:
        np.random.set_state(state)
        np.random.shuffle(arr)


def train():

    last_epoch = 443
    # model = get_model()
    model = load_model('model_data/3frames_{}.h5'.format(last_epoch))

    X1s, X2s, ys, ps = get_data()
    gc.collect()

    X1_shape = X1s[0].shape[1:]
    X2_shape = X2s[0].shape[1:]

    batch_size = 32
    start_epoch = last_epoch + 1
    epochs = 1000
    for epoch in range(start_epoch, start_epoch + epochs):
        print('Epoch {}'.format(epoch))

        # shuffle_together([X1, X2, y, p])

        # creates array like [[0, 0], [0, 1], [0, 2]....[1, 0],...
        sample_schedule = np.concatenate(
            [np.stack(
                [np.ones(len(X1s[sample])-2, dtype=np.int32) * sample,
                 np.arange(2, len(X1s[sample]), dtype=np.int32)]
                , axis=1)
                for sample in range(len(X1s))])
        np.random.shuffle(sample_schedule)

        batch_X1 = np.zeros((32, X1_shape[0], X1_shape[1], 9))
        batch_X2 = np.zeros((32, X2_shape[0], X2_shape[1], 9))
        batch_y = np.zeros((32, 1))
        batch_p = np.zeros((32, 3))

        losses = []
        for batch_num in range(sample_schedule.shape[0] // batch_size):
            start = batch_size * batch_num
            end = batch_size * (batch_num + 1)

            samples = sample_schedule[start:end]

            # bulid batch
            for i, sample in enumerate(samples):
                batch_X1[i] = np.concatenate([
                    X1s[sample[0]][sample[1]-2],
                    X1s[sample[0]][sample[1]-1],
                    X1s[sample[0]][sample[1]]], axis=2)
                batch_X2[i] = np.concatenate([
                    X2s[sample[0]][sample[1]-2],
                    X2s[sample[0]][sample[1]-1],
                    X2s[sample[0]][sample[1]]], axis=2)
                batch_y[i] = ys[sample[0]][sample[1]]
                batch_p[i] = ps[sample[0]][sample[1]]

            predictions = model.predict([batch_X1, batch_X2])
            targets = predictions * (1 - batch_p) + batch_y * batch_p

            loss = model.train_on_batch([batch_X1, batch_X2], targets)
            losses.append(loss)

        print("Training loss: {}".format(sum(losses)/len(losses)))

        model.save('model_data/3frames_{}.h5'.format(epoch))

if __name__ == '__main__':
    train()
