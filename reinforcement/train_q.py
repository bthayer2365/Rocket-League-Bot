import os
import numpy as np

from keras.models import load_model

from reinforcement import models  # Actually is used in eval statement

from util.crop import get_gray_batch
from util.preprocessor import normalize, normalize_gray
from util.data import get_all_streak_data


def train_model(model, model_code, ball_history, car_history, action_history, reward_history, batch_size=32):
    if os.path.exists('model_data/q_net/{}/last_epoch.txt'.format(model_code)):
        with open('model_data/q_net/{}/last_epoch.txt'.format(model_code), 'r') as e:
            epoch = int(e.read()) + 1
    else:
        epoch = 1

    p = np.random.permutation(len(reward_history))
    total_loss = 0
    num_batches = len(reward_history) // batch_size
    print("Training batch 0/{}".format(num_batches), end='')
    for i in range(num_batches):
        print("\rTraining batch {}/{}".format(i + 1, num_batches), end='')

        start_index = i * batch_size
        end_index = start_index + batch_size

        p_batch = p[start_index:end_index]

        ball_batch = ball_history[p_batch]
        car_batch = car_history[p_batch]
        action_batch = action_history[p_batch]
        reward_batch = reward_history[p_batch]

        preds = model.predict([ball_batch, car_batch])
        y = np.zeros((len(p_batch), 3))

        for j in range(len(p_batch)):
            if action_batch[j] == 0:
                y[j] = [reward_batch[j], preds[j][1], preds[j][2]]
            elif action_batch[j] == 1:
                y[j] = [preds[j][0], reward_batch[j], preds[j][2]]
            else:
                y[j] = [preds[j][0], preds[j][1], reward_batch[j]]

        total_loss += model.train_on_batch([ball_batch, car_batch], y) * len(p_batch)

    print()

    loss = total_loss / len(p)
    print('loss - {}'.format(loss))
    model.save('model_data/q_net/{}/{:03d}-{:.3f}.hdf5'.format(model_code, epoch, loss))
    model.save('model_data/q_net/{}/latest.hdf5'.format(model_code, epoch))

    with open('model_data/q_net/{}/last_epoch.txt'.format(model_code), 'w+') as e:
        e.write(str(epoch))


def main():

    model_code = 'E'
    gray = True
    if gray:
        model_code += '_gray'

    if not os.path.exists('model_data/q_net/{}'.format(model_code)):
        os.mkdir('model_data/q_net/{}'.format(model_code))

    if os.path.exists('model_data/q_net/{}/latest.hdf5'.format(model_code)):
        print('Loading latest model')
        model = load_model('model_data/q_net/{}/latest.hdf5'.format(model_code))
    else:
        print('Getting new model')
        model = eval('models.get_model_{}()'.format(model_code))

    num_frames = 30  # No shorter than 2 seconds
    gamma = 0.975
    start_epoch = 0
    end_epoch = 1000

    ball_frames, car_frames, actions, rewards = get_all_streak_data(num_frames, gamma)
    if gray:
        ball_frames = get_gray_batch(ball_frames)
        car_frames = get_gray_batch(car_frames)
        ball_frames, car_frames = normalize_gray(ball_frames, car_frames)
    else:
        ball_frames, car_frames = normalize(ball_frames, car_frames)

    for i in range(start_epoch, end_epoch):
        print("Starting epoch {}/{}".format(i+1, end_epoch))
        train_model(model, model_code, ball_frames, car_frames, actions, rewards)


if __name__ == '__main__':
    main()
