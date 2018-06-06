import pickle

import numpy as np


def get_actions(trial_num):
    with open('data/champions/trial_{}.pkl'.format(trial_num), 'rb') as f:
        keys = pickle.load(f)

    left_pressed = np.array(['A' in key for key in keys])
    right_pressed = np.array(['D' in key for key in keys])

    left = np.logical_and(left_pressed, np.logical_not(right_pressed))
    right = np.logical_and(right_pressed, np.logical_not(left_pressed))

    return left * 1 + right * 2  # Creates single array with action numbers, 0 = forward, 1 = left, 2 = right


def get_labels(trial_num):
    with open('data/champions/trial_{}_labels.pkl'.format(trial_num), 'rb') as f:
        return pickle.load(f)


def build_trial(trial_num):
    trial_ball = np.load('data/champions/trial_{}_ball_normalized.npy'.format(trial_num))
    trial_car = np.load('data/champions/trial_{}_car_normalized.npy'.format(trial_num))

    ball_frames = np.concatenate((trial_ball[2:], trial_ball[1:-1], trial_ball[:-2]), axis=3)
    car_frames = np.concatenate((trial_car[2:], trial_car[1:-1], trial_car[:-2]), axis=3)
    actions = get_actions(trial_num)
    labels = get_labels(trial_num)

    return ball_frames, car_frames, actions, labels


def find_streaks(labels):
    # Streak starts 3 in and
    streaks = []
    streak_start = 0
    on_streak = False
    for i in range(len(labels)):
        if on_streak and labels[i] is 0:
            streaks.append((streak_start, i + 1))  # because we want range(streak_start, i + 1) to end on i
            on_streak = False
            continue

        if not on_streak and labels[i] is 1:
            streak_start = i
            on_streak = True
    if on_streak:
        streaks.append((streak_start, len(labels)))

    return streaks


def select_streaks(streaks, num_frames):
    indexes = []
    for i, streak in enumerate(streaks):
        streak_length = streak[1] - streak[0]
        if streak_length >= num_frames:
            indexes.append(i)
    return indexes


def build_streak_data(trial_num, num_frames, gamma):
    ball_frames, car_frames, actions, labels = build_trial(trial_num)
    streaks = find_streaks(labels)[:-1]  # Remove final streak because trial may have ended well
    streak_indexes = select_streaks(streaks, num_frames)
    indexes = []
    rewards = []
    for streak_index in streak_indexes:
        streak_start, streak_end = streaks[streak_index]
        indexes += list(range(streak_start, streak_end))

        streak_rewards = []
        reward = 0
        for i in reversed(range(streak_start, streak_end)):
            reward = labels[i] + gamma * reward
            streak_rewards.append(reward)

        rewards += reversed(streak_rewards)

    return ball_frames[indexes], car_frames[indexes], actions[indexes], rewards


def get_all_streak_data(num_frames, gamma):
    all_ball_frames = []
    all_car_frames = []
    all_actions = []
    all_rewards = []
    for trial_num in range(7):
        ball_frames, car_frames, actions, rewards = build_streak_data(trial_num, num_frames, gamma)

        all_ball_frames.append(ball_frames)
        all_car_frames.append(car_frames)
        all_actions.append(actions)
        all_rewards.append(rewards)

    return np.concatenate(all_ball_frames),\
           np.concatenate(all_car_frames),\
           np.concatenate(all_actions),\
           np.concatenate(all_rewards)


def key_vec(trials):
    # Convert metadata to one-hot key vector

    def convert(metadata):
        # keys = metadata[1]
        keys = metadata  # new way of doing it
        a = 'A' in keys
        d = 'D' in keys
        return [a == d, a and not d, d and not a]

    return np.array(list(map(convert, trials)))