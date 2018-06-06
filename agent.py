import random

import numpy as np


class Agent:
    def __init__(self, model, epsilon, ball_obs_dims, car_obs_dims):
        self.model = model
        self.epsilon = epsilon
        self.action_space = (0, 1, 2)

        self.ball_obs_dims = ball_obs_dims
        self.car_obs_dims = car_obs_dims

        ball = np.zeros((1, ) + self.ball_obs_dims)
        car = np.zeros((1, ) + self.car_obs_dims)
        self.model.predict([ball, car])  # Initializes model

    def get_action(self, obs):
        ball, car = obs
        ball = ball.reshape((1, ) + self.ball_obs_dims)
        car = car.reshape((1, ) + self.car_obs_dims)
        pred = self.model.predict([ball, car])[0]
        print('\r', '{:>6.2f} {:>6.2f} {:>6.2f}'.format(*pred))
        if random.random() < self.epsilon:
            return random.choice(self.action_space)
        else:
            return np.argmax(pred)
