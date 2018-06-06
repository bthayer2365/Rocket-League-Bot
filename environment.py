import time
from collections import deque

import numpy as np
from PIL import ImageGrab

from util.button_presser import press_key, release_key

from util.dimensions import get_screen_bbox

from util.preprocessor import Preprocessor


class Environment:
    def __init__(self,
                 crop_style=0,
                 gray=False,
                 frame_time=1/8,
                 read_only=False,
                 normalize=True,
                 buffer_size=3,
                 field='champions',
                 reward_func=lambda x: 1,):
        self.field = field
        self.crop_style = crop_style
        if crop_style is None:
            self.prep = None
        else:
            self.prep = Preprocessor(crop_style, gray=gray, should_normalize=normalize, field=field)

            if gray:
                self.ball_obs_dims = self.prep.ball_dims + (buffer_size, )
                self.car_obs_dims = self.prep.car_dims + (buffer_size, )
            else:
                self.ball_obs_dims = self.prep.ball_dims + (3 * buffer_size, )
                self.car_obs_dims = self.prep.car_dims + (3 * buffer_size, )

        self.bboxes = (get_screen_bbox(), )
        self.next_frame = time.time()
        self.read_only = read_only
        self.frame_time = frame_time
        self.get_reward = reward_func

        self.gray = gray
        self.frame_buffer = deque(maxlen=buffer_size)  # Save last three frames

        self.frame_times = []
        self.obs_times = []

    def reset(self, read_only=None):
        self.frame_times = []
        self.obs_times = []

        self.fill_buffer()

        self.next_frame = time.time()
        # True False yes
        # True True no
        # False True no
        # False False yes
        if read_only is False or (self.read_only is False and read_only is not True):
            release_key('A')
            release_key('D')
            press_key('W')
            press_key('SHIFT')
        
        return self.step(read_only=read_only)[0]  # Use step instead of get_obs to initialize reward model

    def end(self):
        release_key('A')
        release_key('D')
        release_key('W')
        release_key('SHIFT')

    def fill_buffer(self):
        imgs = self.get_frame()
        for i in range(3):
            self.frame_buffer.append(imgs)

    def get_frame(self):
        img = np.array(ImageGrab.grab(bbox=get_screen_bbox()))
        if self.crop_style is None:
            return img
        else:
            return self.prep.process_frame(img)

    def get_observation(self):
        if self.crop_style is None:
            return np.concatenate([self.frame_buffer[i] for i in range(self.frame_buffer.maxlen)], axis=2)

        if self.gray:
            balls = np.stack([self.frame_buffer[i][0] for i in range(self.frame_buffer.maxlen)], axis=2)
            cars = np.stack([self.frame_buffer[i][1] for i in range(self.frame_buffer.maxlen)], axis=2)
        else:
            balls = np.concatenate([self.frame_buffer[i][0] for i in range(self.frame_buffer.maxlen)], axis=2)
            cars = np.concatenate([self.frame_buffer[i][1] for i in range(self.frame_buffer.maxlen)], axis=2)
        obs = balls, cars
        return obs

    def step(self, action=0, read_only=None):
        sleep_time = self.next_frame - time.time()
        if time.time() < self.next_frame:
            time.sleep(self.next_frame - time.time())
        self.next_frame = time.time() + self.frame_time

        if read_only is False or (self.read_only is False and read_only is not True):  # read_only overrides
            press_key('W')
            press_key('SHIFT')

            if action == 0:
                release_key('A')
                release_key('D')
            elif action == 1:
                press_key('A')
                release_key('D')
            elif action == 2:
                press_key('D')
                release_key('A')

        frame_start = time.time()
        imgs = self.get_frame()
        self.frame_times.append(time.time() - frame_start)

        r = self.get_reward(imgs)
        self.frame_buffer.append(imgs)

        obs_start = time.time()
        obs = self.get_observation()
        self.obs_times.append(time.time() - obs_start)

        # done and info are just so the env behaves similar to a gym env, but we don't use them
        done = False
        info = sleep_time

        return obs, r, done, info


def test():
    time.sleep(2)
    times = []
    env = Environment()
    env.reset()
    for i in range(15*20):
        start = time.time()
        env.step()
        times.append(time.time() - start)
    env.end()
    print(sum(env.frame_times) / len(env.frame_times))
    print(sum(env.obs_times) / len(env.obs_times))
    print(sum(times)/len(times))


if __name__ == '__main__':
    test()
