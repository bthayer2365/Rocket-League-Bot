import time
import os
import pickle

import numpy as np
import cv2

from keras.models import load_model, clone_model

from environment import Environment
from util.get_keys import key_check

from reinforcement import models  # This is actually used, but in an eval statement
from agent import Agent


def discount_rewards(rewards, gamma, normalize=False):
    if normalize:
        max_reward = 1 / (1 - gamma)
        min_reward = 0
        reward_halfrange = (max_reward - min_reward) / 2
        reward_mid = (max_reward + min_reward) / 2

    # Initializes assuming future is the same as the last few frames
    total_reward = 0
    discounted = []
    for r in reversed(rewards):
        total_reward = r + gamma * total_reward
        if normalize:
            # Constrain to [-1, 1]
            discounted.append((total_reward - reward_mid) / reward_halfrange)
        else:
            discounted.append(total_reward)

    return discounted


class Trial:
    def __init__(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.start_time = time.time()


class Trainer:
    def __init__(self, agent_runner, num_epochs=1, should_print=False, show_mc_loss=False):

        self.agent_runner = agent_runner

        self.model = self.agent_runner.training_model

        self.should_print = should_print
        self.show_mc_loss = show_mc_loss

        self.finished = False

        self.num_batches = 0
        self.batch = 0
        self.p = np.zeros(1)
        self.preds = np.zeros(1)
        self.total_loss = 0

        if os.path.exists(self.agent_runner.model_path + 'last_epoch.txt'):
            with open(self.agent_runner.model_path + 'last_epoch.txt', 'r') as e:
                self.epoch = int(e.read()) + 1
        else:
            self.epoch = 1

        self.num_epochs = num_epochs
        self.end_epoch = self.epoch + num_epochs
        self.initialize_epoch()

        self.agent_runner.move_trials_to_history()

        if self.agent_runner.history_index == 0:
            self.finished = True
        else:
            balls = self.agent_runner.ball_history[:self.agent_runner.history_index]
            cars = self.agent_runner.car_history[:self.agent_runner.history_index]
            self.preds = self.model.predict([balls, cars])

    def print(self, *args, **kwargs):
        if self.should_print:
            print(*args, **kwargs)

    def initialize_epoch(self):

        self.p = np.random.permutation(min(self.agent_runner.history_index, self.agent_runner.history_length))
        self.total_loss = 0
        self.num_batches = len(self.p) // self.agent_runner.batch_size

    def add_epoch(self):
        self.finished = False
        self.num_epochs += 1
        self.end_epoch += 1

    def finish(self):
        self.print()
        if self.show_mc_loss:
            new_preds = self.model.predict(
                [
                    self.agent_runner.ball_history,
                    self.agent_runner.car_history
                ])
            new_preds = new_preds[np.arange(len(new_preds)), self.agent_runner.action_history]
            mc_loss = np.sqrt(np.sum(np.square(
                self.agent_runner.discounted_reward_history - new_preds))/len(new_preds))

            print('mc loss - {:.4f}'.format(mc_loss))

        training_loss = np.sqrt(self.total_loss / len(self.p))
        self.print('training loss - {:.4f}'.format(training_loss))

        self.model.save(
            self.agent_runner.model_path + '{:03d}-{:.3f}.hdf5'.format(self.epoch, training_loss))
        self.model.save(
            self.agent_runner.model_path + 'latest.hdf5'.format(self.epoch))

        with open(self.agent_runner.model_path + 'last_epoch.txt', 'w+') as e:
            e.write(str(self.epoch))

        self.finished = True
    
    def train_batch(self, batch):

        self.print("\rTraining batch {}/{}, epoch {}/{}".format(batch + 1, self.num_batches, self.epoch, self.end_epoch)
                   , end='')

        start_index = batch * self.agent_runner.batch_size
        end_index = start_index + self.agent_runner.batch_size

        p_batch = self.p[start_index:end_index]

        ball_batch = self.agent_runner.ball_history[p_batch]
        car_batch = self.agent_runner.car_history[p_batch]
        action_batch = self.agent_runner.action_history[p_batch]
        reward_batch = self.agent_runner.reward_history[p_batch]
        discounted_reward_batch = self.agent_runner.discounted_reward_history[p_batch]

        # batch_preds = preds[p]
        batch_preds = self.model.predict([ball_batch, car_batch])

        targets = np.zeros((len(p_batch), 3))

        if self.agent_runner.td != 0:
            # V(x_0) = r_0 + V(x_1)
            nexts = (p_batch + 1) % min(self.agent_runner.history_length, self.agent_runner.history_index)
            # next_balls = self.agent_runner.ball_history[nexts]
            # next_cars = self.agent_runner.car_history[nexts]
            next_actions = self.agent_runner.action_history[nexts]

            # next_preds = self.model.predict([next_balls, next_cars])
            next_preds = self.preds[nexts]
            td_reward_targets = reward_batch + self.agent_runner.gamma * next_preds[np.arange(len(p_batch)), next_actions]
            reward_targets = self.agent_runner.td * td_reward_targets + (1 - self.agent_runner.td) * discounted_reward_batch

            for i, reward in enumerate(reward_batch):
                if reward == 0:
                    reward_targets[i] = 0

        else:
            reward_targets = discounted_reward_batch

        for j in range(len(p_batch)):
            if action_batch[j] == 0:
                targets[j] = [reward_targets[j], batch_preds[j][1], batch_preds[j][2]]
            elif action_batch[j] == 1:
                targets[j] = [batch_preds[j][0], reward_targets[j], batch_preds[j][2]]
            else:
                targets[j] = [batch_preds[j][0], batch_preds[j][1], reward_targets[j]]

        self.total_loss += self.model.train_on_batch([ball_batch, car_batch], targets) * len(p_batch)

    def next_batch(self):
        if self.batch >= self.num_batches:
            self.epoch += 1
            self.batch = 0
            self.initialize_epoch()

        if self.epoch < self.end_epoch:
            self.train_batch(self.batch)
            self.batch += 1
        else:
            self.finish()
        
    def train(self, time_limit=float('inf')):

        end_time = time.time() + time_limit
        while time.time() < end_time:
            if self.finished:
                if time_limit != float('inf'):
                    time.sleep(end_time - time.time())
                return
            else:
                self.next_batch()


class AgentRunner:
    def __init__(self, model_code, gamma=0.975, field='champions', crop_style=0, gray=False, fps=8,
                 history_length=2700, train_interval=150, num_epochs=1, keep_training=False, td=1,
                 batch_size=32, epsilon_decay=0.9, epsilon_floor=1/16, decay_interval=10, initial_epsilon=None):
        # reward_model = reward_models.get_model_F()
        # reward_func = reward_models.create_reward_func(self.reward_model)

        self.model_code = model_code
        self.gray = gray

        self.fps = fps
        self.field = field
        self.crop_style = crop_style

        self.gamma = gamma

        self.env = Environment(frame_time=1/fps, gray=gray, field=field, crop_style=crop_style)

        self.history_length = history_length

        self.td = td  # Determine how much td/mc to use
        self.batch_size = batch_size

        self.epsilon_decay = epsilon_decay
        self.epsilon_floor = epsilon_floor
        self.decay_interval = decay_interval

        self.trial_count = 0
        
        self.model_path = 'model_data/{}/q_net/{}'.format(field, model_code)
        for designator in (crop_style, 'G' if gray else 'C', fps, int(gamma*1000)):
            self.model_path += '_{}'.format(designator)
        self.model_path += '/'

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        print('Model path: {}'.format(self.model_path))
        print()
        print('Field: {}'.format(self.field))
        print('Model code: {}'.format(self.model_code))
        print('Crop style: {}'.format(self.crop_style))
        print('Gray: {}'.format(self.gray))
        print('FPS: {}'.format(self.fps))
        print('Gamma: {}'.format(self.gamma))
        try:
            self.acting_model = load_model(self.model_path + 'latest.hdf5')
            self.training_model = load_model(self.model_path + 'latest.hdf5')
            print('Loaded model parameters from disk')
        except OSError as e:
            get_model = eval('models.get_model_{}'.format(self.model_code))
            self.acting_model = get_model(self.env.ball_obs_dims, self.env.car_obs_dims)
            self.training_model = get_model(self.env.ball_obs_dims, self.env.car_obs_dims)
            self.training_model.set_weights(self.acting_model.get_weights())
            print('Generated new parameters')

        self.trials = []

        if os.path.exists(self.model_path + 'history.pkl'):
            ball_history = np.load(self.model_path + 'ball_history.npy')
            car_history = np.load(self.model_path + 'car_history.npy')
            with open(self.model_path + 'history.pkl', 'rb') as history:
                action_history, reward_history, discounted_reward_history, history_index = pickle.load(history)

            old_len = len(reward_history)
            next_idx = history_index % old_len

            if old_len == self.history_length:
                self.ball_history = ball_history
                self.car_history = car_history
                self.action_history = action_history
                self.reward_history = reward_history
                self.discounted_reward_history = discounted_reward_history
                self.history_index = history_index
            else:
                self.ball_history = np.zeros((self.history_length,) + self.env.ball_obs_dims)
                self.car_history = np.zeros((self.history_length,) + self.env.car_obs_dims)

                self.action_history = np.zeros(self.history_length, dtype=np.int8)
                self.reward_history = np.zeros(self.history_length)
                self.discounted_reward_history = np.zeros(self.history_length)

                if history_index > old_len:
                    # Looped
                    # TODO Copy end of array first
                    if old_len - next_idx > self.history_length:
                        # new history length is shorter than first section of old history
                        self.ball_history[:] = ball_history[next_idx:next_idx+self.history_length]
                        self.car_history[:] = car_history[next_idx:next_idx+self.history_length]
                        self.action_history[:] = action_history[next_idx:next_idx+self.history_length]
                        self.reward_history[:] = reward_history[next_idx:next_idx+self.history_length]
                        self.discounted_reward_history[:] =\
                            discounted_reward_history[next_idx:next_idx+self.history_length]

                        self.history_index = history_index
                    else:
                        # new history length is longer than first section of old history
                        self.ball_history[:old_len - next_idx] = ball_history[next_idx:]
                        self.car_history[:old_len - next_idx] = car_history[next_idx:]
                        self.action_history[:old_len - next_idx] = action_history[next_idx:]
                        self.reward_history[:old_len - next_idx] = reward_history[next_idx:]
                        self.discounted_reward_history[:old_len - next_idx] =\
                            discounted_reward_history[next_idx:]

                        # copy second section, when new history is full or we run out of old history
                        stop = min(self.history_length - (old_len - next_idx), next_idx)

                        self.ball_history[old_len - next_idx:old_len] = ball_history[:stop]
                        self.car_history[old_len - next_idx:old_len] = car_history[:stop]
                        self.action_history[old_len - next_idx:old_len] = action_history[:stop]
                        self.reward_history[old_len - next_idx:old_len] = reward_history[:stop]
                        self.discounted_reward_history[old_len - next_idx:old_len] =\
                            discounted_reward_history[:stop]

                        self.history_index = old_len
                else:
                    # Not looped
                    stop = min(self.history_length, history_index)

                    self.ball_history[:stop] = ball_history[:stop]
                    self.car_history[:stop] = car_history[:stop]
                    self.action_history[:stop] = action_history[:stop]
                    self.reward_history[:stop] = reward_history[:stop]
                    self.discounted_reward_history[:stop] = discounted_reward_history[:stop]

                    self.history_index = stop
            if initial_epsilon is None:
                initial_epsilon = max(self.epsilon_floor, self.epsilon_decay ** (self.history_index/self.decay_interval))
        else:
            self.ball_history = np.zeros((self.history_length,) + self.env.ball_obs_dims)
            self.car_history = np.zeros((self.history_length,) + self.env.car_obs_dims)

            self.action_history = np.zeros(self.history_length, dtype=np.int8)
            self.reward_history = np.zeros(self.history_length)
            self.discounted_reward_history = np.zeros(self.history_length)
            self.history_index = 0

            if initial_epsilon is None:
                initial_epsilon = 1.0

        print("epsilon = {}".format(initial_epsilon))

        self.train_interval = train_interval
        self.num_epochs = num_epochs
        self.keep_training = keep_training
        self.trainer = Trainer(self, num_epochs=self.num_epochs)
        self.trainer.finished = True

        self.agent = Agent(self.acting_model, initial_epsilon, self.env.ball_obs_dims, self.env.car_obs_dims)

        self.frames_in_buffer = 0

        # Initializes reward models
        self.env.reset(read_only=True)

        self.playing = False

        print("Ready")

    def clear_history(self):
        if input('Are you sure you want to clear history? [Y]') != 'Y':
            print('History not cleared')
            return

        self.ball_history = np.zeros((self.history_length,) + self.env.ball_obs_dims)
        self.car_history = np.zeros((self.history_length,) + self.env.car_obs_dims)

        self.action_history = np.zeros(self.history_length, dtype=np.int8)
        self.reward_history = np.zeros(self.history_length)
        self.discounted_reward_history = np.zeros(self.history_length)
        self.history_index = 0

        self.trials = []

        print("History cleared")

    def save_final_frames(self):
        path = 'data/screenshots/finals/real_trials'
        for i, trial in enumerate(self.trials):
            car = trial.observations[-1][1][:, :, 6:9]

            car = cv2.cvtColor(car, cv2.COLOR_BGR2RGB)
            cv2.imwrite(path + '{}_car.jpg'.format(i), car)

    def open_menu(self):
        prompt = ''
        prompt += '1. Clear hi'
        response = input(prompt)
        if response == '1':
            self.clear_history()
        elif response == '2':
            self.save_final_frames()

    def end_trial(self, trial):

        self.playing = False
        self.env.end()
        print('\rTrial {} Over. t = {:.2f}s, frames = {}, total={}'
              .format(self.trial_count, time.time() - trial.start_time, len(trial.rewards),
                      self.history_index + self.frames_in_buffer))
        self.trial_count += 1

        if self.trial_count % self.decay_interval == 0 and self.agent.epsilon > self.epsilon_floor:
            self.agent.epsilon = max(self.agent.epsilon * self.epsilon_decay, self.epsilon_floor)
            print('Epsilon updated to {:.4f}'.format(self.agent.epsilon))

        self.trials.append(trial)
        self.frames_in_buffer += len(trial.rewards)

        print("{} frames since last train".format(self.frames_in_buffer))

    def remove_recent_trial(self):
        if len(self.trials) == 0:
            return
        num_frames = len(self.trials[-1].rewards)
        self.frames_in_buffer -= num_frames
        del self.trials[-1]

        print("Trial at index {} removed ({} frames)".format(len(self.trials), num_frames))

    def save_trial(self, trial):
        discounted_rewards = discount_rewards(trial.rewards, self.gamma)

        for (ball, car), action, reward, discounted_reward in \
                zip(trial.observations, trial.actions, trial.rewards, discounted_rewards):
            self.ball_history[self.history_index % self.history_length] = ball
            self.car_history[self.history_index % self.history_length] = car
            self.action_history[self.history_index % self.history_length] = action
            self.reward_history[self.history_index % self.history_length] = reward
            self.discounted_reward_history[self.history_index % self.history_length] = discounted_reward
            self.history_index += 1

    def move_trials_to_history(self):
        num_trials = len(self.trials)
        if num_trials == 0:
            return
        avg_frames = self.frames_in_buffer/num_trials

        for trial in self.trials:
            self.save_trial(trial)

        frames_per_trial = [len(trial.rewards) for trial in self.trials]
        counts = {}
        max_frames = 0
        for num_frames in frames_per_trial:
            if num_frames not in counts:
                counts[num_frames] = 0
            counts[num_frames] += 1
            max_frames = max(max_frames, num_frames)

        mode = 0
        mode_count = 0
        for frames, count in counts.items():
            if count > mode_count:
                mode_count = count
                mode = frames

        self.trials.clear()
        self.frames_in_buffer = 0

        print("{} trials added to history with avg_frames= {:.2f}, max_frames= {:.2f}, mode_frames= {:.2f}"
              .format(num_trials, avg_frames, max_frames, mode))

    def next_trainer(self):
        print('Last trainer trained {} epochs'.format(self.trainer.num_epochs))
        print('Creating new trainer')

        # Trainer copies current model, so model being acted on is being updated
        self.acting_model.set_weights(self.training_model.get_weights())
        self.trainer = Trainer(self, num_epochs=self.num_epochs)
        print('Trainer created')

    def store_history(self):
        # TODO 0s stored if hist_idx < hist_len
        np.save(self.model_path + 'ball_history.npy', self.ball_history)
        np.save(self.model_path + 'car_history.npy', self.car_history)
        with open(self.model_path + 'history.pkl', 'wb+') as history_file:
            pickle.dump((self.action_history, self.reward_history, self.discounted_reward_history, self.history_index),
                        history_file)
    
    def run(self):
        key_check()

        trial = None
        reward = 1

        while True:
            keys = key_check()

            if '1' in keys:
                # Start, continue and give reward of 1

                reward = 1

                if not self.playing:
                    # Start trial

                    trial = Trial()

                    trial.start_index = self.history_index
                    self.playing = True

                    trial.observation = self.env.reset()
            elif '0' in keys:
                # End trial as failure
                if self.playing:
                    # End and set final reward to 0
                    trial.rewards[-1] = 0
                    self.end_trial(trial)
            elif '9' in keys:
                # Remove most recent
                self.remove_recent_trial()
            elif '4' in keys:
                # Train or watch current trainer
                if self.trainer.finished:
                    self.next_trainer()

                print('Watching trainer')
                self.trainer.should_print = True
                self.trainer.train()
                key_check()
                print('Trainer finished')

            elif '5' in keys:
                # Decrease epsilon
                self.agent.epsilon *= self.epsilon_decay
                print('Epsilon updated to {:.4f}'.format(self.agent.epsilon))
            elif '6' in keys:
                # Increase epsilon
                self.agent.epsilon /= self.epsilon_decay
                print('Epsilon updated to {:.4f}'.format(self.agent.epsilon))
            elif '7' in keys:
                self.open_menu()
            elif '8' in keys:
                # End process
                print()
                self.move_trials_to_history()
                print('Saving history')
                self.store_history()
                print('Ending process')
                exit()

            if not self.playing:
                wait_time = 0.5
                if self.trainer.finished:
                    if self.frames_in_buffer >= self.train_interval:
                        start_time = time.time()
                        self.next_trainer()
                        time_left = time.time() - (start_time + wait_time)
                        if time_left > 0:
                            self.trainer.train(time_left)
                        continue
                    elif self.keep_training:
                        self.trainer.add_epoch()

                self.trainer.train(0.5)  # If it's done training, it just waits
                continue

            # Active trial
            action = self.agent.get_action(trial.observation)
            trial.observation, _, _, info = self.env.step(action)

            trial.observations.append(trial.observation)
            trial.actions.append(action)
            trial.rewards.append(reward)

            print('\r{:.2f}s'.format(time.time() - trial.start_time), end='')


def main():
    model_code = 'E'
    gray = False
    fps = 8
    gamma = 0.975
    field = 'champions'
    crop_style = 1
    history_length = 480
    train_interval = history_length
    # train_interval = float('infinity')
    num_epochs = 20
    keep_training = False
    epsilon_floor = 1/8

    runner = AgentRunner(model_code=model_code,
                         gray=gray,
                         fps=fps,
                         gamma=gamma,
                         field=field,
                         crop_style=crop_style,
                         history_length=history_length,
                         train_interval=train_interval,
                         num_epochs=num_epochs,
                         keep_training=keep_training,
                         epsilon_floor=epsilon_floor,)
    runner.run()


if __name__ == '__main__':
    main()
