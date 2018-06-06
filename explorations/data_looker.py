import pickle

trial_num = 2
trial_name = '../data/champions/trial_{}.pkl'.format(trial_num)
with open(trial_name, 'rb') as f:
    trial = pickle.load(f)

print(trial[0:30])


