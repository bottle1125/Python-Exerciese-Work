import random
import numpy as np
import pandas as pd
import math
import tabulate
import matplotlib.pyplot as plt
from itertools import product
from IPython.display import HTML, display


def __init__(self, rows, cols, log_p=False):
    self.rows = rows
    self.cols = cols
    self.actions = list(product(*[[row for row in range(rows)],
                                  [column for column in range(cols)]]))
    self.cell_distance = 6  # how many visual degrees one cell length is in the device
    self.found_reward = 20  # how much finding the target gives reward

    # Use noise for encoding?
    self.encoding_noise = True

    # The visual search model calculates a lot of distances with a fixed
    # number of parameters. Makes it faster to tabulate.
    self.distances = {}

    # Task type. 0 means that the search target is randomly among
    # the items. 1 means that the target is always the last
    # item that is looked at.
    self.task_type = 0

    self.belief = repr(["No Info", "No Info"])
    self.previous_belief = self.belief

    # additional variable to track model statistics.
    self.model_time = 0
    self.total_rewards = 0
    self.total_search_time = 0
    self.fixation_count = 0

    # RL
    self.alpha = 0.1
    self.gamma = 0.9
    self.softmax_temp = 1.0

    self.learning = True

    self.q = {}

    # log configurations.
    self.log_header = "trial task.type encoding model.time mt n.fix reward"
    self.stat_recorder = []
    self.log = []
    self.log_p = log_p

    self.trial = 0


def randomise_search_device(self):
    self.device = np.zeros([self.rows, self.cols], dtype=int)
    self.task_type = random.choice([0, 1])
    if self.task_type == 0:
        row = random.randint(0, self.rows - 1)
        col = random.randint(0, self.cols - 1)
        self.device[row][col] = 1
        self.target = [row, col]
    else:
        self.target = None
    return self.device


def set_belief_state(self):
    # Must turn belief state into string, otherwise too complicated structure.
    self.belief = repr([self.eye_loc, self.obs])
    # Update Q
    if self.belief not in self.q:
        self.q[self.belief] = {}
        for action in self.actions:
            self.q[self.belief][action] = 0.0


def clear(self, starting_location = None):
    self.trial += 1
    self.encoding_n = 0

    self.randomise_search_device()
    self.obs = obs = np.full([self.rows, self.cols], -1, dtype=int)
    self.found = False
    self.reward = 0
    self.mt = 0

    self.total_rewards = 0
    self.total_search_time = 0
    self.fixation_count = 0

    # Start with the given, or a random, eye location but do not
    # observe that location. Reward for the start is 0.
    if starting_location == None:
        self.action = random.choice(self.actions)
    else:
        self.action = self.actions[starting_location]
    self.previous_action = self.action
    self.eye_loc = self.action
    self.eyes_moved = False

    self.set_belief_state()
    self.reward = 0


def observe(self):
    if self.action:
        self.obs[self.action[0], self.action[1]] = self.device[self.action[0], self.action[1]]
    return self.obs


def visual_distance(self, start, end, user_distance):
    """
    Calculate visual distance, as degrees, between two screen
    coordinates. User distance needs to be given in the same unit.
    """
    dist = math.sqrt(pow(start[0] - end[0], 2) + pow(start[1] - end[1], 2))
    return 180 * (math.atan(dist / user_distance) / math.pi)


def distance(self, x1, y1, x2, y2):
    """
    Euclidian distance between two points. Use lookup-table for reference
    or update it with a new entry.
    """
    if (x1, y1, x2, y2) not in self.distances:
        self.distances[(x1, y1, x2, y2)] = math.sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2))
    return self.distances[(x1, y1, x2, y2)]


def EMMA_fixation_time(self, distance, freq=0.1, encoding_noise=False):
    """
    Eye movement and encoding time come from EMMA (Salvucci, 2001). Also
    return if a fixation occurred.
    """
    emma_KK = 0.006
    emma_k = 0.4
    emma_prep = 0.135
    emma_exec = 0.07
    emma_saccade = 0.002
    E = emma_KK * -math.log(freq) * math.exp(emma_k * distance)
    if encoding_noise:
        E += np.random.gamma(E, E / 3)
    if E < emma_prep: return E, False
    S = emma_prep + emma_exec + emma_saccade * distance
    if E <= S: return S, True
    E_new = (emma_k * -math.log(freq))
    if encoding_noise:
        E_new += np.random.gamma(E_new, E_new / 3)
    T = (1 - (S / E)) * E_new
    return S + T, True


def weighted_random(weights):
    number = random.random() * sum(weights.values())
    for k, v in weights.items():
        if number < v:
            break
    return k


def calculate_reward(self):
    # Target found.
    if self.device[self.action[0], self.action[1]] == 1:
        self.mt += 0.15  # add motor movement time for response
        self.reward = self.found_reward - self.mt
        self.found = True
    elif self.task_type == 1 and self.obs.mean() == 0:
        # All elements uncovered (no target present).
        self.mt += 0.15  # add motor movement time for response
        self.reward = self.found_reward - self.mt
        self.found = True
    else:
        # Target not yet found.
        self.reward = -self.mt

    self.model_time += self.mt
    self.total_rewards += self.reward


def choose_action_softmax(self):
    p = {}
    for a in self.q[self.belief].keys():
        p[a] = math.exp(self.q[self.belief][a] / self.softmax_temp)
    s = sum(p.values())
    print(5555, p)
    if s != 0:
        p = {k: v / s for k, v in p.items()}
        self.action = weighted_random(p)
    else:
        self.action = np.random.choice(list(p.keys()))


def calculate_reward(self):
    # Target found.
    if self.device[self.action[0], self.action[1]] == 1:
        self.mt += 0.15  # add motor movement time for response
        self.reward = self.found_reward - self.mt
        self.found = True
    elif self.task_type == 1 and self.obs.mean() == 0:
        # All elements uncovered (no target present).
        self.mt += 0.15  # add motor movement time for response
        self.reward = self.found_reward - self.mt
        self.found = True
    else:
        # Target not yet found.
        self.reward = -self.mt

    self.model_time += self.mt
    self.total_rewards += self.reward


def current_q(self):
    return self.q[self.belief][self.action]


def weighted_random(weights):
    number = random.random() * sum(weights.values())
    for k, v in weights.items():
        if number < v:
            break
        number -= v
    return k


def update_q_sarsa(self):
    if self.learning:
        previous_q = self.q[self.previous_belief][self.previous_action]
        next_q = self.q[self.belief][self.action]
        self.q[self.previous_belief][self.previous_action] = \
            previous_q + self.alpha * (self.reward + self.gamma * next_q - previous_q)


def update_q_td(self):
    if self.learning:
        previous_q = self.q[self.belief][self.action]
        self.q[self.belief][self.action] = \
            previous_q + self.alpha * (self.reward - previous_q)


def do_iteration(self, debug=False):
    if self.found:
        self.clear()

    self.encoding_n += 1

    self.previous_belief = self.belief
    self.observe()
    self.set_belief_state()

    self.previous_action = self.action
    self.choose_action_softmax()

    self.update_q_sarsa()

    # Move eyes, calculate mt.
    eccentricity = self.distance(self.eye_loc[0], self.eye_loc[1], self.action[0], self.action[1])
    self.mt, moved = self.EMMA_fixation_time(eccentricity * self.cell_distance, encoding_noise=self.encoding_noise)
    self.total_search_time += self.mt
    self.eyes_moved = False
    if moved:
        self.eyes_moved = True        
        self.eye_loc = self.action
        self.fixation_count = 1
    else:
        self.fixation_count = 0

    # Calculate reward and mark if target found.
    self.calculate_reward()

    # Learn TD here if found, because next iteration will clear the model.
    if self.found:
        self.update_q_td()

    if self.log_p:
        self.log.append("{} {} {} {} {} {} {}".format(self.trial, self.task_type, self.encoding_n,
                                                      self.model_time, self.mt, self.fixation_count, self.reward))


def learn_model(self, until, debug=False):
    print_time = self.model_time
    long_tasks = 0
    marked_long = False
    while self.model_time < until:

        if not marked_long and self.encoding_n > self.rows * self.cols * +1:
            long_tasks += 1
            marked_long = True
        if self.found:
            marked_long = False
            # end of one trial, log stats for trial.
            if self.log_p:
                tgt = -1 if self.target is None else (self.target[0] * self.cols) + self.target[1]

        if debug and self.model_time >= print_time:
            print("Running model...", round(self.model_time / until, 2), len(self.q),
                  round(self.softmax_temp, 2), long_tasks)
            print_time += until / 10
            long_tasks = 0
        self.do_iteration()        
        # Anneal softmax temp
        if self.learning:
            self.softmax_temp = 1 - ((self.model_time * 0.95) / until) + 0.01


def write_data_to_file(self, filename):
    out = open(filename, "w")
    out.write(self.log_header + "\n")
    for d in self.log:
        out.write(d + "\n")
        out.flush()
    out.close()


def print_stats(self, n=5):
    data = pd.read_csv("logs.txt", sep=" ")

    data = data.groupby(["trial"]).agg({'task.type': 'mean',
                                        'mt': 'sum',
                                        'n.fix': 'sum',
                                        'reward': 'sum',
                                        }).reset_index()
    data = data.tail(n + 1)
    data = data[:-1]

    stat = data.groupby(["task.type"]).agg({'mt': lambda x: list(x),
                                            'n.fix': lambda x: list(x),
                                            'reward': lambda x: list(x)
                                            }).reset_index()

    table = [["task.type"], ["search.time.mean"], ["search.time.std"], ["n.fixation.mean"],
             ["n.fixation.std"], ["total.reward.mean"], ["total.reward.std"]]

    for row in range(len(stat)):
        table[0].append(round(np.mean(stat.iloc[row, stat.columns.get_loc("task.type")]), 1))
        table[1].append(round(np.mean(stat.iloc[row, stat.columns.get_loc("mt")]), 1))
        table[2].append(round(np.std(stat.iloc[row, stat.columns.get_loc("mt")]), 1))
        table[3].append(round(np.mean(stat.iloc[row, stat.columns.get_loc("n.fix")]), 1))
        table[4].append(round(np.std(stat.iloc[row, stat.columns.get_loc("n.fix")]), 1))
        table[5].append(round(np.mean(stat.iloc[row, stat.columns.get_loc("reward")]), 1))
        table[6].append(round(np.std(stat.iloc[row, stat.columns.get_loc("reward")]), 1))

    display(HTML(tabulate.tabulate(table, tablefmt='html')))


def plot_stats(self, chunk=10):
    plot_asc()
    data = pd.read_csv("logs.txt", sep=" ")
    data = data.groupby(["trial"]).agg({'task.type': 'mean',
                                        'mt': 'sum',
                                        'n.fix': 'sum',
                                        'reward': 'sum',
                                        }).reset_index()

    fig, axs = plt.subplots()

    # plot rewards.
    d = data['reward'].values
    means = []
    stds = []
    for i in range(0, len(d), chunk):
        means.append(np.mean(d[i:i + chunk]))
        stds.append(np.std(d[i:i + chunk]))

    means = np.asarray(means)
    stds = np.asarray(stds)

    axs.plot(means, color='orange')
    axs.fill_between(range(len(means)), means - stds, means + stds)
    axs.set_ylabel('Total Reward')
    axs.set_xlabel('Trial')

    rand = data[data['task.type'].isin([0])]
    absnt = data[data['task.type'].isin([1])]
    task = ['Random', 'Absent']
    search_time_mean = [np.mean(rand['mt'].values), np.mean(absnt['mt'].values)]
    search_time_std = [np.std(rand['mt'].values), np.std(absnt['mt'].values)]

    fix_time_mean = [np.mean(rand['n.fix'].values), np.mean(absnt['n.fix'].values)]
    fix_time_std = [np.std(rand['n.fix'].values), np.std(absnt['n.fix'].values)]

    fig, axs = plt.subplots()
    # plot search time
    axs.bar(task, search_time_mean, yerr=search_time_std, align='center', alpha=0.5, ecolor='black', capsize=10)
    axs.set_ylabel('Time (secs)')
    axs.set_xlabel('Task')
    axs.set_xticks(np.arange(len(task)))
    axs.set_xticklabels(task)

    fig, axs = plt.subplots()
    # plot fixation count
    axs.bar(task, fix_time_mean, yerr=fix_time_std, align='center', alpha=0.5, ecolor='black', capsize=10)
    axs.set_ylabel('# Fixations')
    axs.set_xlabel('Task')
    axs.set_xticks(np.arange(len(task)))
    axs.set_xticklabels(task)

    plt.show()


def plot_asc():
    plt.style.use('seaborn-white')
    plt.rcParams['font.serif'] = 'Ubuntu'
    plt.rcParams['font.monospace'] = 'Ubuntu Mono'
    plt.rcParams['font.size'] = 20
    plt.rcParams['axes.labelsize'] = 20
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.titlesize'] = 20
    plt.rcParams['xtick.labelsize'] = 15
    plt.rcParams['ytick.labelsize'] = 15
