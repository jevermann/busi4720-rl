# #########################
# Joerg Evermann, 2024
# Licensed under GNU GPL V3
###########################

import math
import random

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

epsilon = 0.05
alpha = 0.5
gamma = 1


# Define possible actions
Actions = []
for y in range(-1, 2):
    for x in range(-1, 2):
        Actions.append((y,x))

# Initialize action-value function
Q = dict()
def getQ(s, a):
    if (s, a) not in Q:
        return 0
    else:
        return Q[(s, a)]

def argmaxQ(s):
    values = []
    actions = []
    for a in Actions:
        if (s, a) in Q:
            values.append(Q[(s, a)])
            actions.append(a)
    if len(actions) > 0:
        max_idx = values.index(max(values))
        return actions[max_idx]
    else:
        return random.choice(Actions)
def maxQ(s):
    values = []
    actions = []
    for a in Actions:
        if (s, a) in Q:
            values.append(Q[(s, a)])
            actions.append(a)
    if len(actions) > 0:
        return max(values)
    else:
        return 0

# Initialize policy
pi = dict()
def get_action(s):
    weights = []
    for a in Actions:
        if (s, a) in pi:
            weights.append(pi[(s, a)])
    if len(weights) == 0:
        return random.choice(Actions)
    else:
        return random.choices(Actions, weights=weights, k=1)[0]

# Initialize returns
Returns = dict()
def getReturns(s, a):
    if (s, a) not in Returns:
        return []
    else:
        return Returns[(s, a)]
def appendReturn(s, a, r):
    if (s, a) not in Returns:
        Returns[(s, a)] = [r]
    else:
        Returns[(s, a)].append(r)

class racetrack_env:
    def __init__(self, noise):
        self.noise = noise
        self.track = pd.read_csv('racetrack.csv')
        self.track = self.track.rename(columns={x: y for x, y in zip(self.track.columns, range(0, len(self.track.columns)))})

        self.start_pos = self.track[self.track==2].stack().index.tolist()
        self.end_pos = self.track[self.track==3].stack().index.tolist()
        self.ontrack_pos = self.track[self.track==0].stack().index.tolist()
        self.offtrack_pos = self.track[self.track==1].stack().index.tolist()

    def is_out_of_bounds(self, pos):
        if pos[0] < 0 or pos[0] >= self.track.shape[1]:
            return True
        if pos[1] < 0 or pos[1] >= self.track.shape[0]:
            return True
        if pos in self.offtrack_pos:
            return True
        return False

    def is_finished(self, pos):
        if not self.is_out_of_bounds(pos):
            return pos in self.end_pos
        else:
            return False

    def reset(self):
        return (random.choice(self.start_pos), (0, 0))

    def step(self, state, action, noisy=True):
        (pos, velocity) = state
        pos = (pos[0] - velocity[0],
               pos[1] + velocity[1])
            # y-axis is reverse coded

        if self.is_finished(pos):
            return state, +1, True

        if not self.is_out_of_bounds(pos):
            if noisy and (random.random() < self.noise):
                action = (0, 0)
            new_v1 = max(min(velocity[0] + action[0], 5), 0)
            new_v2 = max(min(velocity[1] + action[1], 5), 0)
            velocity = (new_v1, new_v2)
            sprime = (pos, velocity)
            return sprime, -1, False
        else:
            return self.reset(), -1, False

    def generate_episode(self, noisy=True):
        terminal = False
        s = self.reset()
        a = random.choice(Actions)
        states = [s]
        actions = [a]
        rewards = [math.nan]
        while terminal is False:
            sprime, r, terminal = self.step(s, a, noisy=noisy)
            rewards.append(r)
            if not terminal:
                aprime = get_action(sprime)
                states.append(sprime)
                actions.append(aprime)
                s = sprime
                a = aprime

        return states, actions, rewards, len(rewards)

env = racetrack_env(0.1)

def visualize_policy(i):
    S, A, R, T = env.generate_episode(noisy=False)
    trackmap = env.track.copy()
    fig = px.imshow(trackmap, title="T=" + str(T), labels=dict(x="", y="", color=""))

    pos = []
    for ((r, c), _) in S:
        pos.append((r, c))
    pos = pd.DataFrame(pos)
    fig.add_trace(go.Scatter(x = pos[1], y=pos[0], line=dict(color='black')))

    fig.write_image('racetrack_q_learning_episode_' + str(i) + '.png')

steps = []
for e in range(0, 100000):
    terminal = False
    S = env.reset()
    step = 0
    while terminal is False:
        A = get_action(S)
        Sprime, R, terminal = env.step(state=S, action=A, noisy=True)
        Q[(S,A)] = getQ(S,A) + alpha*(R + gamma * maxQ(Sprime) - getQ(S, A))
        S = Sprime
        step += 1

    steps.append(step)

    if e%10000 == 0:
        visualize_policy(e)
