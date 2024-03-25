# #########################
# Joerg Evermann, 2024
# Licensed under GNU GPL V3
###########################

import math
import random

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

gamma = 0.9
epsilon = 0.05

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

# Initialize weight accumulator
C = dict()
def getC(s, a):
    if (s, a) not in C:
        return 0
    else:
        return C[(s, a)]

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
        # No Q info
        return None

# Define policies b and pi
_b = dict()
def b(s):
    a_opt = argmaxQ(s)
    probs = []
    if a_opt is None:
        for a in Actions:
            _b[(s, a)] = 1 / len(Actions)
            probs.append(_b[(s, a)])
    else:
        for a in Actions:
            if a == a_opt:
                _b[(s, a)] = 1 - epsilon + epsilon / len(Actions)
            else:
                _b[(s, a)] = epsilon / len(Actions)
            probs.append(_b[(s, a)])
    return random.choices(Actions, probs, k=1)[0]

def bprob(a, s):
    if (s, a) not in _b:
        a_opt = argmaxQ(s)
        if a_opt is None:
            for a in Actions:
                _b[(s, a)] = 1 / len(Actions)
        else:
            for a in Actions:
                if a == a_opt:
                    _b[(s, a)] = 1 - epsilon + epsilon / len(Actions)
                else:
                    _b[(s, a)] = epsilon / len(Actions)
    return _b[(s, a)]

def pi(s):
    a = argmaxQ(s)
    if a is None:
        return random.choice(Actions)
    else:
        return a

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

    def generate_episode_b(self):
        terminal = False
        s = self.reset()
        a = random.choice(Actions)
        states = [s]
        actions = [a]
        rewards = [math.nan]
        while terminal is False:
            sprime, r, terminal = self.step(s, a, noisy=True)
            rewards.append(r)
            if not terminal:
                aprime = b(sprime)
                states.append(sprime)
                actions.append(aprime)
                s = sprime
                a = aprime

        return states, actions, rewards, len(rewards)

    def generate_episode_pi(self):
        terminal = False
        s = self.reset()
        a = random.choice(Actions)
        states = [s]
        actions = [a]
        rewards = [math.nan]
        while terminal is False:
            sprime, r, terminal = self.step(s, a, noisy=False)
            rewards.append(r)
            if not terminal:
                aprime = pi(sprime)
                if sprime == s and aprime == a:
                    # Stuck on the greedy policy
                    aprime = b(sprime)
                states.append(sprime)
                actions.append(aprime)
                s = sprime
                a = aprime

        return states, actions, rewards, len(rewards)

env = racetrack_env(0.1)

def visualize_policy(i):
    S, A, R, T = env.generate_episode_pi()
    trackmap = env.track.copy()
    fig = px.imshow(trackmap, title="T=" + str(T), labels=dict(x="", y="", color=""))

    pos = []
    for ((r, c), _) in S:
        pos.append((r, c))
    pos = pd.DataFrame(pos)
    fig.add_trace(go.Scatter(x = pos[1], y=pos[0], line=dict(color='black')))

    fig.write_image('racetrack_off_policy_iteration_' + str(i) + '.png')

for e in range(0, 10000+1):
    print('Episode ' + str(e))
    S, A, R, T = env.generate_episode_b()
    G = 0
    W = 1
    for t in reversed(range(0, T-1)):
        G = gamma*G + R[t+1]
        C[(S[t], A[t])] = getC(S[t], A[t]) + W
        Q[(S[t], A[t])] = getQ(S[t], A[t]) + W/getC(S[t], A[t]) * (G - getQ(S[t],A[t]))
        if A[t] != pi(S[t]):
            break
        else:
            W = W * 1/bprob(A[t], S[t])

    if e%100 == 0:
        visualize_policy(e)