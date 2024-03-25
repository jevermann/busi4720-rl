# #########################
# Joerg Evermann, 2024
# Licensed under GNU GPL V3
###########################

import random

import pandas as pd
import plotly.express as px

class WindyWorld:

    def random_state(self):
        return (random.randrange(0, self.nrow), random.randrange(0, self.ncol))

    def __init__(self, nrow, ncol, wind) -> None:
        self.nrow=nrow
        self.ncol=ncol
        self.wind = wind

        self.goal = self.random_state()
        self.start = self.random_state()
        while self.start == self.goal:
            self.start = self.random_state()
        self.pos = self.start

    def reset(self):
        self.pos = self.start
        return self.pos

    def step(self, a):
        (row, col) = self.pos
        # a = 0 is up
        # a = 1 is right
        # a = 2 is down
        # a = 3 is left
        done = False
        r = -1

        match a:
            case 0:
                row = min(row+1, self.nrow-1)
            case 1:
                col = min(col+1, self.ncol-1)
            case 2:
                row = max(row-1, 0)
            case 3:
                col = max(col-1, 0)

        row = max(min(row+self.wind[col], self.nrow-1), 0)
        self.pos = (row, col)
        if self.pos == self.goal:
            r = 1
            done = True

        return self.pos, r, done

nrow = 5
ncol = 10

epsilon = 0.05
alpha = 0.5
gamma = 1

# Define states
States = []
for i in range(nrow):
    for j in range(ncol):
        States.append((i, j))
# Define actions
Actions = range(0, 4)
# Initialize Q
Q = dict()
for s in States:
    for a in Actions:
        Q[(s, a)] = random.random()

def argmaxQ(s):
    values = []
    actions = []
    for a in Actions:
        values.append(Q[(s, a)])
        actions.append(a)
    max_idx = values.index(max(values))
    return actions[max_idx]
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

# Define pi
def pi(s):
    if random.random() < epsilon:
        return random.choice(Actions)
    else:
        return argmaxQ(s)

# Create a windy world with 5 rows and 10 columns
windy = WindyWorld(5, 10, [0, 0, 0, 1, 1, 1, 1, 0, 0, 0])

steps = []
for e in range(0, 1000):
    terminal = False
    S = windy.reset()
    step = 0
    while terminal is False:
        A = pi(S)
        Sprime, R, terminal = windy.step(A)
        Q[(S,A)] = Q[(S,A)] + alpha*(R + gamma * maxQ(Sprime) - Q[(S, A)])
        S = Sprime
        step += 1

    steps.append(step)

fig = px.scatter(pd.DataFrame({'steps': steps}), trendline="lowess", trendline_color_override="red")
fig.update_traces(marker_size=10)
fig.write_image('windyworld_q_learning.png')
