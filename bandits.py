# #########################
# Joerg Evermann, 2024
# Licensed under GNU GPL V3
###########################

import random
import numpy as np
import pandas as pd
import plotly.express as px

class k_bandit_env:

    def __init__(self, k):
        self.k = k
        self.mean_rewards = []

        for i in range(self.k):
            self.mean_rewards.append(random.normalvariate(0, 1))

    def step(self, action):
        mean = self.mean_rewards[action]
        reward = random.normalvariate(mean, 1)
        return reward

class k_bandit_agent:
    def __init__(self, k, epsilon, initial_value):
        self.k = k
        self.epsilon = epsilon
        self.env = k_bandit_env(k)

        self.Q = [initial_value] * self.k
        self.N = [.0] * self.k

    def determine_action(self):
        if random.uniform(0,1) < self.epsilon:
            # explore
            action = random.randint(0, self.k-1)
        else:
            # exploit
            action = self.Q.index(max(self.Q))
        return action

    def train(self, steps):
        rewards = []
        for t in range(1,steps+1):
            action = self.determine_action()
            reward = self.env.step(action)
            self.N[action] += 1
            self.Q[action] = (reward - self.Q[action]) / self.N[action]

            rewards.append(reward)
        return rewards

def evaluate_agent(steps, runs, k, epsilon, initial_value):
    rewards = np.zeros((steps, runs))
    for run in range(runs):
        agent = k_bandit_agent(k, epsilon, initial_value)
        rewards[:,run] = agent.train(steps)

    return np.mean(rewards, axis=1)

eps0 =       evaluate_agent(1000, 5000, 10, 0.0,  0)
eps01 =      evaluate_agent(1000, 5000, 10, 0.1,  0)
eps001 =     evaluate_agent(1000, 5000, 10, 0.01, 0)
eps0_optim = evaluate_agent(1000, 5000, 10, 0.0,  5)

results0 = pd.DataFrame({'agent': ['eps0']*1000, 'results': eps0})
results01 = pd.DataFrame({'agent': ['eps01']*1000, 'results': eps01})
results001 = pd.DataFrame({'agent': ['eps001']*1000, 'results': eps001})
results_optim = pd.DataFrame({'agent': ['eps0opt']*1000, 'results': eps0_optim})

results = pd.concat([results0, results01, results001, results_optim], axis=0)
fig = px.line(results, y='results', color='agent')
fig.show()
fig.write_image('bandits.pdf', height=600, width=1000)