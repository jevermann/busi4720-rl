# #########################
# Joerg Evermann, 2024
# Licensed under GNU GPL V3
###########################

import math
import random

import pandas as pd
import plotly.express as px

theta = 50

lambda1req = 3
lambda2req = 4
lambda1ret = 3
lambda2ret = 2

gamma = 0.9

# Construct States
States = []
for cars1 in range(21):
    for cars2 in range(21):
        States.append((cars1, cars2))

# Initialize value function V
V = dict()
for state in States:
    V[state] = 0

# Initialize policy pi
pi = dict()
for state in States:
    pi[state] = random.choice(range(-5, 6))
    # pi[state] = 0

def poisson(k, l):
    return math.pow(l, k)*math.exp(-l) / math.factorial(k)

def exp_reward(s, a):
    (loc1, loc2) = s
    reward = -2 * abs(a)

    cars1 = max(min(loc1 - a, 20), 0)
    cars2 = max(min(loc2 + a, 20), 0)

    for ret1 in range(0, 3*lambda1ret):
        for ret2 in range(0, 3*lambda2ret):
            for req1 in range(0, 3*lambda1req):
                for req2 in range(0, 3*lambda2req):
                    p = poisson(ret1, lambda1ret) * \
                        poisson(ret2, lambda2ret) * \
                        poisson(req1, lambda1req) * \
                        poisson(req2, lambda2req)

                    valid_req1 = min(cars1, req1)
                    valid_req2 = min(cars2, req2)
                    r = (valid_req1 + valid_req2) * 10

                    sprime = (max(min(cars1 - valid_req1 + ret1, 20), 0),
                              max(min(cars2 - valid_req2 + ret2, 20), 0))
                    reward += p * (r + gamma * V[sprime])
    return reward

def evaluate_policy():
    global theta
    print("evaluate")
    while True:
        Delta = 0
        for s in States:
            v = V[s]
            V[s] = exp_reward(s, pi[s])
            Delta = max(Delta, abs(v - V[s]))
        print(Delta)
        if Delta < theta:
            theta = theta / 5
            break

def improve_policy():
    print("improve")
    stable = True
    for s in States:
        old_action = pi[s]
        # Iterate over all actions
        max_r = -math.inf
        max_a = None
        # number of cars to move from location 1 to location 2
        # may be negative, between -5 and +5
        for action in range(-min(state[1], 5), min(state[0], 5)+1):
            r = exp_reward(s, action)
            if r > max_r:
                max_r = r
                max_a = action
        pi[s] = max_a
        if old_action != pi[s]:
            stable = False
    return stable

def visualize_policy(i):
    pidf = pd.DataFrame(pi.keys())
    pidf = pd.concat([pidf, pd.Series(pi.values())], axis=1)
    pidf.columns = ['Loc1', 'Loc2', 'Move']
    pidf = pidf.pivot(index='Loc1', columns='Loc2')['Move']
    fig = px.imshow(pidf, labels=dict(x="Cars at Location 1", y="Cars at Location 2", color="Cars to Move"))
    fig.update_xaxes(side="top")
    fig.write_image('jacks_pi_iteration_' + str(i) + '.png')

    vdf = pd.DataFrame(V.keys())
    vdf = pd.concat([vdf, pd.Series(V.values())], axis=1)
    vdf.columns = ['Loc1', 'Loc2', 'V']
    vdf = vdf.pivot(index='Loc1', columns='Loc2')['V']
    fig = px.imshow(vdf, labels=dict(x="Cars at Location 1", y="Cars at Location 2", color="State Value"))
    fig.update_xaxes(side="top")
    fig.write_image('jacks_V_iteration_' + str(i) + '.png')

stable = False
i = 0
visualize_policy(i)
while not stable:
    evaluate_policy()
    stable = improve_policy()
    i += 1
    visualize_policy(i)

print("Optimal Policy:")
print(pi)
