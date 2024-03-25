# #########################
# Joerg Evermann, 2024
# Licensed under GNU GPL V3
###########################

import math
import random
from statistics import mean

import pandas as pd
import plotly.express as px

gamma = 1.0
epsilon = 0.05
# Define states
States = []
for ace in [0,1]:
    for dealer_showing in range(1,11):
        for hand_sum in range(12, 22):
            States.append((ace, dealer_showing, hand_sum))

# Define actions
Actions = (0, 1)

# Initialize policy
pi = dict()
for s in States:
    for a in Actions:
        pi[(s, a)] = random.random()

# Initialize action value function
Q = dict()
for s in States:
    for a in Actions:
        Q[(s, a)] = 0

# Initialize returns
Returns = dict()
for s in States:
    for a in Actions:
        Returns[(s, a)] = []

# Draw a card with proper probability
def draw_card():
    return random.choices(range(1,11), weights=[1, 1, 1, 1, 1, 1, 1, 1, 1, 4], k=1)[0]

def step(state, action):
    (player_ace, dealer_showing, player_sum) = state
    if player_sum <= 21 and action == 1:
        card = draw_card()
        if card == 1:
            if player_sum < 11:
                player_sum += 11
                player_ace = 1
            else:
                player_sum += 1
                player_ace = 0
        else:
            player_sum += card
            if player_sum > 21 and player_ace == 1:
                player_sum += -10
                player_ace = 0
        sprime = (player_ace, dealer_showing, player_sum)
        if player_sum > 21:
            return sprime, -1, True
        else:
            return sprime, 0, False
    else:
        sprime = state
        if dealer_showing == 1:
            dealer_sum = 11
            dealer_ace = 1
        else:
            dealer_sum = dealer_showing
            dealer_ace = 0
        while dealer_sum < 17:
            card = draw_card()
            if card == 1:
                if dealer_sum < 11:
                    dealer_sum += 11
                    dealer_ace = 1
                else:
                    dealer_sum += 1
                    dealer_ace = 0
            else:
                dealer_sum += card
                if dealer_sum > 21 and dealer_ace == 1:
                    dealer_sum += -10
                    dealer_ace = 0
        if dealer_sum > 21:
            return sprime, +1, True
        if dealer_sum == 21 and player_sum == 21:
            return sprime, 0, True
        if abs(21 - dealer_sum) == abs(21 - player_sum):
            return sprime, 0, True
        if abs(21 - dealer_sum) < abs(21 - player_sum):
            return sprime, -1, True
        if abs(21 - dealer_sum) > abs(21 - player_sum):
            return sprime, +1, True

def choose_action(pi, s):
    weights = []
    for a in Actions:
        weights.append(pi[(s, a)])
    return random.choices(Actions, weights=weights, k=1)[0]

def generate_episode(pi, s0, a0):
    terminal = False
    s = s0
    a = a0
    states = [s0]
    actions = [a0]
    rewards = [math.nan]
    while terminal is False:
        sprime, r, terminal = step(s, a)
        rewards.append(r)
        if not terminal:
            aprime = choose_action(pi, sprime)
            states.append(sprime)
            actions.append(aprime)
            s = sprime
            a = aprime

    return states, actions, rewards, len(rewards)

def visualize_policy(i):
    pidf = pd.DataFrame([(ace, dealer, player, action) for ((ace, dealer, player), action) in pi.keys()])
    pidf = pd.concat([pidf, pd.Series(pi.values())], axis=1)
    pidf.columns = ['Ace', 'Dealer', 'Player', 'Action', 'Pi']
    for ace in (0, 1):
        acedf = pidf.loc[pidf['Ace']==ace][['Dealer', 'Player', 'Action', 'Pi']]
        acedf = acedf.sort_values('Pi').drop_duplicates(['Dealer', 'Player'])
        acedf = acedf.pivot(index='Dealer', columns='Player')['Action']
        fig = px.imshow(acedf, labels=dict(x="Player sum", y="Dealer showing", color="Action"))
        fig.update_xaxes(side="top")
        fig.write_image('blackjack_eps_pi_iteration_' + str(i) + '_ace' + str(ace) + '.png')

    qdf = pd.DataFrame([(ace, dealer, player, action) for ((ace, dealer, player), action) in Q.keys()])
    qdf = pd.concat([qdf, pd.Series(Q.values())], axis=1)
    qdf.columns = ['Ace', 'Dealer', 'Player', 'Action', 'Value']
    for ace in (0, 1):
        acedf = qdf.loc[qdf['Ace']==ace][['Dealer', 'Player', 'Action', 'Value']]
        acedf = acedf.groupby(['Dealer', 'Player']).mean().reset_index()
        acedf = acedf.pivot(index='Dealer', columns='Player')['Value']
        fig = px.imshow(acedf, labels=dict(x="Player sum", y="Dealer showing", color="Value"))
        fig.update_xaxes(side="top")
        fig.write_image('blackjack_eps_v_iteration_' + str(i) + '_ace' + str(ace) + '.png')

for e in range(0, 1000000+1):
    s0 = random.choice(States)
    a0 = random.choice(Actions)
    S, A, R, T = generate_episode(pi, s0, a0)
    G = 0
    for t in reversed(range(0, T-1)):
        G = gamma*G + R[t+1]
        if (t == 0) or ((S[t], A[t]) not in zip(S[0:t-1], A[0:t-1])):
            Returns[(S[t], A[t])].append(G)
            Q[(S[t], A[t])] = mean(Returns[(S[t], A[t])])
            A_star = 1 if Q[(S[t], 1)] > Q[(S[t], 0)] else 0
            for a in Actions:
                if a == A_star:
                    pi[(S[t],a)] = 1 - epsilon + epsilon/len(Actions)
                else:
                    pi[(S[t],a)] = epsilon/len(Actions)

    if e%100000 == 0:
        visualize_policy(e)