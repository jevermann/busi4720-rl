# #########################
# Joerg Evermann, 2024
# Licensed under GNU GPL V3
###########################

import math
import random
import keras
from keras import layers
import gymnasium as gym
import tensorflow as tf
import numpy as np
import pygame

env = gym.make("CartPole-v1", render_mode="human")
# env = gym.make("CartPole-v1")

Actions = range(0, env.action_space.n)
Ssize = env.observation_space.shape[0]
# Neural net parameters
batch_size = 8
dropout = 0.25
activation = 'relu'
# activation = 'sigmoid'
# activation = 'tanh'

epsilon = 0.80
gamma = 0.90
neps = 10000
# When to update weights
C = 50*batch_size
# Replay buffer
maxD = 5000
# D needs to hold two states, plus action, reward,
# and indicator for terminal
D = np.zeros((maxD, 2*Ssize+3))

# Main network, used to select actions
Q1 = keras.Sequential([
    layers.InputLayer(input_shape=(Ssize+1),
                      batch_size=batch_size,
                      dtype=tf.float32),
    layers.Dense(Ssize*4, activation=activation),
    layers.Dropout(rate=dropout),
    layers.Dense(Ssize*2, activation=activation),
    layers.Dropout(rate=dropout),
    layers.Dense(1, activation='linear')
])
Q1.compile(loss='huber', optimizer='adam')
# Second main network
Q2 = keras.models.clone_model(Q1)
Q2.compile(loss='huber', optimizer='adam')
Q2.set_weights(Q1.get_weights())

# Target network, used to compute targets
Qhat = keras.models.clone_model(Q1)
Qhat.compile(loss='huber', optimizer='adam')
Qhat.set_weights(Q1.get_weights())

def maxQ(Q, s, arg):
    maxq = -np.inf
    maxa = None
    for a in Actions:
        q = Q.predict(np.expand_dims(np.array(s.tolist()+[a]), axis=0), verbose=0)[0][0]
        if q > maxq:
            maxq = q
            maxa = a
    return maxa if arg else maxq

def pi(Q, s, epsilon):
    if random.random() < epsilon:
        return random.choice(Actions)
    else:
        return maxQ(Q, s, True)

def get_targets(Q, batch):
    x = batch[:,0:Ssize+1]
    y = np.zeros((batch.shape[0]))
    for i in range(batch.shape[0]):
        if batch[i,Ssize+2] == 1:
            y[i] = batch[i, Ssize + 1]
        else:
            y[i] = batch[i, Ssize + 1] + gamma * Qhat.predict(np.expand_dims(np.array(s.tolist()+[maxQ(Q, batch[i,Ssize+3:], True)]), axis=0), verbose=0)[0][0]
    return x, y

t = 0
G = 0
for e in range(neps):
    terminal = False
    s = env.reset()
    s = s[0]
    while terminal is False:
        if random.random() < 0.5:
            a = pi(Q1, s, epsilon*math.exp(-t/neps))
        else:
            a = pi(Q2, s, epsilon * math.exp(-t / neps))
        sprime, r, terminal, _, _ = env.step(a)
        G += r
        D[t % maxD] = s.tolist() + [a] + [r] + [int(terminal)] + sprime.tolist()
        s = np.copy(sprime)
        if t >= batch_size:
            batch = D[np.random.choice(min(D.shape[0], t), batch_size, replace=False), :]
            if random.random() < 0.5:
                x, y = get_targets(Q1, batch)
                loss = Q1.train_on_batch(x=x, y=y)
            else:
                x, y = get_targets(Q2, batch)
                loss = Q2.train_on_batch(x=x, y=y)
            print('Step {:>8n}, Episode {:>8n}, Action {:>1n}, Epsilon {:>.4f}, Loss: {:>.6f}'.format(t, e, a, epsilon * math.exp(-t / neps), loss))
        t += 1

        if t % C == 0:
            print("Updating target network")
            Qhat.set_weights(Q1.get_weights())

    print('Terminal state at step {:>8n} with return {:>4n}'.format(t, G))
    t_term = t
    G = 0
