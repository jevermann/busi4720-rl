# #########################
# Joerg Evermann, 2024
# Licensed under GNU GPL V3
###########################
import collections
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
max_steps = 1000000
# When to update weights
C = 50*batch_size
# Replay buffer D
D = collections.deque(maxlen=5000)
ddqn = True # True for DDQN, False for DQN

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

Q2 = keras.models.clone_model(Q1)
Q2.compile(loss='huber', optimizer='adam')
Q2.set_weights(Q1.get_weights())

# Target network, used to compute targets
Qhat = keras.models.clone_model(Q1)
Qhat.compile(loss='huber', optimizer='adam')
Qhat.set_weights(Q1.get_weights())

def getQ(Q, s, a):
    return Q.predict(np.expand_dims(np.array(s.tolist() + [a]), axis=0), verbose=0)[0][0]

def maxQ(Q, s, arg):
    maxq = -np.inf
    maxa = None
    for a in Actions:
        q = getQ(Q, s, a)
        if q > maxq:
            maxq = q
            maxa = a
    return maxa if arg else maxq

def pi(Q, s, epsilon):
    if random.random() < epsilon:
        return random.choice(Actions)
    else:
        return maxQ(Q, s, True)

def target_DQN(Q, Qhat, a, r, sprime):
    return r + gamma * maxQ(Qhat, sprime, False)

def target_DDQN(Q, Qhat, a, r, sprime):
    return r + gamma * getQ(Qhat, sprime, maxQ(Q, sprime, False))

def training_xy(Q, batch, ddqn=False):
    x = np.zeros((batch_size, Ssize+1))
    y = np.zeros(batch_size)
    for i, (s, a, r, t, sprime) in enumerate(batch):
        x[i] = list(s) + [a]
        if t == 1:
            y[i] = r
        else:
            if ddqn:
                y[i] = target_DDQN(Q, Qhat, a, r, sprime)
            else:
                y[i] = target_DQN(Q, Qhat, a, r, sprime)
    return x, y

G = 0
for t in range(max_steps):
    s = env.reset()[0]
    if random.random() < .5:
        a = pi(Q1, s, epsilon*math.exp(-t/neps))
    else:
        a = pi(Q2, s, epsilon * math.exp(-t / neps))
    sprime, r, terminal, _, _ = env.step(a)
    G += r
    D.append((s, a, r, int(terminal), sprime))
    s = sprime
    if t >= batch_size:
        batch = random.sample(D, batch_size)
        if random.random() < .5:
            x, y = training_xy(Q1, batch, ddqn=True)
            loss = Q1.train_on_batch(x=x, y=y)
        else:
            x, y = training_xy(Q2, batch, ddqn=True)
            loss = Q2.train_on_batch(x=x, y=y)
        print('Step {:>8n}, Action {:>1n}, Epsilon {:>.4f}, Loss: {:>.6f}'.format(t, a, epsilon * math.exp(-t / neps), loss))

    if t % C == 0:
        print("Updating target network")
        Qhat.set_weights(Q1.get_weights())

    if terminal:
        print('Terminal state at step {:>8n} with return {:>4n}'.format(t, G))
        G = 0
