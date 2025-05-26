# #########################
# Joerg Evermann, 2024
# Licensed under GNU GPL V3
###########################
#
# This is barebones REINFORCE directly
# from the pseudo-code in Sutton & Barto's book
#
import keras
from keras import layers
import gymnasium as gym
import tensorflow as tf
import numpy as np

env = gym.make("CartPole-v1", render_mode="human")
# env = gym.make("CartPole-v1")

Actions = range(0, env.action_space.n)
Ssize = env.observation_space.shape[0]

# RL parameters
gamma = 0.90
neps = 100
max_steps = 1000

# Policy network for discrete actions
policy_network = keras.Sequential([
    layers.InputLayer(batch_input_shape=(1, Ssize), dtype=tf.float32),
    layers.Dense(Ssize*4, activation='relu'),
    layers.Dropout(rate=0.25),
    layers.Dense(Ssize*2, activation='relu'),
    layers.Dropout(rate=0.25),
    layers.Dense(len(Actions), activation='softmax')
])
p_optimizer = keras.optimizers.Adam()

# Value network for the value of each state
value_network = keras.Sequential([
    layers.InputLayer(batch_input_shape=(1, Ssize), dtype=tf.float32),
    layers.Dense(Ssize * 2, activation='relu'),
    layers.Dropout(rate=0.25),
    layers.Dense(Ssize, activation='relu'),
    layers.Dropout(rate=0.25),
    layers.Dense(1, activation=None)
])
v_optimizer = keras.optimizers.Adam()

# Selecting an action for a state means getting
# predictions for all actions and then sampling
# from actions using those probabilities
def select_action(state):
    probs = policy_network(np.expand_dims(state, axis=0))[0].numpy()
    # Normalize probabilities just in case they're off by a bit
    probs = probs / probs.sum()
    # Sample actions
    action = np.random.choice(Actions, size=1, p=probs)[0]
    return action

# Given a list of rewards and a discount factor
# return a list of discounted returns
def discounted_returns(rewards, gamma):
    returns = [0] * len(rewards)
    running_return = 0
    for t in reversed(range(len(rewards))):
        running_return = rewards[t] + gamma * running_return
        returns[t] = running_return
    return returns

# Do this for each episode
for e in range(neps):
    # Initialize variables and lists
    T = 0
    rewards, states, actions = [], [], []
    # Reset environment
    s = env.reset()[0]
    terminal = False
    # Generate an episode and keep track
    # of states, actions, rewards
    while (T < max_steps) and not terminal:
        a = select_action(s)
        sprime, r, terminal, _, _ = env.step(a)
        states.append(s)
        actions.append(a)
        rewards.append(r)
        s = sprime
        T += 1

    print(f'Episode {e:5} goes to step {T:3}')
    # Compute discounted returns
    returns = discounted_returns(rewards, gamma)

    # Learn for each step of the episode
    for t in range(len(returns)):
        with tf.GradientTape() as p_tape:
            with tf.GradientTape() as v_tape:
                # Action probabilities
                pi = policy_network(np.expand_dims(states[t], axis=0))
                v = value_network(np.expand_dims(states[t], axis=0))

                # Action index
                action_idx = np.array(actions[t], dtype=np.int32)
                # Return
                G = np.array(returns[t])
                delta = G - v
                # Loss is the negative log of the probability of the action, times the return
                p_loss = -delta * gamma**t * tf.math.log(tf.reduce_sum(tf.math.multiply(pi, tf.one_hot(action_idx, env.action_space.n)), axis=1))
                v_loss = -delta * v

        # Calculate gradients and update parameters for policy network
        p_grads = p_tape.gradient(p_loss, policy_network.trainable_variables)
        p_optimizer.apply_gradients(zip(p_grads, policy_network.trainable_variables))
        # Calculate gradients and update parameters for value network
        v_grads = v_tape.gradient(v_loss, value_network.trainable_variables)
        v_optimizer.apply_gradients(zip(v_grads, value_network.trainable_variables))
