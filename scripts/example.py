# Derived from keras-rl
# to run: python example.py --train --model sample
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
import opensim as osim
import numpy as np
import sys

from keras import backend as be
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Lambda, concatenate
from keras.optimizers import Adam

import numpy as np

from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess

from osim.env import *
from osim.http.client import Client

from keras.optimizers import RMSprop
from keras.layers.advanced_activations import LeakyReLU

import argparse
import math

# Command line parameters
parser = argparse.ArgumentParser(description='Train or test neural net motor controller')
parser.add_argument('--train', dest='train', action='store_true', default=True)
parser.add_argument('--test', dest='train', action='store_false', default=True)
parser.add_argument('--steps', dest='steps', action='store', default=10000, type=int)
parser.add_argument('--visualize', dest='visualize', action='store_true', default=False)
parser.add_argument('--model', dest='model', action='store', default="example.h5f")
parser.add_argument('--token', dest='token', action='store', required=False)
args = parser.parse_args()

# Load walking environment
env = RunEnv(args.visualize)
env.reset(difficulty = 0, seed = None)

nb_actions = env.action_space.shape[0]#/2

# Total number of steps in training
nallsteps = args.steps

## Preprocessing
#def relativize(x):
#	return be.concatenate([
#		x[:,:,0],
#		x[:,:,1:2] - x[:,:,18:19],
#		x[:,:,3:21],
#		x[:,:,22:23] - x[:,:,18:19],
#		x[:,:,24:25] - x[:,:,18:19],
#		x[:,:,26:27] - x[:,:,18:19],
#		x[:,:,28:29] - x[:,:,18:19],
#		x[:,:,30:31] - x[:,:,18:19],
#		x[:,:,32:33] - x[:,:,18:19],
#		x[:,:,34:35] - x[:,:,18:19],
#		x[:,:,36:40],
#	], axis=2)

#relativize_layer = Lambda(relativize, input_shape=(1,) + env.observation_space.shape, output_shape=(41,))

#print(env.observation_space.shape)
#print((1,) + env.observation_space.shape)

# Create networks for DDPG
# Next, we build a very simple model.
leak_const = 0.2
actor = Sequential()
actor.add(Flatten(input_shape=(1,) + env.observation_space.shape))
#actor.add(relativize_layer)
actor.add(Dense(32))
actor.add(LeakyReLU(alpha=leak_const))
actor.add(Dense(32))
actor.add(LeakyReLU(alpha=leak_const))
actor.add(Dense(32))
actor.add(LeakyReLU(alpha=leak_const))
actor.add(Dense(nb_actions))
actor.add(Activation('sigmoid'))
print(actor.summary())

action_input = Input(shape=(nb_actions,), name='action_input')
observation_input = Input(shape=(1,) + env.observation_space.shape, name='observation_input')
flattened_observation = Flatten()(observation_input)
x = concatenate([action_input, flattened_observation])
x = Dense(64)(x)
x = LeakyReLU(alpha=leak_const)(x)
x = Dense(64)(x)
x = LeakyReLU(alpha=leak_const)(x)
x = Dense(64)(x)
x = LeakyReLU(alpha=leak_const)(x)
x = Dense(1)(x)
x = Activation('linear')(x)
critic = Model(inputs=[action_input, observation_input], outputs=x)
print(critic.summary())

# Set up the agent for training

memory = SequentialMemory(limit=100000, window_length=1)
random_process = OrnsteinUhlenbeckProcess(theta=.15, mu=0., sigma=.2, size=env.noutput)
agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input, memory=memory, nb_steps_warmup_critic=100, nb_steps_warmup_actor=100, random_process=random_process, gamma=.99, target_model_update=1e-3, delta_clip=1.)

agent.compile(Adam(lr=.001, clipnorm=1.), metrics=['mae'])

#quit()

# Learning

if args.train:
    agent.fit(env, nb_steps=nallsteps, visualize=False, verbose=1, nb_max_episode_steps=env.timestep_limit, log_interval=10000)
    # After training is done, we save the final weights.
    agent.save_weights(args.model, overwrite=True)

# If TEST and no TOKEN, run some test experiments
if not args.train and not args.token:
    agent.load_weights(args.model)
    # Finally, evaluate our algorithm for 1 episode.
    agent.test(env, nb_episodes=1, visualize=False, nb_max_episode_steps=500)
