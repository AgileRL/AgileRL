import torch.nn as nn
import numpy as np
import gymnasium as gym
from gymnasium.wrappers.atari_preprocessing import AtariPreprocessing
import yaml
import supersuit as ss
from tqdm import trange
import torch
import os
import sys
from pettingzoo.mpe import simple_speaker_listener_v4

sys.path.append('../')

from agilerl.wrappers.make_evolvable import MakeEvolvable

from agilerl.components.replay_buffer import ReplayBuffer
from agilerl.components.multi_agent_replay_buffer import MultiAgentReplayBuffer
from agilerl.hpo.mutation import Mutations
from agilerl.training.train_on_policy import train_on_policy
from agilerl.training.train_multi_agent import train_multi_agent
from agilerl.networks.custom_activation import GumbelSoftmax
from agilerl.training.train import train
from agilerl.hpo.tournament import TournamentSelection
from agilerl.utils.utils import initialPopulation, makeVectEnvs, printHyperparams


from pettingzoo.atari import pong_v3

from networks import ClipReward, BasicNetActor, BasicNetCritic, SimpleCNNActor, \
SimpleCNNCritic, MultiCNNActor, MultiCNNCritic

device = "cuda" if torch.cuda.is_available() else "cpu"

network_actor = BasicNetActor(6 ,[64, 64], 4)
actor = MakeEvolvable(network_actor,
                      input_tensor=torch.ones(6),
                      device=device)
critic = None


new_actor = actor.clone()

def compare_models(model_1, model_2):
    models_differ = 0
    for key_item_1, key_item_2 in zip(model_1.state_dict().items(), model_2.state_dict().items()):
        if torch.equal(key_item_1[1], key_item_2[1]):
            pass
        else:
            models_differ += 1
            if (key_item_1[0] == key_item_2[0]):
                print('Mismtach found at', key_item_1[0])
            else:
                raise Exception
    if models_differ == 0:
        print('Models match perfectly! :)')

compare_models(new_actor, actor)