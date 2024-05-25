import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np 
import random
import game
from environment import DQEnv, WalkerPlayer, DQLearnPlayer
import datetime
import torch.jit as jit
import copy
import time


# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


demonstration = True
if demonstration:
    game.HEADLESS = False
    game.TIME_CONST = 0.25
else:
    game.HEADLESS = True
    game.TIME_CONST = 0.001



# Initialize the environment and get its state
game.grid = game.get_start_grid()
game.alive_players = []
game.dead_players = []
game.global_bombs = set()

policy_net_copy = jit.load("DQL_model-Colab")
p1 = DQLearnPlayer(policy_net_copy, device)

walker = WalkerPlayer() #DQLearnPlayer(policy_net_copy, device) #WalkerPlayer() #None
walker1 = WalkerPlayer() #DQLearnPlayer(policy_net_copy, device) #WalkerPlayer() #None
walker2 = WalkerPlayer() #DQLearnPlayer(policy_net_copy, device) #WalkerPlayer()

p1.start()
walker.start()
walker1.start()
walker2.start()


while True:
    game.print_grid()
    time.sleep(0.0010)
    if len(game.alive_players) < 2:
        p1.stop()
        walker.stop()
        walker1.stop()
        walker2.stop()
        break
    