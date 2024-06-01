import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np 
import random
import game
from environment import DQEnv, WalkerPlayer, DQLearnPlayer, SmartPlayer, DDQLearnPlayer, QLearnPlayer
import datetime
import torch.jit as jit
import copy
import time
import os


# if GPU is to be used
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(device)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def load(load_path, device):
    net = DDQLAgentNet((1, 10, 10), 6).float()
    net = net.to(device=device)
    if not load_path.exists():
        raise ValueError(f"{load_path} does not exist")

    ckp = torch.load(load_path, map_location=device) #('cuda' if self.use_cuda else 'cpu'))
    exploration_rate = ckp.get('exploration_rate')
    state_dict = ckp.get('model')

    #print(f"Loading model at {load_path} with exploration rate {exploration_rate}")
    net.load_state_dict(state_dict)
    exploration_rate = exploration_rate

    return net

class DDQLAgentNet(nn.Module):
    """mini CNN structure
  input -> (conv2d + relu) x 3 -> flatten -> (dense + relu) x 2 -> output
  """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        c, h, w = input_dim

        if h != 10: #84:
            raise ValueError(f"Expecting input height: 10, got: {h}")
        if w != 10: #84:
            raise ValueError(f"Expecting input width: 10, got: {w}")

        self.online = self.__build_cnn(c, output_dim)

        self.target = self.__build_cnn(c, output_dim)
        self.target.load_state_dict(self.online.state_dict())

        # Q_target parameters are frozen.
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, input, model):
        """
        if inputs is not None:
            inputs = inputs.float()###############################################################################
        else:
            print(inputs)
        """
        if model == "online":
            return self.online(input)
        elif model == "target":
            return self.target(input)

    def __build_cnn(self, c, output_dim):
        return nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=16, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )


demonstration = False #True
if demonstration:
    game.HEADLESS = False
    game.TIME_CONST = 0.25 #1
else:
    game.HEADLESS = True
    game.TIME_CONST = 0.01

countVictoriesP1 = 0
countVictoriesW = 0
countVictoriesW1 = 0
countVictoriesW2 = 0

for iteration in range(1000):
    if os.name == 'nt':
        os.system('cls')
    # For macOS and Linux
    else:
        os.system('clear')

    # Initialize the environment and get its state
    game.grid = game.get_start_grid()
    game.alive_players = []
    game.dead_players = []
    game.global_bombs = set()

    #policy_net_copy = jit.load("DQL_model-Colab") 
    #DDQL_Agent_net_4.chkpt
    ############################ DDQL
    
    #'DDQL_model_Dict-SmallerRewards' -> mal mn agresiven; Pol od prej pa idk a tista mapa deluje pa se kj (mogoc vsiStarti/2 pa changingBots)
    #Pac glej datum izdelave al neki

    
    load_path = Path('C:/Users/Adminj/Desktop/UmetnaInteligenca/Bomberman/checkpoints/2024-05-26T19-46-47-DELUJE/DDQL_Agent_net_2.chkpt') #Path('DDQL_model_Dict-PauseGameExtra') #Path('DDQL_model_Dict-PauseGameExtra') #Path('C:/Users/Adminj/Desktop/UmetnaInteligenca/Bomberman/checkpoints/2024-05-26T00-00-10-PauseGame/DDQL_Agent_net_4.chkpt') #Path('DDQL_model_Dict-PauseGame') #Path('DDQL_model_Dict-Normalized')
    #load_path = Path('DDQL_model_Dict-SmallerRewards') #Path('DDQL_model_Dict-ChangingBots') #Path('DDQL_model_Dict-VsiStarti2')
    model = load(load_path, device)
    p1 = DDQLearnPlayer(model, device)#SmartPlayer() #DQLearnPlayer(policy_net_copy, device)
    
    walker = QLearnPlayer()

    #walker = SmartPlayer() #WalkerPlayer() #DQLearnPlayer(policy_net_copy, device) #WalkerPlayer() #None
    #walker1 = SmartPlayer() #WalkerPlayer() #DQLearnPlayer(policy_net_copy, device) #WalkerPlayer() #None
    ############################ DQL
    policy_net_copy = jit.load("DQL_model-3Position")
    walker1 = DQLearnPlayer(policy_net_copy, device)
    
    walker2 = SmartPlayer() #WalkerPlayer() #SmartPlayer() #DQLearnPlayer(policy_net_copy, device) #WalkerPlayer()

    p1.start()
    walker.start()
    walker1.start()
    walker2.start()

    startTime = time.time()
    while True:
        game.print_grid()
        print("DDQL nr. of victories:", countVictoriesP1, "/", iteration+1)
        print("QL nr. of victories:", countVictoriesW, "/", iteration+1)
        print("DQL nr. of victories:", countVictoriesW1, "/", iteration+1)
        print("SmartPlayer nr. of victories:", countVictoriesW2, "/", iteration+1)

        time.sleep(0.0010)
        if len(game.alive_players) < 2 or time.time() - startTime > 30: #or p1.player.dead:
            if len(game.alive_players) < 2:
                if not p1.player.dead:
                    countVictoriesP1 += 1
                elif not walker.player.dead:
                    countVictoriesW += 1
                elif not walker1.player.dead:
                    countVictoriesW1 += 1
                elif not walker2.player.dead:
                    countVictoriesW2 += 1

            p1.stop()
            walker.stop()
            walker1.stop()
            walker2.stop()
            break