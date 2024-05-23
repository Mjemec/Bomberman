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

#from game import * 
import numpy as np 
import random
import game
from environment import DQEnv, WalkerPlayer, DQLearnPlayer
import datetime
import torch.jit as jit
import copy



# set up matplotlib
"""
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display
"""

#plt.ion()

# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)


################################################## Replay memory

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    

################################################## Q-network

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        #print(n_observations, n_actions)
        self.layer1 = nn.Linear(n_observations, 512) #nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(512, 256) #nn.Linear(128, 128)
        #self.layer3 = nn.Linear(256, 256)
        self.layer4 = nn.Linear(256, 128)
        self.layer5 = nn.Linear(128, n_actions)
        #self.layer3 = nn.Linear(256, n_actions) #nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        #x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        return self.layer5(x) #self.layer3(x)
    
################################################## Training

# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer
BATCH_SIZE = 512 #256 #128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 100000 #500000 #80000 #1000
TAU = 0.005
LR = 1e-4

# Get number of actions from gym action space
n_actions = len(game.ACTION_SPACE) #env.action_space.n
# Get the number of state observations
grid = game.get_start_grid()
map_x_len = len(grid[0])
map_y_len = len(grid)

#state, info = env.reset()
n_observations = (map_x_len * map_y_len * 4) + 1 #len(state) # 4 matrices representing entire game

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)


steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1).indices.view(1, 1)
    else: # Uniform random selection of action
        randAction = random.randint(0, n_actions-1)
        return torch.tensor([[randAction]], device=device, dtype=torch.long) #torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)


episode_durations = []


def plot_durations(show_result=False):
    #plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    #if show_result:
    plt.plot(durations_t.numpy())
    plt.title('Result')
    #else:
    #    plt.clf()
    #    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Cumulated Rewards') #plt.ylabel('Duration')
    # Take 100 episode averages and plot them too
    #if len(durations_t) >= 100:
    #    means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
    #    means = torch.cat((torch.zeros(99), means))
    #    plt.plot(means.numpy())

    
    plt.savefig('DQL-rewardsPlot.png') 

    #plt.pause(0.001)  # pause a bit so that plots are updated
    #if is_ipython:
        #if not show_result:
        #    display.display(plt.gcf())
        #    display.clear_output(wait=True)
        #else:
        #   display.display(plt.gcf())

################################################## Training optimizer

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

################################################## Training loop
demonstration = False
if demonstration:
    game.HEADLESS = False
    game.TIME_CONST = 0.25
else:
    game.HEADLESS = True
    game.TIME_CONST = 0.001

average_of_100 = deque([0]*100,maxlen=100)

max_steps = 1000 #Nr of actions before ending game if the player did not win/die

num_episodes = 0
if torch.cuda.is_available():
    num_episodes = 100000 #50000 #10000 #600
#else:
#    num_episodes = 50

timelast = datetime.datetime.now()
last_score = 0
best_score = -float('inf')

policy_net_copy = None
policy_net_copy2 = None

for i_episode in range(num_episodes):
    average_of_100.appendleft(last_score)
    if (i_episode + 1) % 100 == 0:
        time_per_epoch = (datetime.datetime.now() - timelast).total_seconds()/100
        print(f'Epoch {i_episode} of {num_episodes} ({i_episode/num_episodes*100:.2f}%),'
              f' ETA: {(num_episodes-i_episode)*time_per_epoch:.2f} seconds,'
              f' last score: {last_score}, best score: {best_score}'
              f' average of last 100: {sum(average_of_100)/len(average_of_100):.2f}')
        timelast = datetime.datetime.now()

    # Initialize the environment and get its state
    #state, info = env.reset()
    game.grid = game.get_start_grid()
    game.alive_players = []
    game.dead_players = []
    game.global_bombs = set()
    p1 = game.Player('DQL')
    env = DQEnv(p1) #QEnv(p1)

    walker = None
    walker1 = None
    walker2 = WalkerPlayer()
    """
    if i_episode % 10000 == 0 and (i_episode / 10000) % 2 != 0 :
        policy_net_copy = copy.deepcopy(policy_net)
    if i_episode % 10000 == 0 and (i_episode / 10000) % 2 == 0:
        policy_net_copy2 = copy.deepcopy(policy_net)


    if i_episode >= 10000:
        walker = DQLearnPlayer(policy_net_copy, device)
    else:
        walker = WalkerPlayer()
    if i_episode >= 20000:
        walker1 = DQLearnPlayer(policy_net_copy2, device)
    else:
        walker1 = WalkerPlayer()
    """
    walker = WalkerPlayer()
    walker1 = WalkerPlayer()


    walker.start()
    walker1.start()
    walker2.start()

    players_grid, power_up_grid, blocks_grid, bomb_grid = p1.get_self_grid()
    nrBombs = np.array([p1.get_bombs()], dtype=np.uint8)
    state = np.concatenate((players_grid.flatten(), power_up_grid.flatten(), blocks_grid.flatten(), bomb_grid.flatten(), nrBombs))
    #print(state.shape)
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    #print(state.shape)

    total_reward = 0

    for t in range(max_steps): #for t in count():
        action = select_action(state)
        #observation, reward, terminated, truncated, _ = env.step(action.item())
        observation, reward, done, _ = env.step(action)
        total_reward += reward

        reward = torch.tensor([reward], device=device)
        #done = terminated or truncated

        #if done:
        """
        next_state = None
        if not done:
            players_grid2, power_up_grid2, blocks_grid2, bomb_grid2 = p1.get_self_grid()
            nrBombs2 = np.array([p1.get_bombs()], dtype=np.uint8)
            observation = np.concatenate((players_grid2.flatten(), power_up_grid2.flatten(), blocks_grid2.flatten(), bomb_grid2.flatten(), nrBombs2))
            #observation = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
        """
        players_grid2, power_up_grid2, blocks_grid2, bomb_grid2 = p1.get_self_grid()
        nrBombs2 = np.array([p1.get_bombs()], dtype=np.uint8)
        observation = np.concatenate((players_grid2.flatten(), power_up_grid2.flatten(), blocks_grid2.flatten(), bomb_grid2.flatten(), nrBombs2))
        #observation = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        optimize_model()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)


        if done or len(game.alive_players) < 2 or t == max_steps - 1:
            last_score = p1.get_score()
            if last_score > best_score:
                best_score = last_score
            """
            env.reset()
            walker.stop()
            walker1.stop()
            walker2.stop()
            """
            episode_durations.append(total_reward)
            break

        """
        if done or t == max_steps - 1:
            episode_durations.append(total_reward)#t + 1)
            #plot_durations()
            break
        """
    
    env.reset()
    walker.stop()
    walker1.stop()
    walker2.stop()
    

jit.save(torch.jit.script(policy_net), "DQL_model")
#model = jit.load("DQL_model")

print('Complete')
#plot_durations(show_result=True)
#plt.ioff()

#plt.show()
"""
plt.plot(episode_durations)
plt.title('Result')
plt.xlabel('Episode')
plt.ylabel('Cumulated Rewards') #plt.ylabel('Duration')
plt.savefig('DQL-rewardsPlot.png')
plt.close()
"""
np.savetxt("DQL_rewardsList.txt", np.array(episode_durations), delimiter=',')