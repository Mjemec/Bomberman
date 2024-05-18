import os.path
import datetime
import numpy as np
import game
from environment import QEnv, WalkerPlayer

n_states = 3**14  # Number of states in the grid world
n_actions = 6  # Number of possible actions (up, down, left, right, noop, bomb)

if os.path.exists('array.npy'):
    Q_table = np.load('array.npy')
else:
    Q_table = np.zeros((n_states, n_actions))

# Define parameters
learning_rate = 0.8
discount_factor = 0.95
exploration_prob = 0.05
epochs = 2000
max_steps = 1000

# set this for demonstration purposes
demonstration = False
if demonstration:
    game.HEADLESS = False
    game.TIME_CONST = 1
else:
    game.HEADLESS = True
    game.TIME_CONST = 0.001

timelast = datetime.datetime.now()
last_score = 0
best_score = -float('inf')

for epoch in range(epochs):  # TODO: fix this
    if epoch % 100 == 0:
        time_per_epoch = (datetime.datetime.now() - timelast).total_seconds()/100
        print(f'Epoch {epoch} of {epochs} ({epoch/epochs*100:.2f}%),'
              f' ETA: {(epochs-epoch)*time_per_epoch:.2f} seconds,'
              f' last score: {last_score}, best score: {best_score}')
        timelast = datetime.datetime.now()
    game.grid = game.get_start_grid()
    game.alive_players = []
    game.dead_players = []
    game.global_bombs = set()
    p1 = game.Player('QL')
    env = QEnv(p1)
    walker = WalkerPlayer()
    walker1 = WalkerPlayer()
    walker2 = WalkerPlayer()
    walker.start()
    walker1.start()
    walker2.start()

    for _ in range(max_steps):
        current_state = env.get_state()
        if np.random.rand() < exploration_prob:
            action = np.random.randint(0, n_actions)
        else:
            action = np.argmax(Q_table[current_state])

        observation, reward, done, _ = env.step(action)

        next_state = env.get_state()

        Q_table[current_state, action] += learning_rate * \
            (reward + discount_factor *
             np.max(Q_table[next_state]) - Q_table[current_state, action])

        if done or len(game.alive_players) < 2:
            last_score = p1.get_score()
            if last_score > best_score:
                best_score = last_score
            env.reset()
            walker.stop()
            break

        current_state = next_state

print("Learned Q-table:")
np.save('array.npy', Q_table)
