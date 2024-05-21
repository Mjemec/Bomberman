import os.path
import datetime
import argparse
import time

import numpy as np
import game
from collections import deque
from environment import QEnv, WalkerPlayer


class CustomDeque(deque):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def appendleft(self, element):
        removed_element = None
        if len(self) == self.maxlen:
            removed_element = self.pop()
        super().appendleft(element)
        return removed_element

    def append(self, element):
        removed_element = None
        if len(self) == self.maxlen:
            removed_element = self.popleft()
        super().append(element)
        return removed_element


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A script with a boolean argument")
    parser.add_argument('--headless', action='store_true', help='Do not display the game, increases speed')
    parser.add_argument('--epochs', '-n', type=int, help='Number of epochs', default=10000)
    args = parser.parse_args()
    n_states = 4**8*2**6*3**2  # Number of states in the grid world
    n_actions = 6  # Number of possible actions (up, down, left, right, noop, bomb)

    if os.path.exists('array.npy'):
        Q_table = np.load('array.npy')
    else:
        Q_table = np.zeros((n_states, n_actions))

    # Define parameters
    learning_rate = 0.075
    discount_factor = 0.5
    decay_factor = 0.25
    exploration_prob = 0.03
    epochs = args.epochs
    max_steps = 1000
    # set this for demonstration purposes
    demonstration = not args.headless

    if demonstration:
        game.HEADLESS = False
        game.TIME_CONST = 0.25
    else:
        game.HEADLESS = True
        game.TIME_CONST = 0.0005

    timelast = datetime.datetime.now()
    last_score = 0
    best_score = -float('inf')
    average_of_5 = deque([0]*100,maxlen=100)
    last_decisions = CustomDeque(maxlen=4)

    for epoch in range(epochs):  # TODO: fix this
        average_of_5.appendleft(last_score)
        #exploration_prob = 0.5*np.e**(-0.00062146*epoch)
        if epoch % 100 == 0:
            time_per_epoch = (datetime.datetime.now() - timelast).total_seconds()/100
            print(f'Epoch {epoch} of {epochs} ({epoch/epochs*100:.2f}%),'
                  f' ETA: {(epochs-epoch)*time_per_epoch:.2f} seconds,'
                  f' last score: {last_score}, best score: {best_score}, average of last 10: {sum(average_of_5)/len(average_of_5):.2f}')
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
            is_exploration_move = False
            if demonstration:
                rounded = [e.__format__('010.2f') for e in Q_table[current_state]]
                print(f'state: {current_state:10} table: {rounded}')

            if np.random.rand() < 0.1:
                action = 5
            elif np.random.rand() < exploration_prob:
                action = np.random.randint(0, n_actions)
                is_exploration_move = True
            else:
                if len(last_decisions) and last_decisions[0][1] < 4:
                    lastmovegroup = int((last_decisions[0][1]//2)*2) + 1 - last_decisions[0][1] % 2
                    temp = Q_table[current_state].copy()
                    temp[lastmovegroup] = -np.inf
                    action = np.argmax(temp)
                elif len(last_decisions):
                    noop_count = 0
                    for state, decision in last_decisions:
                        if decision == 4:
                            noop_count += 1
                        else:
                            break
                    if noop_count > 4:
                        temp = Q_table[current_state].copy()
                        temp[4] = -np.inf
                        action = np.argmax(temp)
                    else:
                        action = np.argmax(Q_table[current_state])
                else:
                    action = np.argmax(Q_table[current_state])
                is_exploration_move = False

            if demonstration:
                print(f'action: {game.ACTION_SPACE[action]}, is exploratory move: {is_exploration_move}')
            observation, reward, done, _ = env.step(action)

            t = last_decisions.appendleft((current_state, action))

            next_state = env.get_state()

            for index, i in enumerate(last_decisions):
                Q_table[i] += learning_rate * decay_factor**index * reward

            # Q_table[current_state, action] += learning_rate * \
            #     (reward + discount_factor *
            #      np.max(Q_table[next_state]) - Q_table[current_state, action])

            if done:
                if demonstration:
                    print(f'action traceback: {last_decisions}')
                    pass
                last_score = p1.get_score()
                if last_score > best_score:
                    best_score = last_score
                walker.stop()
                walker1.stop()
                walker2.stop()
                env.reset()
                break

            current_state = next_state

    print("Learned Q-table:")
    np.save('array.npy', Q_table)
