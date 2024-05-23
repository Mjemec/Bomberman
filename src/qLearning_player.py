import os.path
import datetime
import argparse
import threading
import time
import numpy as np
import game
from collections import deque
from environment import QEnv, WalkerPlayer, QLearnPlayer
import environment

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


def learn_to_dodge(epochs: int, Q_table: np.array, demonstration):
    timelast = datetime.datetime.now()
    learning_rate = 0.1
    decay_factor = np.log(0.5 / 0.001) / epochs
    discount_factor = 0.5
    max_steps = 100
    n_actions = len(game.ACTION_SPACE)
    dq_steps = deque(maxlen=100)

    for epoch in range(epochs):
        exploration_prob = 0.5 * np.exp(-decay_factor * epoch)
        if epoch % 100 == 0:
            avg_steps = f' AVERAGE steps: {sum(dq_steps)/len(dq_steps):.2f}' if len(dq_steps) else ''
            time_per_epoch = (datetime.datetime.now() - timelast).total_seconds() / 100
            print(f'Epoch {epoch} of {epochs} ({epoch / epochs * 100:.2f}%),'
                  f' ETA: {(epochs - epoch) * time_per_epoch:.2f} seconds', avg_steps)
            timelast = datetime.datetime.now()
        while_ctr = 0
        game.alive_players.clear()
        game.dead_players.clear()
        environment.threads.clear()
        game.global_bombs.clear()
        while len(game.alive_players) or len(game.global_bombs) or len(environment.threads) or len(game.dead_players):
            if while_ctr > 1000:
                pass
            while_ctr += 1
            time.sleep(0.001)

        game.grid = game.get_start_grid()

        walker = WalkerPlayer()

        if epoch > epochs//2:
            p1 = game.Player('QL', max_bombs=0)
        else:
            p1 = game.Player('QL')

        env = QEnv(p1)
        walker1 = WalkerPlayer()
        walker2 = WalkerPlayer()
        walker.start()
        walker1.start()
        walker2.start()

        for step in range(max_steps):
            current_state = env.get_state()
            is_exploration_move = False
            if demonstration:
                rounded = [e.__format__('.2f') for e in Q_table[current_state]]
                print(f'state: {current_state} table: {rounded}')
            if np.random.rand() < 0.1:  # 10% chance of placing a bomb
                p1.place_bomb()
                continue
            elif np.random.rand() < exploration_prob:
                action = np.random.randint(0, n_actions)

                is_exploration_move = True
            else:
                action = np.argmax(Q_table[current_state])
                is_exploration_move = False
            if action == 5:
                continue  # we do not let the player place bombs
            if demonstration:
                print(f'action: {game.ACTION_SPACE[action]}, is exploratory move: {is_exploration_move}')
            observation, reward, done, _ = env.step(action)
            next_state = env.get_state()
            Q_table[current_state, action] += learning_rate * \
                                              (reward + discount_factor *
                                               np.max(Q_table[next_state]) - Q_table[current_state, action])
            if done:
                dq_steps.append(step)
                walker.stop()
                walker1.stop()
                walker2.stop()
                env.reset()
                if len(game.alive_players):
                    pass
                break


def main():
    parser = argparse.ArgumentParser(description="Bomberman")
    parser.add_argument('--headless', action='store_true', help='Do not display the game, increases speed')
    parser.add_argument('--epochs', '-n', type=int, help='Number of epochs', default=10000)
    parser.add_argument('--learn_to_dodge', '-d', action='store_true', help='Number of epochs')
    args = parser.parse_args()
    n_states = 5**8*2**6*3**2  # Number of states in the grid world
    n_actions = 6  # Number of possible actions (up, down, left, right, noop, bomb)

    learning_rate = 0.08
    discount_factor = 0.5
    decay_factor = 0.25
    exploration_prob = 0.0675
    epochs = args.epochs
    max_steps = 1000
    demonstration = not args.headless

    if demonstration:
        game.HEADLESS = False
        game.TIME_CONST = 0.25
    else:
        game.HEADLESS = True
        game.TIME_CONST = 0.0005

    if os.path.exists('array.npy'):
        Q_table = np.load('array.npy')
    else:
        Q_table = np.zeros((n_states, n_actions))

    if args.learn_to_dodge:
        learn_to_dodge(15000, Q_table, demonstration)
        np.save('array.npy', Q_table)

    timelast = datetime.datetime.now()
    last_score = 0
    best_score = -float('inf')
    average_of_100 = deque([0] * 100, maxlen=100)
    last_decisions = CustomDeque(maxlen=2)

    for epoch in range(epochs):
        average_of_100.appendleft(last_score)
        if epoch % 100 == 0:
            time_per_epoch = (datetime.datetime.now() - timelast).total_seconds()/100
            print(f'Epoch {epoch} of {epochs} ({epoch/epochs*100:.2f}%),'
                  f' ETA: {(epochs-epoch)*time_per_epoch:.2f} seconds,'
                  f' last score: {last_score}, best score: {best_score}, average of last 10: {sum(average_of_100) / len(average_of_100):.2f}')
            timelast = datetime.datetime.now()
        game.grid_lock.acquire()
        game.grid = game.get_start_grid()
        game.grid_lock.release()
        game.alive_players = []
        game.dead_players = []
        game.global_bombs = set()
        walker = WalkerPlayer()
        p1 = game.Player('QL')
        env = QEnv(p1)
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
                print(f'\rstate: {current_state:10} table: {rounded}')

            if np.random.rand() < exploration_prob:
                action = np.random.randint(0, n_actions)
                is_exploration_move = True
            else:
                action = np.argmax(Q_table[current_state])
                is_exploration_move = False

            if demonstration:
                print(f'\raction: {game.ACTION_SPACE[action]}, is exploratory move: {is_exploration_move}')
            observation, reward, done, _ = env.step(action)
            t = last_decisions.append((current_state, action))

            next_state = env.get_state()

            Q_table[current_state, action] += learning_rate * \
                (reward + discount_factor *
                 np.max(Q_table[next_state]) - Q_table[current_state, action])

            if done or len(game.alive_players) < 2:
                if demonstration:
                    print(f'action traceback: {last_decisions}')
                    pass
                # if env.player.dead:
                #     for i, tpl_decision in enumerate(last_decisions):
                if env.player.dead:
                    Q_table[current_state, action] -= learning_rate*1000
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

if __name__ == "__main__":
    main()
