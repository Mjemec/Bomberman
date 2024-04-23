from game import *
from environment import Environment
import random
import threading

def get_random_action():
    actions = ['up', 'down', 'left', 'right', 'noop', 'bomb']
    mainProb = 0.15#1 / len(actions)
    weights = [mainProb, mainProb, mainProb, mainProb, mainProb * 2, 1 - 6 * mainProb]
    # random action based on the probability of the action
    return random.choices(actions, weights=weights, k=1)[0]


stop_threads = False
def tralala(playername):
    global stop_threads

    player = Player(playername)
    env = Environment(player)
    observetime = 100
    for t in range(observetime):
        if stop_threads:
            break
        action = get_random_action()
        observation_new, reward, done, info = env.step(action)
        if done:
            env.reset()
            print('reseting the environment')
            stop_threads = True
            exit(0)# Restart game if it's finished
            break

if __name__ == '__main__':
    threads = []
    for name in ['Prvi', 'Drugi', 'Tretji', 'Zadnji']:
        th = threading.Thread(target=tralala, args=(name,))
        th.start()
        threads.append(th)

    for thread in threads:
        thread.join()
