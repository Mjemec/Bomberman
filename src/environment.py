import time
import game
import threading
import random
import numpy as np
import torch
import torch.jit as jit


class WalkerPlayer:
    player: game.Player

    def __init__(self):
        self.player = game.Player('Walker')
        self._stop_event = threading.Event()
        self._thread = None

    def _run(self):
        while not self._stop_event.is_set() and not self.player.dead:
            weights = [0.198, 0.198, 0.198, 0.198, 0.198, 0.01]
            action = random.choices(game.ACTION_SPACE, weights=weights, k=1)[0]
            if action == 'bomb':
                self.player.place_bomb()
            else:
                self.player.move(action)

    def start(self):
        if self._thread is None or not self._thread.is_alive():
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._run)
            self._thread.start()

    def stop(self):
        if self._thread is not None:
            self._stop_event.set()
            self._thread.join()


class QLearnPlayer:
    Q_table: np.array

    def __init__(self, player):
        self.Q_table = np.load('array.npy')
        self.player = game.Player('QWalker')
        self.env = QEnv(self.player)
        self._stop_event = threading.Event()
        self._thread = None

    def _run(self):
        done = False
        while not self._stop_event.is_set() and not done:
            current_state = self.env.get_state()
            action = np.argmax(self.Q_table[current_state])
            observation, reward, done, _ = self.env.step(action)

    def start(self):
        if self._thread is None or not self._thread.is_alive():
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._run)
            self._thread.start()

    def stop(self):
        if self._thread is not None:
            self._stop_event.set()
            self._thread.join()


class DQLearnPlayer:

    def __init__(self, model, device): #player,
        self.policy_net = model #jit.load(modelFile)
        self.policy_net.eval()
        self.device = device

        self.player = game.Player('DQWalker')
        self.env = DQEnv(self.player)
        self._stop_event = threading.Event()
        self._thread = None

    def _run(self):
        players_grid, power_up_grid, blocks_grid, bomb_grid = self.player.get_self_grid()
        state = np.concatenate((players_grid.flatten(), power_up_grid.flatten(), blocks_grid.flatten(), bomb_grid.flatten()))
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
    
        done = False
        while not self._stop_event.is_set() and not done:
            action = None
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                action = self.policy_net(state).max(1).indices.view(1, 1)
        
            observation, reward, done, _ = self.env.step(action)

            players_grid2, power_up_grid2, blocks_grid2, bomb_grid2 = self.player.get_self_grid()
            observation = np.concatenate((players_grid2.flatten(), power_up_grid2.flatten(), blocks_grid2.flatten(), bomb_grid2.flatten()))
            next_state = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)

            state = next_state


    def start(self):
        if self._thread is None or not self._thread.is_alive():
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._run)
            self._thread.start()

    def stop(self):
        if self._thread is not None:
            self._stop_event.set()
            self._thread.join()


class Environment:
    player : game.Player

    def __init__(self, player):
        self.player = player

    def reset(self):
        name = self.player.name
        while len(game.alive_players):
            p = game.alive_players[0]
            if p.name == 'QL':
                pass
            p.terminate()

        while len(game.dead_players):
            p = game.dead_players[0]
            game.dead_players.remove(p)
        game.grid = game.get_start_grid()
        timeout_ctr = 10
        while len(game.alive_players) > 0:
            time.sleep(0.01)
            assert timeout_ctr > 0, 'WTF is garbage not working?'

    def step(self, action):
        r1 = self.player.score
        moved = True
        if action != 5:
            moved = self.player.move(game.ACTION_SPACE[action])
        else:
            self.player.place_bomb()
        r2 = self.player.score
        done = self.player.dead  # to je fertik
        observation_new = ''  # todo
        info = ''  # recimo
        reward = r2 - r1  # todo
        if moved:
            reward -= 1
        else:
            reward -= 2
        return observation_new, reward, done, info


class QEnv(Environment):
    def __init__(self, player):
        super().__init__(player)

    """
    Use enums to shrink space
    Qstate = [int[8-neigh],
              int[4dirBombCriticalityEnum]{NO_BOMB_OR_LOTS_TIME(>2*speed), WILL_EXPLODE_JUST_NOW},
              int[2 signed dir (y, x)]{NEGATIVE(left , down), ON_POSITION, POSITIVE(right, up)}]
    """
    def get_state(self):
        neigh8 = []
        x = self.player.get_position()[1]
        y = self.player.get_position()[0]
        for i in range(-1, 2, 1):
            for j in range(-1, 2, 1):
                if i == 0 and j == 0:
                    continue

                match game.grid[y-i][x-j][-1]:
                    case game.Wall():  # destroyable
                        neigh8.append(1)
                    case game.Border():
                        neigh8.append(2)
                    case _:
                        neigh8.append(0)

        dir_bomb_severity = [0, 0, 0, 0]  # up, right, down, left
        game.global_bomb_lock.acquire()
        for bomb in game.global_bombs:
            b_x = bomb.x
            b_y = bomb.y
            distance = game.get_distance(x, y, b_x, b_y)
            if distance > bomb.strength:
                continue
            if b_x == x:
                if b_y < y:
                    if bomb.get_time_left_ms() < 2* self.player.speed:
                        dir_bomb_severity[0] = 1
                else:
                    if bomb.get_time_left_ms() < 2* self.player.speed:
                        dir_bomb_severity[2] = 1

            if b_y == y:
                if b_x > x:
                    if bomb.get_time_left_ms() < 2 * self.player.speed:
                        dir_bomb_severity[1] = 1
                else:
                    if bomb.get_time_left_ms() < 2 * self.player.speed:
                        dir_bomb_severity[3] = 1

        min_distance = float("inf")
        direction = [0, 0]
        for player in game.alive_players:
            if player == self.player:
                continue
            p_y, p_x = player.get_position()
            p_dist = game.get_distance(x,y, p_x, p_y)
            if p_dist < min_distance:
                min_distance = p_dist
                if y < p_y:
                    direction[0] = 1
                elif y > p_y:
                    direction[0] = -1
                else:
                    direction[0] = 0

                if x > p_x:
                    direction[1] = -1
                elif x < p_x:
                    direction[1] = 1
                else:
                    direction[1] = 0
        game.global_bomb_lock.release()
        neigh8.extend(direction)
        neigh8.extend(dir_bomb_severity)

        return QEnv.encode_state(neigh8)

    @staticmethod
    def encode_state(state) -> int:
        result = 0
        for j in range(10):
            result += state[j]**(3 + j)

        for i in range(10, 14):
            result += state[i]*2**i
        return result

class DQEnv(Environment):
    def __init__(self, player):
        super().__init__(player)
