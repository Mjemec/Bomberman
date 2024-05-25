import time
import game
import threading
import random
import numpy as np
import torch
import torch.jit as jit

ILLIGAL_PENALTY = 10 #5 #500

threads_lock = game.LogedLock()
threads = dict()

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
            threads_lock.acquire()
            threads[self] = self._thread
            threads_lock.release()
            self._thread.start()

    def stop(self):
        if self._thread is not None:
            self._stop_event.set()
            if self._thread.is_alive():
                self._thread.join()
            threads_lock.acquire()
            #try:
            threads.pop(self)
            #except KeyError:
            #    pass
            threads_lock.release()
        else:
            pass


QLearnPlayerQ_table = None


class QLearnPlayer:
    Q_table: np.array

    def __init__(self):
        global QLearnPlayerQ_table
        if QLearnPlayerQ_table is None:
            QLearnPlayerQ_table = np.load('array.npy')
        self.Q_table = QLearnPlayerQ_table
        self.player = game.Player('QWalker')
        self.env = QEnv(self.player)
        self._stop_event = threading.Event()
        self._thread = None

    def _run(self):
        done = False
        while not self._stop_event.is_set() and not done:
            current_state = self.env.get_state()
            action = np.argmax(self.Q_table[current_state]) if np.random.rand() < 0.05 else np.random.randint(0, 6)
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
        nrBombs = np.array([self.player.get_bombs()], dtype=np.uint8)
        state = np.concatenate((players_grid.flatten(), power_up_grid.flatten(), blocks_grid.flatten(), bomb_grid.flatten(), nrBombs))
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
            nrBombs2 = np.array([self.player.get_bombs()], dtype=np.uint8)
            observation = np.concatenate((players_grid2.flatten(), power_up_grid2.flatten(), blocks_grid2.flatten(), bomb_grid2.flatten(), nrBombs2))
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


class DDQLearnPlayer:

    def __init__(self, model, device): #player,
        self.policy_net = model #jit.load(modelFile)
        self.policy_net.eval()
        self.device = device

        self.player = game.Player('DDQWalker')
        self.env = DQEnv(self.player)
        self._stop_event = threading.Event()
        self._thread = None

    def _run(self):
        #players_grid, power_up_grid, blocks_grid, bomb_grid = self.player.get_self_grid()
        #state = np.stack((players_grid, power_up_grid, blocks_grid, bomb_grid))
        state = self.player.get_self_entire_grid()
        state = state / np.max(state)
    
        done = False
        while not self._stop_event.is_set() and not done:
            action = None
            with torch.no_grad():
                state = state[0].__array__() if isinstance(state, tuple) else state.__array__()
                state = torch.tensor(state, device=self.device).unsqueeze(0)
                state = state.float()
                action_values = self.policy_net(state, model="online")
                action = torch.argmax(action_values, axis=1).item()
        
            observation, reward, done, _ = self.env.step(action)

            #players_grid2, power_up_grid2, blocks_grid2, bomb_grid2 = self.player.get_self_grid()
            #next_state = np.stack((players_grid2, power_up_grid2, blocks_grid2, bomb_grid2))
            state = self.player.get_self_entire_grid()
            state = state / np.max(state)

            #state = next_state


    def start(self):
        if self._thread is None or not self._thread.is_alive():
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._run)
            self._thread.start()

    def stop(self):
        if self._thread is not None:
            self._stop_event.set()
            self._thread.join()


class SmartPlayer:
    player: game.Player

    def __init__(self):
        self.player = game.Player('Smart')
        self._stop_event = threading.Event()
        self._thread = None
        #self.env = DQEnv(self.player)
        self.spottingRange = 1#2
        self.prevMove = None
        self.performedMoves = []

    def _run(self):
        while not self._stop_event.is_set() and not self.player.dead:
            players_grid, power_up_grid, blocks_grid, bomb_grid = self.player.get_self_gridSmart() #.get_self_grid()

            playerLocation = np.argwhere(players_grid == 1)#[0]
            #if len(playerLocation) == 0:
            #    break

            playerLocation = playerLocation[0]

            #action = None
            weights = [0.2, 0.2, 0.2, 0.2, 0.2, 0.0]
            action = random.choices(game.ACTION_SPACE, weights=weights, k=1)[0]

            for priorityRange in range(1, self.spottingRange + 1): #('up', 'down', 'left', 'right', 'noop', 'bomb')

                xStart = None
                xEnd = None
                yStart = None
                yEnd = None

                if playerLocation[0] - priorityRange < 0:
                    xStart = 0
                else:
                    xStart = playerLocation[0] - priorityRange

                if playerLocation[0] + priorityRange >= len(players_grid):
                    xEnd = len(players_grid) - 1
                else:
                    xEnd = playerLocation[0] + priorityRange

                if playerLocation[1] - priorityRange < 0:
                    yStart = 0
                else:
                    yStart = playerLocation[1] - priorityRange

                if playerLocation[1] + priorityRange >= len(players_grid[0]):
                    yEnd = len(players_grid[0]) - 1
                else:
                    yEnd = playerLocation[1] + priorityRange

                mask = np.ones((xEnd - xStart + 1, yEnd - yStart + 1))
                rowsCols = np.array([[priorityRange-1, priorityRange-1], 
                                     [priorityRange-1, priorityRange+1],
                                     [priorityRange+1, priorityRange-1], 
                                     [priorityRange+1, priorityRange+1]])
                #cols = np.array([priorityRange-1, priorityRange+1, priorityRange-1, priorityRange+1])
                for row, col in rowsCols:
                    if row < 0 or row >= len(mask) or col < 0 or col >= len(mask[0]):
                        continue
                    mask[row, col] = 0
                mask = mask.astype(bool)

                if priorityRange == 1:
                    nearBombs = np.argwhere(bomb_grid[xStart:xEnd + 1, yStart:yEnd + 1] == 1) #!= 1e6)
                    if len(nearBombs) != 0 and not np.array_equal(self.prevMove, playerLocation): #self.prevMove != playerLocation:
                        self.prevMove = playerLocation
                        firstBombPos = [nearBombs[0,0] + xStart, nearBombs[0,1] + yStart]
                        if playerLocation[1] - firstBombPos[1] == -1:
                            action = game.ACTION_SPACE[2] #left
                        elif playerLocation[1] - firstBombPos[1] == 1:
                            action = game.ACTION_SPACE[3] #right
                        elif playerLocation[0] - firstBombPos[0] == -1:
                            action = game.ACTION_SPACE[0] #up
                        elif playerLocation[0] - firstBombPos[0] == 1:
                            action = game.ACTION_SPACE[1] #down
                        else:
                            #weights = [0.2, 0.2, 0.2, 0.2, 0.2, 0.0]
                            #action = random.choices(game.ACTION_SPACE, weights=weights, k=1)[0]
                            if blocks_grid[playerLocation[0], playerLocation[1] - 1] == 0 and bomb_grid[playerLocation[0], playerLocation[1] - 1] == 0:
                                action = game.ACTION_SPACE[2] #left
                            elif blocks_grid[playerLocation[0], playerLocation[1] + 1] == 0 and bomb_grid[playerLocation[0], playerLocation[1] + 1] == 0:
                                action = game.ACTION_SPACE[3] #right
                            elif blocks_grid[playerLocation[0] + 1, playerLocation[1]] == 0 and bomb_grid[playerLocation[0] + 1, playerLocation[1]] == 0:
                                action = game.ACTION_SPACE[1] #down
                            else:
                                action = game.ACTION_SPACE[0] #up
                        break
                    elif len(nearBombs) != 0 and np.array_equal(self.prevMove, playerLocation):
                        self.prevMove = playerLocation
                        #weights = [0.2, 0.2, 0.2, 0.2, 0.2, 0.0]
                        #action = random.choices(game.ACTION_SPACE, weights=weights, k=1)[0]
                        if bomb_grid[playerLocation[0]-1, playerLocation[1] - 1] == 1 or bomb_grid[playerLocation[0]-1, playerLocation[1] + 1] == 1 or bomb_grid[playerLocation[0]+1, playerLocation[1] - 1] == 1 or bomb_grid[playerLocation[0]+1, playerLocation[1] + 1] == 1:
                            action = game.ACTION_SPACE[4] #noop
                        elif blocks_grid[playerLocation[0], playerLocation[1] - 1] == 0 and bomb_grid[playerLocation[0], playerLocation[1] - 1] == 0 and players_grid[playerLocation[0], playerLocation[1] - 1] == 0:
                            action = game.ACTION_SPACE[2] #left
                        elif blocks_grid[playerLocation[0], playerLocation[1] + 1] == 0 and bomb_grid[playerLocation[0], playerLocation[1] + 1] == 0 and players_grid[playerLocation[0], playerLocation[1] + 1] == 0:
                            action = game.ACTION_SPACE[3] #right
                        elif blocks_grid[playerLocation[0] + 1, playerLocation[1]] == 0 and bomb_grid[playerLocation[0] + 1, playerLocation[1]] == 0 and players_grid[playerLocation[0] + 1, playerLocation[1]] == 0:
                            action = game.ACTION_SPACE[1] #down
                        else:
                            action = game.ACTION_SPACE[0] #up
                        break
                    

                #else:
                tmp_blocks_grid = blocks_grid[xStart:xEnd + 1, yStart:yEnd + 1]
                tmp_blocks_grid[~mask] = 0
                nearBoxes = np.argwhere(tmp_blocks_grid == 1)
                if len(nearBoxes) != 0:
                    self.prevMove = playerLocation
                    firstBoxPos = [nearBoxes[0,0] + xStart, nearBoxes[0,1] + yStart] #nearBoxes[0]
                    if (playerLocation[1] - firstBoxPos[1] == 0 or playerLocation[0] - firstBoxPos[0] == 0) and priorityRange == 1:
                        action = game.ACTION_SPACE[5] #Bomb
                    elif playerLocation[1] - firstBoxPos[1] < 0:
                        action = game.ACTION_SPACE[3] #right
                    elif playerLocation[1] - firstBoxPos[1] > 0:
                        action = game.ACTION_SPACE[2] #left
                    elif playerLocation[0] - firstBoxPos[0] < 0:
                        action = game.ACTION_SPACE[1] #down
                    elif playerLocation[0] - firstBoxPos[0] > 0:
                        action = game.ACTION_SPACE[0] #up
                    else:
                        weights = [0.2, 0.2, 0.2, 0.2, 0.2, 0.0]
                        action = random.choices(game.ACTION_SPACE, weights=weights, k=1)[0]
                    break
                    
                tmp_players_grid = players_grid[xStart:xEnd + 1, yStart:yEnd + 1]
                tmp_players_grid[~mask] = 0
                nearPlayers = np.argwhere(tmp_players_grid == -1)
                if len(nearPlayers) != 0:
                    self.prevMove = playerLocation
                    firstPlayerPos = [nearPlayers[0,0] + xStart, nearPlayers[0,1] + yStart] #nearPlayers[0]
                    if (playerLocation[1] - firstPlayerPos[1] == 0 or playerLocation[0] - firstPlayerPos[0] == 0) and priorityRange == 1:
                        action = game.ACTION_SPACE[5] #Bomb
                    elif playerLocation[1] - firstPlayerPos[1] < 0:
                        action = game.ACTION_SPACE[3] #right
                    elif playerLocation[1] - firstPlayerPos[1] > 0:
                        action = game.ACTION_SPACE[2] #left
                    elif playerLocation[0] - firstPlayerPos[0] < 0:
                        action = game.ACTION_SPACE[1] #down
                    elif playerLocation[0] - firstPlayerPos[0] > 0:
                        action = game.ACTION_SPACE[0] #up
                    else:
                        weights = [0.2, 0.2, 0.2, 0.2, 0.2, 0.0]
                        action = random.choices(game.ACTION_SPACE, weights=weights, k=1)[0]
                    break



            #weights = [0.198, 0.198, 0.198, 0.198, 0.198, 0.01]
            #action = random.choices(game.ACTION_SPACE, weights=weights, k=1)[0]
            game.pause_lock.acquire()
            game.pause_lock.release()
            if action == 'bomb':
                self.player.place_bomb()
            else:
                self.player.move(action)


    def start(self):
        if self._thread is None or not self._thread.is_alive():
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._run)
            threads_lock.acquire()
            threads[self] = self._thread
            threads_lock.release()
            self._thread.start()

    def stop(self):
        if self._thread is not None:
            self._stop_event.set()
            if self._thread.is_alive():
                self._thread.join()
            threads_lock.acquire()
            try:
                threads.pop(self)
            except KeyError:
               pass
            threads_lock.release()
        else:
            pass


class Environment:
    player: game.Player

    def __init__(self, player):
        self.player = player

    def reset(self):
        name = self.player.name
        while len(game.alive_players):
            try:
                p = game.alive_players[0]
                p.terminate()
            except IndexError:
                break


        while len(game.dead_players):
            p = game.dead_players[0]
            game.dead_players.remove(p)

        game.global_bomb_lock.acquire()
        for bomb in game.global_bombs:
            del bomb

        game.global_bomb_lock.release()
        game.grid_lock.acquire('game.get_start_grid()')
        game.grid = game.get_start_grid()
        game.grid_lock.release('game.get_start_grid()')
        timeout_ctr = 10
        while len(game.alive_players) > 0:
            time.sleep(0.01)
            assert timeout_ctr > 0, 'WTF is garbage not working?'

        while len(threads):
            pass  # fsr threads are not always cleared
            list(threads.keys())[0].stop()

    def step(self, action):
        r1 = self.player.score
        is_legal_move = True
        if action != 5:
            is_legal_move = self.player.move(game.ACTION_SPACE[action])
        else:
            y, x = self.player.get_position()
            for t in game.grid[y][x]:
                if type(t) is game.Bomb:
                    self.player.score -= ILLIGAL_PENALTY
            else:
                if self.player.bombs:
                    self.player.place_bomb()
                else:
                    self.player.score -= ILLIGAL_PENALTY
        r2 = self.player.score
        done = self.player.dead  # to je fertik
        observation_new = ''  # todo
        info = ''  # recimo
        reward = r2 - r1  # todo
        if is_legal_move:
            reward -= 1
        else:
            reward -= ILLIGAL_PENALTY
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
    def step(self, action):
        game.pause_lock.acquire()
        ILLIGAL_PENALTY = 50
        r1 = self.player.score
        is_legal_move = True
        if action != 5:
            is_legal_move = self.player.move(game.ACTION_SPACE[action])
        else:
            y, x = self.player.get_position()
            for t in game.grid[y][x]:
                if type(t) is game.Bomb:
                    self.player.score -= ILLIGAL_PENALTY*10
            else:
                if self.player.bombs:
                    self.player.place_bomb()
                else:
                    self.player.score -= ILLIGAL_PENALTY*10
        r2 = self.player.score
        done = self.player.dead  # to je fertik
        observation_new = ''  # todo
        info = ''  # recimo
        reward = r2 - r1  # todo
        if is_legal_move:
            reward -= ILLIGAL_PENALTY//2
        else:
            reward -= ILLIGAL_PENALTY
        game.pause_lock.release()
        return observation_new, reward, done, not is_legal_move

    def get_state(self):
        game.pause_lock.acquire()
        neigh8 = []
        x = self.player.get_position()[1]
        y = self.player.get_position()[0]
        game.grid_lock.acquire('get_state')
        for i in range(-1, 2, 1):
            for j in range(-1, 2, 1):
                if i == 0 and j == 0:
                    continue

                match game.grid[y-i][x-j][-1]:
                    case game.Wall():  # destroyable
                        neigh8.append(1)
                    case game.Bomb():
                        neigh8.append(2)
                    case game.Border:
                        neigh8.append(3)
                    case game.Player:
                        neigh8.append(4)
                    case _:
                        neigh8.append(0)
        game.grid_lock.release('get_state')
        dir_bomb_severity = [0, 0, 0, 0]  # up, right, down, left
        game.global_bomb_lock.acquire()
        for bomb in game.global_bombs:
            b_x = bomb.x
            b_y = bomb.y

            if not game.is_in_range(x,y, b_x, b_y, bomb.strength):
                continue
            if b_x == x:
                if b_y < y:
                    if bomb.get_time_left_ms() < 2 * self.player.speed:
                        dir_bomb_severity[0] = 1
                elif b_y > y:
                    if bomb.get_time_left_ms() < 2 * self.player.speed:
                        dir_bomb_severity[2] = 1
                else:
                    dir_bomb_severity[0] = 1
                    dir_bomb_severity[2] = 1
            if b_y == y:
                if b_x > x:
                    if bomb.get_time_left_ms() < 2000 * self.player.speed * game.TIME_CONST:
                        dir_bomb_severity[1] = 1
                elif b_x < x:
                    if bomb.get_time_left_ms() < 2000 * self.player.speed * game.TIME_CONST:
                        dir_bomb_severity[3] = 1
                else:
                    dir_bomb_severity[1] = 1
                    dir_bomb_severity[3] = 1

        for e in game.grid[y][x]:
            if type(e) is game.Bomb:
                dir_bomb_severity.append(1)
                break
        else:
            dir_bomb_severity.append(0)

        dir_bomb_severity.append(int(self.player.bombs > 0))
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
        game.pause_lock.release()
        return QEnv.encode_state(neigh8)

    @staticmethod
    def encode_state(state) -> int:
        a = 0
        for j in range(8):
            a += state[j]*5**(j)
        a *= 2**6*3**2

        b = 0
        for t in range(6):
            b += state[t+8]*2**(t)
        b *= 3**2

        c = 0
        for m in range(2):
            c += state[m+12]*2**m

        result = a + b + c
        return result

class DQEnv(Environment):
    def __init__(self, player):
        super().__init__(player)
