import threading
import sys
import time
from datetime import datetime
import numpy as np

TIME_CONST = 0.25  # factor for delaying the simulation, useful for debugging purposes
print_lock = threading.Lock()
players_lock = threading.Lock()
grid_lock = threading.Lock()
alive_players = list()
dead_players = list()

bomb_lock = threading.Lock()
global_bombs = set()


def move_cursor(x, y):
    sys.stdout.write(f"\033[{y};{x}H")


class Border:
    def __format__(self, format_spec):
        if format_spec == '2':
            return '\033[94m' + 'XX' + '\033[0m'
        else:
            # Default behavior
            return str(self)[:int(format_spec)]


class Empty:
    def __format__(self, format_spec):
        if format_spec == '2':
            return '\033[92m' + '  ' + '\033[0m'
        else:
            return str(self)[:int(format_spec)]


class Wall:
    def __format__(self, format_spec):
        if format_spec == '2':
            return '\033[93m' + '[]' + '\033[0m'
        else:
            return str(self)[:int(format_spec)]


class PowerUp:
    pass


class PowerBoost(PowerUp):
    strength: int

    def __init__(self, intensity=1):
        self.strength = intensity

    def __format__(self, format_spec):
        if format_spec == '2':
            return '\033[96m' + 'pB' + '\033[0m'
        else:
            return str(self)[:int(format_spec)]


class BombBoost(PowerUp):
    def __format__(self, format_spec):
        if format_spec == '2':
            return '\033[96m' + '+B' + '\033[0m'
        else:
            return str(self)[:int(format_spec)]


class SpeedBoost(PowerUp):
    def __format__(self, format_spec):
        if format_spec == '2':
            return '\033[95m' + '+S' + '\033[0m'
        else:
            return str(self)[:int(format_spec)]

b = Border()
w = Wall()
e = Empty()
p = PowerBoost()
t = BombBoost()
s = SpeedBoost()

def get_start_grid():
    g_tem = [[[b], [b], [b], [b], [b], [b], [b], [b], [b], [b]],
            [[b], [e], [t], [w], [w], [w], [w], [e], [e], [b]],
            [[b], [e], [w], [w], [p], [w], [w], [w], [e], [b]],
            [[b], [w], [w], [w], [w], [s], [w], [w], [w], [b]],
            [[b], [w], [w], [w], [w], [w], [w], [w], [w], [b]],
            [[b], [w], [w], [s], [w], [w], [w], [p], [w], [b]],
            [[b], [w], [w], [p], [w], [w], [w], [s], [w], [b]],
            [[b], [e], [w], [w], [w], [w], [w], [w], [e], [b]],
            [[b], [e], [e], [w], [w], [w], [w], [e], [e], [b]],
            [[b], [b], [b], [b], [b], [b], [b], [b], [b], [b]]]
    return g_tem


grid = get_start_grid()
map_x_len = len(grid[0])
map_y_len = len(grid)


class Bomb:
    owner = None
    strength: int
    x: int
    y: int
    timer: threading.Timer
    time_created = None

    def __init__(self, timeout: float, strength: int, owner, x, y):
        self.owner = owner
        self.strength = strength
        self.x = x
        self.y = y
        if type(grid[y][x][0]) is Empty and self.owner.bombs:
            grid[y][x].append(self)
            self.owner.bombs -= 1
            self.timeout = timeout*TIME_CONST
            self.timer = threading.Timer(self.timeout, self.explode_bomb)
            self.timer.start()
        else:
            pass
        bomb_lock.acquire()
        global_bombs.add(self)
        bomb_lock.release()
        self.time_created = datetime.now()

    def explode_bomb(self):
        y = self.y
        x = self.x
        if isinstance(grid[y][x][-1], Bomb):
            grid[y][x].pop()
        for ra in [range(y, y + self.strength),
                   range(y, y - self.strength - 1, -1)]:
            for elem in ra:
                poi = grid[elem][x][-1]
                match poi:
                    case Empty():
                        continue
                    case Wall():
                        grid[elem][x].pop()
                        grid[elem][x].append(e)
                        self.owner.score += 5
                        break
                    case Border():
                        break
                    case Player():
                        self.owner.score += 50
                        poi.terminate()
                        players_lock.acquire()
                        alive_players.remove(poi)
                        dead_players.append(poi)
                        players_lock.release()
                    case Bomb():
                        poi.explode_bomb()
                        poi.timer.cancel()
                    case _:
                        grid[elem][x].pop()
                        grid[elem][x].append(e)

        for ra in [range(x, x + self.strength),
                   range(x, x - self.strength - 1, -1)]:
            for elem in ra:
                poi = grid[y][elem][-1]
                match poi:
                    case Empty():
                        continue
                    case Wall():
                        self.owner.score += 5
                        grid[y][elem].pop()
                        grid[y][elem].append(e)
                        break
                    case Border():
                        break
                    case Player():
                        self.owner.score += 50
                        poi.terminate()
                        players_lock.acquire()
                        alive_players.remove(poi)
                        dead_players.append(poi)
                        players_lock.release()
                    case Bomb():
                        poi.explode_bomb()
                        poi.timer.cancel()
                    case _:
                        grid[y][elem].pop()
                        grid[y][elem].append(e)

        self.owner.bomb_lock.acquire()
        self.owner.bombs += 1
        self.owner.bomb_lock.release()
        bomb_lock.acquire()
        global_bombs.remove(self)
        bomb_lock.release()
        print_grid()

        if len(alive_players) < 2:
            print_final()
            exit(0)

    def get_time_left_ms(self):
        return self.timeout - (self.time_created - datetime.now()).total_seconds()*1000


    def __format__(self, format_spec):
        if format_spec == '2':
            return '\033[92m' + ':'*int(format_spec) + '\033[0m'
        else:
            return str(self)[:int(format_spec)]


class Player:
    _position: [int, int]
    score: float = 0.0
    _max_bombs: int
    bombs: int
    _bomb_strength: int = 2
    name: str
    dead: bool = False
    speed = 0.5
    global grid
    bomb_lock = threading.Lock()

    # getters
    def get_position(self):
        return self._position

    def get_bombs(self):
        return self.bombs

    def get_max_bombs(self):
        return self._max_bombs

    def get_score(self):
        return self.score

    def __init__(self, name: str) -> None:
        self.name = name
        players_lock.acquire()
        match len(alive_players):
            case 0:
                self._position = [1, 1]
            case 1:
                self._position = [1, len(grid[0]) - 2]
            case 2:
                self._position = [len(grid) - 2, 1]
            case 3:
                self._position = [len(grid) - 2, len(grid) - 2]
        alive_players.append(self)
        players_lock.release()
        self.bomb_lock.acquire()
        self._max_bombs = 1
        self.bombs = self._max_bombs
        self.bomb_lock.release()
        grid_lock.acquire()
        grid[self._position[0]][self._position[1]].append(self)
        grid_lock.release()
        print_grid()


    """
    Terminate this player. Used by the bombs.
    """
    def terminate(self):
        self.dead = True
        self.score -= 100
        grid_lock.acquire()
        grid[self._position[0]][self._position[1]].pop()
        grid_lock.release()
    """
    :return True, if player can move to the new location
    """
    def process_loc(self, loc: list) -> bool:
        match loc[-1]:
            case Empty():
                return True
            case PowerBoost():
                print('power boost acquired')
                self._bomb_strength += loc.pop().strength
                loc.append(e)
                return True
            case BombBoost():
                print('bomb added')
                self._max_bombs += 1
                self.bombs += 1
                loc.pop()
                loc.append(e)
                return True
            case SpeedBoost():
                self.speed /= 1.5
                loc.pop()
                loc.append(e)
                print('speed increased')
                return True
        return False

    def is_movable(self, loc: list) -> bool:
        match loc[-1]:
            case Empty():
                return True
            case PowerBoost():
                return True
            case BombBoost():
                return True
            case SpeedBoost():
                return True
        return False

    """
    Move player in the direction. If unsuccessful, eg. move in the wall,
    returns False, otherwise True.
    """
    def move(self, direction: str) -> bool:
        retval = False
        if self.dead:
            return
        y = self._position[0]
        x = self._position[1]
        if type(grid[y][x][-1]) is Bomb:
            tmp = grid[y][x].pop()
            grid[y][x].pop()
            grid[y][x].append(tmp)
        else:
            grid[y][x].pop()

        match direction:
            case 'noop':
                grid[y][x].append(self)
                retval = True
            case 'up':
                if self.process_loc(grid[y-1][x]):  # type(grid[y-1][x][-1]) in [Empty]:
                    self._position[0] -= 1
                    grid[y - 1][x].append(self)
                    self.score += 1
                    retval = True
                else: grid[y][x].append(self)
            case 'down':
                if self.process_loc(grid[y + 1][x]):  # type(grid[y + 1][x][-1]) in [Empty]:
                    self._position[0] += 1
                    grid[y + 1][x].append(self)
                    self.score += 1
                    retval = True
                else: grid[y][x].append(self)
            case 'left':
                if self.process_loc(grid[y][x-1]):  # type(grid[y][x-1][0]) in [Empty]:
                    self._position[1] -= 1
                    grid[y][x-1].append(self)
                    self.score += 1
                    retval = True
                else: grid[y][x].append(self)
            case 'right':
                if self.process_loc(grid[y][x+1]):  # type(grid[y][x+1][0]) in [Empty]:
                    self._position[1] += 1
                    grid[y][x+1].append(self)
                    self.score += 1
                    retval = True
                else: grid[y][x].append(self)
        time.sleep(self.speed*TIME_CONST)
        print_grid()
        return retval

    def place_bomb(self):
        y = self._position[0]
        x = self._position[1]
        bomb = Bomb(3, self._bomb_strength, owner=self, x=x, y=y)
        print_grid()
        
    def get_self_grid(self):
        bomb_grid = np.full((map_x_len, map_y_len), 1e6)
        blocks_grid = np.zeros(shape=(map_x_len, map_y_len), dtype=np.uint8)
        players_grid = np.zeros(shape=(map_x_len, map_y_len), dtype=np.uint8)
        power_up_grid = np.zeros((map_x_len, map_y_len), dtype=np.uint8)

        for y in range(map_y_len):
            for x in range(map_x_len):
                for k in range(len(grid[y][x])):
                    match(grid[y][x][k]):
                        case Empty():
                            continue
                        case Wall():
                            blocks_grid[y][x] = 1
                        case Border():
                            blocks_grid[y][x] = -1
                        case Bomb():
                            bomb_grid[y][x] = grid[y][x][k].get_time_left_ms()
                        case Player():
                            if grid[y][x][k] == self:
                                players_grid[y][x] = 1
                            else:
                                players_grid[y][x] = -1
                        case PowerBoost():
                            power_up_grid[y][x] = 2
                        case SpeedBoost():
                            power_up_grid[y][x] = 1
                        case BombBoost():
                            power_up_grid[y][x] = 3

        return players_grid, power_up_grid, blocks_grid, bomb_grid

    def __str__(self):
        return f"name={self.name}, score={self.score}, bombs={self._max_bombs}, bombStrngth={self._bomb_strength}"

    def __format__(self, format_spec):
        if format_spec == '2':
            return '\033[91m\033[42m' + self.name[:2] + '\033[0m'
        else:
            return str(self)[:int(format_spec)]


def get_bombs():
    bomb_locations = list()
    for bomb in global_bombs:
        bomb_locations.append((bomb.y, bomb.y))
    return  bomb_locations


def print_grid():
    print_lock.acquire()
    move_cursor(0, 2)
    grid_lock.acquire()
    for row in grid:
        for item in row:
            print(f"{item[-1]:2}", end=" ")
        print()
    print()
    print_lock.release()
    grid_lock.release()


def print_final():
    return None
    print(f"The winner:\n \t {alive_players[0].name}, score={alive_players[0].score}")
    for p in dead_players:
        print(f"\t name={p.name}, score={p.score}")

