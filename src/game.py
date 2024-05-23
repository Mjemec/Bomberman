import threading
import sys
import time
from datetime import datetime
import numpy as np

ACTION_SPACE = ('up', 'down', 'left', 'right', 'noop', 'bomb')
TIME_CONST = 0.001
print_lock = threading.Lock()
players_lock = threading.Lock()
grid_lock = threading.Lock()
alive_players = list()
dead_players = list()

global_bomb_lock = threading.Lock()
global_bombs = set()
main_thread_id = threading.current_thread().ident

def is_in_range(x0, y0, x1, y1, rg) -> bool:
    if x0 == x1:
        return abs(y0-y1) <= rg
    elif y0 == y1:
        return abs(x0 - x1) <= rg
    else:
        return False


def move_cursor(x, y):
    sys.stdout.write(f"\033[{y};{x}H")


def get_distance(x0, y0, x1, y1):
    return abs(x0 - x1) + abs(y0 - y1)


class Border:
    def __format__(self, format_spec):
        if format_spec == '2':
            return '\033[94m' + 'XX' + '\033[0m'
        else:
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
            [[b], [e], [e], [e], [w], [w], [e], [e], [e], [b]],
            [[b], [e], [e], [w], [t], [w], [w], [e], [e], [b]],
            [[b], [e], [w], [w], [w], [s], [w], [w], [e], [b]],
            [[b], [w], [w], [w], [w], [w], [w], [w], [w], [b]],
            [[b], [w], [w], [s], [w], [w], [w], [t], [w], [b]],
            [[b], [e], [w], [t], [w], [w], [w], [s], [e], [b]],
            [[b], [e], [e], [w], [w], [w], [w], [e], [e], [b]],
            [[b], [e], [e], [e], [w], [w], [e], [e], [e], [b]],
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
    timeout = float('inf')

    def __init__(self, timeout: float, strength: int, owner, x, y):
        self.owner = owner
        self.strength = strength
        self.x = x
        self.y = y
        grid_lock.acquire()
        if type(grid[y][x][0]) is Empty and self.owner.bombs:
            self.owner.bombs -= 1
            self.timeout = timeout*TIME_CONST
            self.timer = threading.Timer(self.timeout, self.explode_bomb)
            global_bomb_lock.acquire()
            self.time_created = datetime.now()
            global_bombs.add(self)
            global_bomb_lock.release()

            grid[y][x].append(self)

            self.timer.start()
        else:
            pass
        grid_lock.release()

    def explode_bomb(self, caller=None):
        y = self.y
        x = self.x
        grid_lock.acquire()
        for l in range(len(grid[y][x])):
            if grid[y][x][l] == self:
                grid[y][x].remove(self)
                break
        grid_lock.release()

        for ra in [range(y, y + self.strength),
                   range(y, y - self.strength, -1)]:
            for elem in ra:
                grid_lock.acquire()
                poi = grid[elem][x][-1]
                grid_lock.release()
                match poi:
                    case Empty():
                        continue
                    case Wall():
                        grid_lock.acquire()
                        grid[elem][x].pop()
                        grid[elem][x].append(e)
                        grid_lock.release()
                        self.owner.score += 20 #10
                        break
                    case Border():
                        break
                    case Player():
                        if poi is not self.owner:
                            self.owner.score += 400 #100
                        try:
                            poi.terminate()
                        except:
                            print(poi.name)
                    case Bomb():
                        if poi is not caller:
                            try:
                                poi.explode_bomb(self)
                            except:
                                pass
                            poi.timer.cancel()
                    case _:
                        grid_lock.acquire()
                        grid[elem][x].pop()
                        grid[elem][x].append(e)
                        grid_lock.release()

        for ra in [range(x, x + self.strength),
                   range(x, x - self.strength, -1)]:
            for elem in ra:
                poi = grid[y][elem][-1]
                match poi:
                    case Empty():
                        continue
                    case Wall():
                        self.owner.score += 20 #10
                        grid_lock.acquire()
                        grid[y][elem].pop()
                        grid[y][elem].append(e)
                        grid_lock.release()
                        break
                    case Border():
                        break
                    case Player():
                        if self.owner is not poi:
                            self.owner.score += 400 #100
                        poi.terminate()
                    case Bomb():
                        if poi is not caller:
                            poi.explode_bomb(self)
                            poi.timer.cancel()
                    case _:
                        grid_lock.acquire()
                        grid[y][elem].pop()
                        grid[y][elem].append(e)
                        grid_lock.release()

        self.owner.bomb_lock.acquire()
        self.owner.bombs += 1
        self.owner.bomb_lock.release()
        global_bomb_lock.acquire()
        if self in global_bombs:
            global_bombs.remove(self)
        global_bomb_lock.release()
        print_grid()

    def get_time_left_ms(self):
        return self.timeout - (datetime.now() - self.time_created).total_seconds()*1000

    def __format__(self, format_spec):
        if format_spec == '2':
            return '\033[92m' + ':'*int(format_spec) + '\033[0m'
        else:
            return str(self)[:int(format_spec)]


class Player:
    _position = [0, 0]
    score: float = 0.0
    _max_bombs: int
    bombs: int
    _bomb_strength: int = 2
    name: str
    dead: bool = False
    speed = 0.75
    global grid
    bomb_lock = threading.Lock()

    def get_position(self):
        return self._position

    def get_bombs(self):
        return self.bombs

    def get_max_bombs(self):
        return self._max_bombs

    def get_score(self):
        return self.score

    def __init__(self, name: str, max_bombs=1) -> None:
        self.name = name

        grid_lock.acquire()
        match len(alive_players):
            case 0:
                self._position = [1, 1]
            case 1:
                self._position = [1, len(grid[0]) - 2]
            case 2:
                self._position = [len(grid) - 2, 1]
            case 3:
                self._position = [len(grid) - 2, len(grid) - 2]
            case _:
                assert False, f'Invalid player number{len(alive_players)}'
        grid_lock.release()
        players_lock.acquire()
        alive_players.append(self)
        players_lock.release()
        self.bomb_lock.acquire()
        self._max_bombs = max_bombs
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
        #print(f'player{self:2} terminated')
        players_lock.acquire()
        try:
            alive_players.remove(self)
            self.dead = True
            self.score -= 500 #300
            dead_players.append(self)
        except ValueError:
            pass
        players_lock.release()
        grid_lock.acquire()
        for element in grid[self._position[0]][self._position[1]]:
            if element is self:
                grid[self._position[0]][self._position[1]].remove(self)
        grid_lock.release()
    """
    :return True, if player can move to the new location
    """
    def process_loc(self, loc: list) -> bool:
        if self.dead:
            return False
        match loc[-1]:
            case Empty():
                return True
            case PowerBoost():
                self._bomb_strength += loc.pop().strength
                loc.append(e)
                return True
            case BombBoost():
                self._max_bombs += 1
                self.bombs += 1
                loc.pop()
                loc.append(e)
                return True
            case SpeedBoost():
                self.speed /= 1.5
                loc.pop()
                loc.append(e)
                return True
        return False

    def is_movable(self, loc: list) -> bool:
        if self.dead:
            return False
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
            return False
        y = self._position[0]
        x = self._position[1]
        grid_lock.acquire()
        try:
            for l in range(len(grid[y][x])):
                if grid[y][x][l] == self:
                    grid[y][x].remove(self)
                    break
            else:
                grid_lock.release()
                return False
        except:
            grid_lock.release()
            return False

        match direction:
            case 'noop':
                grid[y][x].append(self)
                self.score -= 5
                retval = True
            case 'up':
                if self.process_loc(grid[y-1][x]):
                    self._position[0] -= 1
                    grid[y - 1][x].append(self)
                    retval = True
                else: grid[y][x].append(self)
            case 'down':
                if self.process_loc(grid[y + 1][x]):
                    self._position[0] += 1
                    grid[y + 1][x].append(self)
                    retval = True
                else: grid[y][x].append(self)
            case 'left':
                if self.process_loc(grid[y][x-1]):
                    self._position[1] -= 1
                    grid[y][x-1].append(self)
                    retval = True
                else: grid[y][x].append(self)
            case 'right':
                if self.process_loc(grid[y][x+1]):
                    self._position[1] += 1
                    grid[y][x+1].append(self)
                    retval = True
                else: grid[y][x].append(self)
        grid_lock.release()
        time.sleep(self.speed*TIME_CONST)
        print_grid()
        return retval

    def place_bomb(self):
        y = self._position[0]
        x = self._position[1]
        bomb = Bomb(2, self._bomb_strength, owner=self, x=x, y=y)
        b_x = bomb.x
        b_y = bomb.y
        is_smart_by_player = False
        is_smart_by_wall = False

        for p in alive_players:
            if p is self:
                continue
            p_pos = p.get_position()
            p_x = p_pos[1]
            p_y = p_pos[0]
            if is_in_range(p_x, p_y, b_x, b_y, bomb.strength):
                is_smart_by_player = True
                break
        if is_smart_by_player:
            self.score += 200
        else:
            for neigh_block in [grid[b_y - 1][b_x], grid[b_y + 1][b_x],
                                grid[b_y][b_x - 1], grid[b_y][b_y + 1]]:
                if type(neigh_block[0]) is Wall:
                    is_smart_by_wall = True
                    break

            if is_smart_by_wall:
                self.score += 100

        if not is_smart_by_player and not is_smart_by_wall:
            self.score -= 50
        print_grid()
        
    def get_self_grid(self):
        grid_lock.acquire()

        bomb_grid = np.full((map_x_len, map_y_len), 1e6)
        blocks_grid = np.zeros(shape=(map_x_len, map_y_len), dtype=np.int8)
        players_grid = np.zeros(shape=(map_x_len, map_y_len), dtype=np.int8)
        power_up_grid = np.zeros((map_x_len, map_y_len), dtype=np.int8)

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
                            global_bomb_lock.acquire()
                            bomb_grid[y][x] = grid[y][x][k].get_time_left_ms()
                            global_bomb_lock.release()
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

        grid_lock.release()
        return players_grid, power_up_grid, blocks_grid, bomb_grid
    
    def get_self_entire_grid(self):
        grid_lock.acquire()

        environment_grid = np.zeros((map_x_len, map_y_len), dtype=np.int8)

        for y in range(map_y_len):
            for x in range(map_x_len):
                for k in range(len(grid[y][x])):
                    match(grid[y][x][k]):
                        case Empty():
                            continue
                        case Wall():
                            environment_grid[y][x] = 1
                        case Border():
                            environment_grid[y][x] = -1
                        case Bomb():
                            global_bomb_lock.acquire()
                            if grid[y][x][k].get_time_left_ms() <= 1 and environment_grid[y][x] != 100 and environment_grid[y][x] != 50:
                                environment_grid[y][x] = 10
                            elif(environment_grid[y][x] != 100 and environment_grid[y][x] != 50):
                                environment_grid[y][x] = 20
                            global_bomb_lock.release()
                        case Player():
                            if grid[y][x][k] == self:
                                environment_grid[y][x] = 100
                            else:
                                environment_grid[y][x] = 50
                        case PowerBoost():
                            environment_grid[y][x] = 2
                        case SpeedBoost():
                            environment_grid[y][x] = 4
                        case BombBoost():
                            environment_grid[y][x] = 3

        grid_lock.release()
        return environment_grid

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


HEADLESS = False
def print_grid():
    if HEADLESS or threading.current_thread().ident is not main_thread_id:
        return

    print_lock.acquire()
    move_cursor(0, 2)
    grid_lock.acquire()
    for row in grid:
        for item in row:
            try:

                print(f"{item[-1]:2}", end=" ")
            except IndexError:
                ...
        print()
    print()
    print_lock.release()
    grid_lock.release()


def print_final():
    return None
    print(f"The winner:\n \t {alive_players[0].name}, score={alive_players[0].score}")
    for p in dead_players:
        print(f"\t name={p.name}, score={p.score}")

