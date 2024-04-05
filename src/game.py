import threading
import sys
import time

TIME_CONST = 0.25  # factor for delaying the simulation, useful for debugging purposes
print_lock = threading.Lock()


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


class PowerBoost:
    strength: int

    def __init__(self, intensity=1):
        self.strength = intensity

    def __format__(self, format_spec):
        if format_spec == '2':
            return '\033[96m' + 'pB' + '\033[0m'
        else:
            return str(self)[:int(format_spec)]


class BombBoost:
    def __format__(self, format_spec):
        if format_spec == '2':
            return '\033[96m' + '+B' + '\033[0m'
        else:
            return str(self)[:int(format_spec)]


b = Border()
w = Wall()
e = Empty()
p = PowerBoost()
t = BombBoost()


grid = [[[b], [b], [b], [b], [b], [b], [b], [b], [b], [b]],
        [[b], [e], [t], [w], [w], [w], [w], [e], [e], [b]],
        [[b], [e], [w], [w], [p], [w], [w], [w], [e], [b]],
        [[b], [w], [w], [w], [w], [w], [w], [w], [w], [b]],
        [[b], [w], [w], [w], [w], [w], [w], [w], [w], [b]],
        [[b], [w], [w], [w], [w], [w], [w], [p], [w], [b]],
        [[b], [w], [w], [p], [w], [w], [w], [w], [w], [b]],
        [[b], [e], [w], [w], [w], [w], [w], [w], [e], [b]],
        [[b], [e], [e], [w], [w], [w], [w], [e], [e], [b]],
        [[b], [b], [b], [b], [b], [b], [b], [b], [b], [b]]]


class Bomb:
    owner = None
    strength: int
    x: int
    y: int
    timer: threading.Timer

    def __init__(self, timeout: float, strength: int, owner, x, y):
        self.owner = owner
        self.strength = strength
        self.x = x
        self.y = y
        if type(grid[y][x][0]) is Empty and self.owner.bombs:
            grid[y][x].append(self)
            self.owner.bombs -= 1
            self.timer = threading.Timer(timeout * TIME_CONST, self.explode_bomb)
            self.timer.start()
        else:
            pass

    def explode_bomb(self):
        y = self.y
        x = self.x
        grid[y][x].pop()
        for ra in [range(y, y + self.strength),
                   range(y, y - self.strength - 1, -1)]:
            for elem in ra:
                poi = grid[elem][x][-1]
                if type(poi) is Empty:
                    continue
                elif type(poi) is Wall:
                    grid[elem][x].pop()
                    grid[elem][x].append(e)
                    self.owner.score += 1
                elif type(poi) is Border:
                    break
                elif type(poi) is Player:
                    self.owner.score += 20
                    poi.terminate()
                elif type(poi) is Bomb:
                    poi.explode_bomb()
                    poi.timer.cancel()
        for ra in [range(x, x + self.strength),
                   range(x, x - self.strength - 1, -1)]:
            for elem in ra:
                poi = grid[y][elem][-1]
                if type(poi) is Empty:
                    continue
                elif type(poi) is Wall:
                    self.owner.score += 1
                    grid[y][elem].pop()
                    grid[y][elem].append(e)
                elif type(poi) is Border:
                    break
                elif type(poi) is Player:
                    self.owner.score += 20
                    poi.terminate()
                elif type(poi) is Bomb:
                    poi.explode_bomb()
                    poi.timer.cancel()

        self.owner.bomb_lock.acquire()
        self.owner.bombs += 1
        self.owner.bomb_lock.release()
        print_grid()

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
    global grid
    bomb_lock = threading.Lock()

    def __init__(self, name: str, player_id: int) -> None:
        self.name = name
        match player_id:
            case 0:
                self._position = [1, 1]
            case 1:
                self._position = [1, len(grid[0]) - 2]
            case 2:
                self._position = [len(grid) - 2, 1]
            case 3:
                self._position = [len(grid) - 2, len(grid) - 2]
        self.bomb_lock.acquire()
        self._max_bombs = 2
        self.bombs = self._max_bombs
        self.bomb_lock.release()
        grid[self._position[0]][self._position[1]].append(self)
        print_grid()

    """
    Terminate this player. Used by the bombs.
    """
    def terminate(self):
        self.dead = True
        grid[self._position[0]][self._position[1]].pop()

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
        return False

    def move(self, direction: str):
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
            case 'up':
                if self.process_loc(grid[y-1][x]):  # type(grid[y-1][x][-1]) in [Empty]:
                    self._position[0] -= 1
                    grid[y - 1][x].append(self)
            case 'down':
                if self.process_loc(grid[y + 1][x]):  # type(grid[y + 1][x][-1]) in [Empty]:
                    self._position[0] += 1
                    grid[y + 1][x].append(self)
            case 'left':
                if self.process_loc(grid[y][x-1]):  # type(grid[y][x-1][0]) in [Empty]:
                    self._position[1] -= 1
                    grid[y][x-1].append(self)
            case 'right':
                if self.process_loc(grid[y][x+1]):  # type(grid[y][x+1][0]) in [Empty]:
                    self._position[1] += 1
                    grid[y][x+1].append(self)

        time.sleep(0.5*TIME_CONST)
        print_grid()

    def place_bomb(self):
        y = self._position[0]
        x = self._position[1]
        bomb = Bomb(3, self._bomb_strength, owner=self, x=x, y=y)
        print_grid()

    def __str__(self):
        return f"name={self.name}, score={self.score}, bombs={self._max_bombs}, bombStrngth={self._bomb_strength}"

    def __format__(self, format_spec):
        if format_spec == '2':
            return '\033[91m' + self.name[:2] + '\033[0m'
        else:
            return str(self)[:int(format_spec)]


def print_grid():
    print_lock.acquire()
    move_cursor(0, 2)
    for row in grid:
        for item in row:
            print(f"{item[-1]:2}", end=" ")
        print()
    print()
    print_lock.release()
