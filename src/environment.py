import time
import game


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
        done = self.player.dead # to je fertik
        observation_new = '' # todo
        info = '' # recimo
        reward = r2-r1 # todo
        if moved == True:
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
              int[4dirBombCriticalityEnum]{NO_BOMB, LOTS_TIME(>2*speed), WILL_EXPLODE_JUST_NOW},
              int[2 signes dir (y, x)]{NEGATIVE(left , down), ON_POSITION, POSITIVE(right, up)}]
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
                    case game.Wall(): # destroyable
                        neigh8.append(1)
                    case game.Bomb():
                        neigh8.append(2)
                    case _:
                        neigh8.append(0)


        dir_bomb_severity = [0, 0, 0, 0] # up, right, down, left
        game.global_bomb_lock.acquire()
        for bomb in game.global_bombs:
            b_x = bomb.x
            b_y = bomb.y
            distance = game.get_distance(x,y, b_x, b_y)
            if distance > bomb.strength:
                continue
            if b_x == x:
                if b_y < y:
                    if bomb.get_time_left_ms() < 2* self.player.speed:
                        dir_bomb_severity[0] = 2 # vari vari dangarus
                    else:
                        dir_bomb_severity[0] = 1
                else:
                    if bomb.get_time_left_ms() < 2* self.player.speed:
                        dir_bomb_severity[2] = 2 # vari vari dangarus
                    elif dir_bomb_severity[2] != 2:
                        dir_bomb_severity[2] = 1

            if b_y == y:
                if b_x > x:
                    if bomb.get_time_left_ms() < 2* self.player.speed:
                        dir_bomb_severity[1] = 2 # vari vari dangarus
                    else:
                        dir_bomb_severity[1] = 1
                else:
                    if bomb.get_time_left_ms() < 2* self.player.speed:
                        dir_bomb_severity[3] = 2 # vari vari dangarus
                    elif dir_bomb_severity[3] != 2:
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
        neigh8.extend(dir_bomb_severity)
        neigh8.extend(direction)
        return QEnv.encode_state(neigh8)

    @staticmethod
    def encode_state(state) -> int:
        result = 0
        for i in range(14):
            result += state[i]*3**i

        return result

