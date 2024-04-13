import game


class Environment:
    action_space = ('up', 'down', 'left', 'right', 'noop', 'bomb')
    player : game.Player

    def __init__(self, player):
        self.player = player

    def reset(self):
        game.grid = game.get_start_grid()
        game.players_lock.acquire()
        name = self.player.name
        for p in game.alive_players:
            game.alive_players.remove(p)
            del p

        for p in game.dead_players:
            game.dead_players.remove(p)
            del p
        game.players_lock.release()
        del self.player
        self.player = game.Player(name)

    def step(self, action):
        r1 = self.player.score
        if action != 'bomb':
            self.player.move(action)
        else:
            self.player.place_bomb()
        r2 = self.player.score
        done = self.player.dead # to je fertik
        observation_new = '' # todo
        info = '' # recimo
        reward = r2-r1 # todo
        return observation_new, reward, done, info
