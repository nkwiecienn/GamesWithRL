# Sources:
# https://flothesof.github.io/2048-game.html
# https://github.com/qwert12500/2048_rl

import numpy as np
from numpy import zeros, array, rot90
import random
import gymnasium as gym

class Board2048(gym.Env):
    def __init__(self):
        super(Board2048, self).__init__()
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(4, 4), dtype=np.float32)
        self.board = zeros((4, 4), dtype=int)
        self.fill_cell()
        self.game_over = False
        self.total_score = 0
        self.max_tile = 0

    def _get_obs(self):
        return (self.board / 2048.0).astype(np.float32)

    def reset(self, seed=None, options=None):
        self.__init__()
        return self._get_obs(), {}

    def fill_cell(self):
        i, j = (self.board == 0).nonzero()
        if i.size != 0:
            rnd = random.randint(0, i.size - 1)
            self.board[i[rnd], j[rnd]] = 2 * ((random.random() > .9) + 1)

    def move_left(self, col):
        new_col = zeros((4), dtype=col.dtype)
        j = 0
        previous = None
        for i in range(col.size):
            if col[i] != 0:
                if previous is None:
                    previous = col[i]
                else:
                    if previous == col[i]:
                        new_col[j] = 2 * col[i]
                        self.total_score += new_col[j]
                        self.max_tile = max(self.max_tile, new_col[j])
                        j += 1
                        previous = None
                    else:
                        new_col[j] = previous
                        j += 1
                        previous = col[i]
        if previous is not None:
            new_col[j] = previous
        return new_col

    def move(self, direction):
        rotated_board = rot90(self.board, direction)
        cols = [rotated_board[i, :] for i in range(4)]
        new_board = array([self.move_left(col) for col in cols])
        return rot90(new_board, -direction)

    def is_game_over(self):
        for i in range(self.board.shape[0]):
            for j in range(self.board.shape[1]):
                if self.board[i][j] == 0:
                    return False
                if i != 0 and self.board[i - 1][j] == self.board[i][j]:
                    return False
                if j != 0 and self.board[i][j - 1] == self.board[i][j]:
                    return False
        return True

    def snake_strategy(self):
        path = [
            (3, 0), (3, 1), (3, 2), (3, 3),
            (2, 3), (2, 2), (2, 1), (2, 0),
            (1, 0), (1, 1), (1, 2), (1, 3),
            (0, 3), (0, 2), (0, 1), (0, 0)
        ]
        values = [self.board[r][c] for r, c in path]
        score = 0
        for i in range(1, len(values)):
            if values[i - 1] >= values[i]:
                score += 1
        return score / (len(values) - 1)

    def step(self, direction):
        prev_score = self.total_score
        new_board = self.move(direction)

        # Domyślna nagroda = punkty za scalenie
        reward = self.total_score - prev_score

        # Kara za nielegalny ruch (brak zmian)
        if (new_board == self.board).all():
            reward -= 20
            return self._get_obs(), reward, self.is_game_over(), False, {}

        # Wykonaj ruch
        self.board = new_board
        self.fill_cell()

        # Premia za największy kafelek w rogu
        if self.board[3][0] == self.max_tile:
            reward += 50
        else:
            reward -= 50

        # Premia za zgodność z „wężem”
        reward += self.snake_strategy() * 50

        # Kara za przegraną
        if self.is_game_over():
            reward -= 200
            return self._get_obs(), reward, True, False, {}

        return self._get_obs(), reward, False, False, {}
