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
        self.observation_space = gym.spaces.Box(low=0, high=2048, shape=(4, 4), dtype=np.int32)
        self.board = zeros((4, 4), dtype=int)
        self.fill_cell()
        self.game_over = False
        self.total_score = 0

    def reset(self, seed=None, options=None):
        self.__init__()
        return self.board, {}

    # Adding a random 2/4 into the board
    def fill_cell(self):
      i, j = (self.board == 0).nonzero()
      if i.size != 0:
          rnd = random.randint(0, i.size - 1)
          self.board[i[rnd], j[rnd]] = 2 * ((random.random() > .9) + 1)

    # Moving tiles in a column to left and merge if possible
    def move_left(self, col):
      new_col = zeros((4), dtype=col.dtype)
      j = 0
      previous = None
      for i in range(col.size):
          if col[i] != 0: # number different from zero
              if previous == None:
                  previous = col[i]
              else:
                  if previous == col[i]:
                      new_col[j] = 2 * col[i]
                      self.total_score += new_col[j]
                      j += 1
                      previous = None
                  else:
                      new_col[j] = previous
                      j += 1
                      previous = col[i]
      if previous != None:
          new_col[j] = previous
      return new_col

    def move(self, direction):
      # 0: left, 1: up, 2: right, 3: down
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


    def step(self, direction):
      prev_score = self.total_score
      new_board = self.move(direction)
      if self.is_game_over():
        reward = -100
        return self.board, reward, True, False, {}
      elif (new_board == self.board).all():
        reward = -10
      else:
        self.board = new_board
        self.fill_cell()
        reward = self.total_score - prev_score
        
      return self.board, reward, self.is_game_over(), False, {}