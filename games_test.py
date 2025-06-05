from other_games import eval_action_mask
from sb3_contrib import MaskablePPO

from pettingzoo.classic import connect_four_v3
from pettingzoo.classic import chess_v6
from pettingzoo.classic import gin_rummy_v4
from pettingzoo.classic import go_v5
from pettingzoo.classic import hanabi_v5
from pettingzoo.classic import leduc_holdem_v4
from pettingzoo.classic import texas_holdem_no_limit_v6
from pettingzoo.classic import texas_holdem_v4
from pettingzoo.classic import tictactoe_v3

model = MaskablePPO.load('connect_four_v3_20250605-190921.zip')

eval_action_mask(connect_four_v3, num_games=1, render_mode='human', is_random=False, is_deterministic=True)
