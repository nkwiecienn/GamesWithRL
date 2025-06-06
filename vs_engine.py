from sb3_contrib.common.maskable.utils import get_action_masks
import chess
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import gymnasium as gym
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from pettingzoo.classic.chess import chess_utils
from sb3_contrib.common.envs import InvalidActionEnvDiscrete


class SingleAgentChessEnv(gym.Env):
    """
    A single-agent chess environment using python-chess.
    The agent always plays as White, and Black moves are determined by a built-in engine or heuristic.
    """

    def __init__(self):
        super().__init__()
        self.board = chess.Board()

        self.action_space = spaces.Discrete(8 * 8 * 73)
        self.observation_space = spaces.Box(low=0, high=1, shape=(8, 8, 111), dtype=bool)

        self.board_history = np.zeros((8, 8, 104), dtype=bool)

    def get_obs(self):
        observation = chess_utils.get_observation(self.board, player=0)
        observation = np.dstack((observation[:, :, :7], self.board_history))

        legal_moves = chess_utils.legal_moves(self.board)
        action_mask = np.zeros(4672, dtype=np.uint8)
        for move in legal_moves:
            action_mask[move] = 1

        return {"observation": observation, "action_mask": action_mask}

    def reset(self, seed=None, options=None):
        self.board = chess.Board()
        self.board_history = np.zeros((8, 8, 104), dtype=bool)

        return self.observe(), {}


    def observe(self):
        return self.get_obs()["observation"]

    
    def action_masks(self):
        return self.get_obs()["action_mask"]

    def step(self, action):
        advantage_before = self._calculate_material_advantage()

        if self.is_game_over():
            return self.observe(), 0, True, False, {}

        move = chess_utils.action_to_move(self.board, action, player=0)

        assert move in self.board.legal_moves
        self.board.push(move)

        if self.is_game_over():
            reward = self._evaluate_result()
            return self.observe(), reward, True, False, {}

        # Engine (Black) move
        black_move = self._engine_move()
        self.board.push(black_move)

        if self.is_game_over():
            reward = self._evaluate_result()
            return self.observe(), reward, True, False, {}

        # Update board history
        # print(self.board.halfmove_clock)
        new_obs = chess_utils.get_observation(self.board, player=0)
        self.board_history = np.dstack((new_obs[:, :, 7:], self.board_history[:, :, :-13]))

        reward = (self._calculate_material_advantage() - advantage_before) / 100.0
        return self.observe(), reward, False, False, {}

    def is_game_over(self):
        return any([
            self.board.is_game_over(),
            self.board.is_checkmate(),
            self.board.is_stalemate(),
            self.board.is_insufficient_material(),
            self.board.can_claim_draw(),

        ])

    def _engine_move(self):
        """Use a basic heuristic to choose a legal move for Black."""
        legal_moves = list(self.board.legal_moves)
        return np.random.choice(legal_moves) if legal_moves else None

    def _evaluate_result(self):
        result = self.board.result(claim_draw=True)
        if result == '1-0':
            return 1
        elif result == '0-1':
            return -1
        else:
            return 0

    def close(self):
        pass

    def _calculate_material_advantage(self):
        """
        Calculate the material advantage for White.
        Positive values mean White is ahead, negative values mean Black is ahead.
        """
        piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 0
        }

        white_material = sum(
            piece_values[piece.piece_type] for piece in self.board.piece_map().values() if piece.color == chess.WHITE
        )
        black_material = sum(
            piece_values[piece.piece_type] for piece in self.board.piece_map().values() if piece.color == chess.BLACK
        )

        return white_material - black_material


def mask_fn(env: SingleAgentChessEnv):
    # Do whatever you'd like in this function to return the action mask
    # for the current env. In this example, we assume the env has a
    # helpful method we can rely on.
    return env.action_masks()


env_fn = SingleAgentChessEnv

env = env_fn()

env = ActionMasker(env, mask_fn)

model = MaskablePPO(
    policy=MaskableActorCriticPolicy,
    env=env,
    learning_rate=1e-5,
    n_steps=4096,
    batch_size=64,
    n_epochs=20,
    gamma=0.99,
    ent_coef=0.0,
    verbose=1,
    tensorboard_log='tensorboard/chess_single_agent'
)


model.set_random_seed(42)
model.learn(total_timesteps=200_000, progress_bar=True)

# Example gameplay
obs, _ = env.reset()
done = False

while not done:
    # Use the model to predict the next action
    action_mask = mask_fn(env)
    action, _ = model.predict(obs, deterministic=True, action_masks=action_mask)
    action = int(action)  # Ensure action is an integer
    
    # Take the action in the environment
    obs, reward, done, _, info = env.step(action)
    
    # Render the board state
    print(env.env.board)
    print('--------------------------------------------------')


    if done:
        print(f"Game Over! Reward: {reward}")