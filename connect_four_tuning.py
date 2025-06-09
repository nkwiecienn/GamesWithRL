import glob
import os
import time
import pandas as pd

from itertools import product
from matplotlib import pyplot as plt

import gymnasium as gym
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker

import pettingzoo.utils

from pettingzoo.classic import connect_four_v3
from pettingzoo.classic import chess_v6
from pettingzoo.classic import gin_rummy_v4
from pettingzoo.classic import go_v5
from pettingzoo.classic import hanabi_v5
from pettingzoo.classic import leduc_holdem_v4
from pettingzoo.classic import texas_holdem_no_limit_v6
from pettingzoo.classic import texas_holdem_v4
from pettingzoo.classic import tictactoe_v3


class SB3ActionMaskWrapper(pettingzoo.utils.BaseWrapper, gym.Env):

    def reset(self, seed=None, options=None):
        super().reset(seed, options)

        self.observation_space = super().observation_space(self.possible_agents[0])[
            "observation"
        ]
        self.action_space = super().action_space(self.possible_agents[0])

        return self.observe(self.agent_selection), {}

    def step(self, action):
        current_agent = self.agent_selection

        super().step(action)

        next_agent = self.agent_selection
        return (
            self.observe(next_agent),
            self._cumulative_rewards[current_agent],
            self.terminations[current_agent],
            self.truncations[current_agent],
            self.infos[current_agent],
        )

    def observe(self, agent):
        return super().observe(agent)["observation"]

    def action_mask(self):
        return super().observe(self.agent_selection)["action_mask"]


def get_latest_policy(env):
    try:
        latest_policy = max(
            glob.glob(f"{env.metadata['name']}*.zip"), key=os.path.getctime
        )
        return latest_policy
    except ValueError:
        print("Policy not found.")
        exit(0)

def mask_fn(env):
    return env.action_mask()


def train_action_mask(env, model: MaskablePPO, steps=10_000, seed=0, **env_kwargs):
    env.reset(seed=seed)

    print(f"Starting training on {env.unwrapped.metadata.get('name')}_{time.strftime('%Y%m%d-%H%M%S')}.")
    
    model.set_random_seed(seed)
    model.learn(total_timesteps=steps, progress_bar=True)

    print("Model has been saved.")

    print(f"Finished training on {str(env.unwrapped.metadata['name'])}.\n")
    # env.close()


def eval_action_mask(env, model: MaskablePPO, num_games=100, render_mode=None, is_random=True, is_deterministic=True, **env_kwargs):
    env = env_fn.env(render_mode=render_mode, **env_kwargs)

    print(
        f"Starting evaluation vs a random agent. Trained agent will play as {env.possible_agents[1]}."
    )

    scores = {agent: 0 for agent in env.possible_agents}
    total_rewards = {agent: 0 for agent in env.possible_agents}
    round_rewards = []

    for i in range(num_games):
        env.reset(seed=i)
        env.action_space(env.possible_agents[0]).seed(i)

        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()

            observation, action_mask = obs.values()

            if termination or truncation:
                if (
                    env.rewards[env.possible_agents[0]]
                    != env.rewards[env.possible_agents[1]]
                ):
                    winner = max(env.rewards, key=env.rewards.get)
                    scores[winner] += env.rewards[
                        winner
                    ]
                for a in env.possible_agents:
                    total_rewards[a] += env.rewards[a]
                round_rewards.append(env.rewards)
                break
            else:
                if agent == env.possible_agents[0]:
                    if is_random:
                        act = env.action_space(agent).sample(action_mask)
                    else:
                        act = int(
                            model.predict(
                                observation, action_masks=action_mask, deterministic=is_deterministic
                            )[0]
                        )
                else:
                    act = int(
                        model.predict(
                            observation, action_masks=action_mask, deterministic=is_deterministic
                        )[0]
                    )
            env.step(act)
    # env.close()

    if sum(scores.values()) == 0:
        winrate = 0
    else:
        winrate = scores[env.possible_agents[1]] / sum(scores.values())
    print("Rewards by round: ", round_rewards)
    print("Total rewards (incl. negative rewards): ", total_rewards)
    print("Winrate: ", winrate)
    print("Final scores: ", scores)
    return round_rewards, total_rewards, winrate, scores


if __name__ == "__main__":
    env_fn = tictactoe_v3
    raw_env = env_fn.env()
    env = SB3ActionMaskWrapper(raw_env)
    env.reset()
    env = ActionMasker(env, mask_fn)

    env_kwargs = {}

    params = {
        "learning_rate": [1e-4, 1e-5],
        "gae_lambda": [0.95, 0.8],
        "ent_coef": [0.0, 0.01]
    }

    param_strings = []
    winrates = []

    for learning_rate, gae_lambda, ent_coef, in product(*params.values()):
        param_string = ' '.join(map(str, [learning_rate, gae_lambda, ent_coef]))
        model = MaskablePPO(
            policy=MaskableActorCriticPolicy,
            env=env,
            verbose=1,
            tensorboard_log=f'tensorboard/{raw_env.metadata.get("name", "unknown_env")}/{param_string}',
            learning_rate=learning_rate,
            n_steps= 2048,
            gamma=0.9,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef= 0.5,
        )

        train_action_mask(env, model, steps=10*(2**13), seed=0, **env_kwargs)

        _, _, winrate, _ = eval_action_mask(env, model, num_games=500, render_mode=None, **env_kwargs)
        param_strings.append(param_string)
        winrates.append(winrate)

    df = pd.DataFrame({
        "params": param_strings,
        "winrate": winrates})
    
    df.sort_values(by="winrate", ascending=False, inplace=True)
    df.to_csv(f"results_{raw_env.metadata.get('name', 'unknown_env')}_{time.strftime('%Y%m%d-%H%M%S')}.csv", index=False)
    df.plot(x="params", y="winrate", kind="bar", title=f"Winrates", ylabel="Winrate")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()