import ast
import argparse
import logging

import os
import numpy as np

# Duckietown Specific
# from reinforcement.pytorch.ddpg import DDPG
from reinforcement.pytorch.utils import seed, evaluate_policy, ReplayBuffer
from utils.env import launch_env
from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import CnnPolicy, MlpPolicy

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


# def _train(args):
#     if not os.path.exists("./results"):
#         os.makedirs("./results")
#     if not os.path.exists(args.model_dir):
#         os.makedirs(args.model_dir)

#     # Launch the env with our helper function



#     total_timesteps = 0
#     timesteps_since_eval = 0
#     episode_num = 0
#     done = True
#     episode_reward = None
#     env_counter = 0
#     reward = 0
#     episode_timesteps = 0
#     print("Starting training")
#     while total_timesteps < args.max_timesteps:

#         print("timestep: {} | reward: {}".format(total_timesteps, reward))

#         if done:
#             if total_timesteps != 0:
#                 print(
#                     ("Total T: %d Episode Num: %d Episode T: %d Reward: %f")
#                     % (total_timesteps, episode_num, episode_timesteps, episode_reward)
#                 )
#                 policy.train(replay_buffer, episode_timesteps, args.batch_size, args.discount, args.tau)

#                 # Evaluate episode
#                 if timesteps_since_eval >= args.eval_freq:
#                     timesteps_since_eval %= args.eval_freq
#                     evaluations.append(evaluate_policy(env, policy))
#                     print("rewards at time {}: {}".format(total_timesteps, evaluations[-1]))

#                     if args.save_models:
#                         policy.save(file_name="ddpg", directory=args.model_dir)
#                     np.savez("./results/rewards.npz", evaluations)

#             # Reset environment
#             env_counter += 1
#             obs = env.reset()
#             done = False
#             episode_reward = 0
#             episode_num += 1
#             # episode_timesteps = 0

#         # Select action randomly or according to policy
#         if total_timesteps < args.start_timesteps:
#             action = env.action_space.sample()
#         else:
#             action = policy.predict(np.array(obs))
#             if args.expl_noise != 0:
#                 action = (action + np.random.normal(0, args.expl_noise, size=env.action_space.shape[0])).clip(
#                     env.action_space.low, env.action_space.high
#                 )

#         # Perform action
#         new_obs, reward, done, _ = env.step(action)
#         print("I AM TAKING AN ACTION")
#         env.render()
#         if episode_timesteps >= args.env_timesteps:
#             done = True

#         done_bool = 0 if episode_timesteps + 1 == args.env_timesteps else float(done)
#         episode_reward += reward

#         # Store data in replay buffer
#         replay_buffer.add(obs, new_obs, action, reward, done_bool)

#         obs = new_obs

#         episode_timesteps += 1
#         total_timesteps += 1
#         timesteps_since_eval += 1

#     print("Training done, about to save..")
#     policy.save(filename="ddpg", directory=args.model_dir)
#     print("Finished saving..should return now!")

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env
from utils.wrappers import NormalizeWrapper, ImgWrapper, DtRewardWrapper, ActionWrapper, ResizeWrapper

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # DDPG Args
    parser.add_argument("--seed", default=47, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument(
        "--start_timesteps", default=1e4, type=int
    )  # How many time steps purely random policy is run for
    parser.add_argument("--eval_freq", default=5e3, type=float)  # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e4, type=float)  # Max time steps to run environment for
    parser.add_argument("--save_models", action="store_true", default=True)  # Whether or not models are saved
    parser.add_argument("--expl_noise", default=0.1, type=float)  # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=32, type=int)  # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99, type=float)  # Discount factor
    parser.add_argument("--tau", default=0.005, type=float)  # Target network update rate
    parser.add_argument(
        "--policy_noise", default=0.2, type=float
    )  # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5, type=float)  # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)  # Frequency of delayed policy updates
    parser.add_argument("--env_timesteps", default=500, type=int)  # Frequency of delayed policy updates
    parser.add_argument(
        "--replay_buffer_max_size", default=10000, type=int
    )  # Maximum number of steps to keep in the replay buffer
    parser.add_argument("--model-dir", type=str, default="reinforcement/pytorch/models/")
    parser.add_argument("--model_file", type=str, default="ppo_duckie")
    # _train(parser.parse_args())
    args = parser.parse_args()

    env = launch_env()
    env = ResizeWrapper(env)
    env = NormalizeWrapper(env)
    env = ImgWrapper(env)  # to make the images from 160x120x3 into 3x160x120
    env = ActionWrapper(env)
    env = DtRewardWrapper(env)
    print("Initialized environment")

    # Wrappers
    check_env(env)

    print("Initialized Wrappers")
    # env.render()
    # Set seeds
    seed(args.seed)

    # state_dim = env.observation_space.shape
    # action_dim = env.action_space.shape[0]
    # max_action = float(env.action_space.high[0])

    # Initialize policy
    # model = PPO(MlpPolicy, env, verbose=2)
    max_reward = -1
    max_i = -1
    max_std = -1
    i = 6
    model = PPO.load(f"{args.model_dir}/{args.model_file}{i}")
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10,warn=True, render=True)
    print(f"mean_reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    if(max_i == -1 or max_reward < mean_reward):
      max_i = i
      max_reward = mean_reward
      max_std = std_reward
    print(f"MY BEST MODEL IS PPODUCK{max_i} mean_reward: {max_reward:.2f} +/- {max_std:.2f}")
    # obs, _ = env.reset()
    # while True:
    #     action, _states = model.predict(obs)
    #     obs, rewards, dones, info, misc = env.step(action)
    #     print(rewards)
    #     if(dones):
    #       obs, _ = env.reset()
    #     env.render()
    # replay_buffer = ReplayBuffer(args.replay_buffer_max_size)
    # print("Initialized PPO Stable Baseline")
    # # Evaluate untrained policy
    # print("ABOUT TO EVALUATE POLICY")
    # mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=1, render=True)
    # print(f"mean_reward: {mean_reward:.2f} +/- {std_reward:.2f}")
