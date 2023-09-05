import functools
import operator
import gym
import gym_duckietown
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def launch_env(id=None):
    env = None
    if id is None:
        # Launch the environment
        from gym_duckietown.simulator import Simulator

        env = Simulator(
            seed=47,  # random seed
            map_name="loop_empty",
            max_steps=500001,  # we don't want the gym to reset itself
            domain_rand=False,
            camera_width=640,
            camera_height=400,
            accept_start_angle_deg=4,  # start close to straight
            full_transparency=True,
            distortion=True,
        )
    else:
        env = gym.make(id)

    return env

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'device:{device}')
    # trace appears to be the map
    # I will use default duckiebot for now
    # sensor config in env.py
    # task config use map file
    env = launch_env()
    obs = env.reset()

if __name__ == '__main__':
    main()




