import gymnasium as gym
import gym_duckietown

def launch_env(id=None):
    env = None
    if id is None:
        # Launch the environment
        from gym_duckietown.simulator import Simulator

        env = Simulator(
            seed=47,  # random seed
            map_name="loop_empty",
            max_steps=10000,  # we don't want the gym to reset itself
            domain_rand=False,
            camera_width=640,
            camera_height=480,
            accept_start_angle_deg=4,  # start close to straight
            full_transparency=True,
            distortion=True,
            randomize_maps_on_reset=False,
        )
    else:
        env = gym.make(id, render_mode="rgb_array")
    
    return env
