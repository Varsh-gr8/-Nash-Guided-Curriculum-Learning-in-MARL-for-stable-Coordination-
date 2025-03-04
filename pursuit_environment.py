import gymnasium as gym
import meld  # Import MELD environments
from stable_baselines3.common.env_util import make_vec_env

def make_pursuit_env():
    """Creates a Pursuit environment using MELD."""
    env = gym.make("meld-coordination-v0")  # Replace with a suitable MELD environment
    return env

def make_vec_pursuit_env(num_envs=4):
    """Creates a vectorized environment for multi-agent training."""
    return make_vec_env(make_pursuit_env, n_envs=num_envs)

if __name__ == "__main__":
    env = make_pursuit_env()
    obs, _ = env.reset()
    print("Observation Space:", env.observation_space)
    print("Action Space:", env.action_space)
