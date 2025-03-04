import gymnasium as gym
import meld
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from pursuit_environment import make_pursuit_env, make_vec_pursuit_env

# Define number of parallel environments
NUM_ENVS = 4

# Create the multi-agent environment
env = make_vec_pursuit_env(NUM_ENVS)

# Define Nash-Guided PPO model
model = PPO("MlpPolicy", env, verbose=1)

# Train the model
model.learn(total_timesteps=100000)

# Save the trained model
model.save("ppo_meld_pursuit")

# Test the trained model
obs = env.reset()
for _ in range(100):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
