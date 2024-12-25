import gymnasium as gym
from stable_baselines3 import PPO
from gymnasium import RewardWrapper
import os

# Paths for models and logs
models_dir = "models/PPO_wrapped"
logdir = "logs/PPo_wrapped"
model_path = f"{models_dir}/.zip"

os.makedirs(models_dir, exist_ok=True)
os.makedirs(logdir, exist_ok=True)

class BalanceRewardWrapper(gym.RewardWrapper):
    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)

        # Add a reward for keeping balance (absolute forward velocity)
        forward_reward = abs(obs[2])
        reward += forward_reward

        return obs, reward, done, truncated, info

# Create the single environment with the wrapper
env = gym.make("BipedalWalker-v3", hardcore=True)
env = BalanceRewardWrapper(env)

# # Load or initialize the model
# model = PPO(
#     "MlpPolicy",
#     env,
#     n_steps=2048,  # Larger buffer to capture longer-term dependencies
#     learning_rate=3e-4,  # Standard learning rate
#     clip_range=0.1,  # Smaller clip range for smoother updates
#     ent_coef=0.01,  # Encourage exploration
#     verbose=1,
#     tensorboard_log=logs_dir,
# )
model = PPO.load(
    model_path, 
    env,
    n_steps=2048,  # Larger buffer to capture longer-term dependencies
    learning_rate=3e-4,  # Standard learning rate
    clip_range=0.1,  # Smaller clip range for smoother updates
    ent_coef=0.01,  # Encourage exploration
    verbose=1,
    tensorboard_log=logdir,)


# Training loop<
TIMESTEPS = 1000000
for ep in range(1, 21):  # Train for 20M steps
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO_Wrapped5")
    model.save(f"{models_dir}/hardcore{TIMESTEPS * ep}")  # Save the model at each timestep

env.close()
