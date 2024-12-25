import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from gymnasium import RewardWrapper
from sb3_contrib import TRPO
import os

# Paths for models and logs
models_dir = "models/TRPO_wrapped"
logdir = "logs/TRPO_wrapped"
model_path = "models/TRPO_wrapped/.zip"

os.makedirs(models_dir, exist_ok=True)
os.makedirs(logdir, exist_ok=True)

class BalanceRewardWrapper(gym.RewardWrapper):
    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)

        # Add a reward for keeping balance (absolute forward velocity)
        forward_reward = abs(obs[2])
        reward += forward_reward

        return obs, reward, done, truncated, info

env = gym.make("BipedalWalker-v3", hardcore=True)
env = BalanceRewardWrapper(env)

# Initialize the model
# model = TRPO(
#     "MlpPolicy", 
#     env, 
#     verbose=1, 
#     tensorboard_log=logs_dir, 
# )

model = TRPO.load(model_path, env, verbose =1, tensorboard_log=logs_dir, device= "cpu")

# Training loop
TIMESTEPS = 1000000
for ep in range(1, 21):  # Train for 20M steps
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="TRPO_wrapped")
    model.save(f"{models_dir}/hardcore{TIMESTEPS * ep}")  # Save the model at each timestep

# Evaluate the trained model
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, render=False)
print(f"Mean reward: {mean_reward} ± {std_reward}")

env.close()
