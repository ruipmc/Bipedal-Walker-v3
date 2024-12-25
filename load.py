import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import A2C, PPO, SAC, DDPG
from sb3_contrib import ARS, RecurrentPPO, TQC, TRPO

models_dir = "models/PPO"
model_path = f"{models_dir}/PPO.zip"

env = gym.make('BipedalWalker-v3', render_mode = "human", hardcore=True) 

model = PPO.load(model_path, env=env)

episodes = 5
for ep in range(episodes):
    obs, _ = env.reset()
    done = False
    while not done:
        env.render()
        action, _states = model.predict(obs, deterministic=False)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

env.close()