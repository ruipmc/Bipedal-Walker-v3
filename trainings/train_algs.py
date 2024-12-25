import gymnasium as gym
from stable_baselines3 import A2C, PPO, SAC, DDPG
from sb3_contrib import ARS, RecurrentPPO, TQC, TRPO
import os

# Directories for saving models and logs
models_dir = "models"
logs_dir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)

# Environment
env = gym.make("BipedalWalker-v3")

# Algorithms to train
algorithms = {
    "A2C": A2C("MlpPolicy", env, verbose=1, tensorboard_log=logs_dir),
    "ARS": ARS("MlpPolicy", env, verbose=1, tensorboard_log=logs_dir),
    "DDPG": DDPG("MlpPolicy", env, verbose=1, tensorboard_log=logs_dir),
    "PPO": PPO("MlpPolicy", env, verbose=1, tensorboard_log=logs_dir),
    "RecurrentPPO": RecurrentPPO("MlpPolicy", env, verbose=1, tensorboard_log=logs_dir),
    "SAC": SAC("MlpPolicy", env, verbose=1, tensorboard_log=logs_dir),
    "TQC": TQC("MlpPolicy", env, verbose=1, tensorboard_log=logs_dir),
    "TRPO": TRPO("MlpPolicy", env, verbose=1, tensorboard_log=logs_dir),
}

# Training parameters
TOTAL_TIMESTEPS = 1_000_000
TIMESTEPS_PER_ITERATION = 100_000

# Train each algorithm
for name, model in algorithms.items():
    print(f"Starting training for {name}")
    model_save_dir = os.path.join(models_dir, name)
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    for ep in range(1, TOTAL_TIMESTEPS // TIMESTEPS_PER_ITERATION + 1):
        model.learn(total_timesteps=TIMESTEPS_PER_ITERATION, reset_num_timesteps=False, tb_log_name=name)
        model.save(f"{model_save_dir}/{TIMESTEPS_PER_ITERATION * ep}")
        print(f"{name} - Completed {TIMESTEPS_PER_ITERATION * ep} timesteps")

env.close()
print("Training completed for all algorithms.")
