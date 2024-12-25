import gymnasium as gym
from stable_baselines3 import PPO
from sb3_contrib import TQC, TRPO
from gymnasium import RewardWrapper
from PIL import Image  # Para criar o GIF
import numpy as np
import time  # Para medir o tempo de execução

models_dir = "models/"

# Create the single environment with the wrapper
env = gym.make("BipedalWalker-v3", hardcore=True, render_mode="rgb_array")  # Use "rgb_array" render mode for frame capture

# Load the model
model_path = f"{models_dir}"
model = "model".load(model_path, env=env, device="cpu")

# Run the episodes and capture frames
episodes = 1
all_frames = []  # To store the frames
max_episode_duration = 25  # Maximum duration per episode in seconds

for ep in range(episodes):
    print(f"Starting episode {ep + 1}")
    obs, _ = env.reset()
    done = False
    episode_start_time = time.time()  # Record the start time of the episode
    episode_frames = []  # Collect frames for the current episode

    while not done:
        # Check elapsed time
        elapsed_time = time.time() - episode_start_time
        if elapsed_time > max_episode_duration:
            print(f"Episode {ep + 1} exceeded {max_episode_duration} seconds. Moving to the next episode.")
            break

        # Capture the current frame
        frame = env.render()  # This will give an RGB array (height, width, 3)
        frame = Image.fromarray(frame)  # Convert numpy array to PIL Image
        episode_frames.append(frame)  # Append to the episode frame list

        # Step through the environment
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    # Add frames from this episode to the main list if it has frames
    if episode_frames:
        all_frames.extend(episode_frames)

# Save the captured frames as a GIF
if all_frames:  # Ensure we have frames captured
    output_path = ".gif"
    all_frames[0].save(output_path, save_all=True, append_images=all_frames[1:], duration=25, loop=0)
    print(f"GIF saved to {output_path}")
else:
    print("No frames were captured. GIF not created.")

env.close()
