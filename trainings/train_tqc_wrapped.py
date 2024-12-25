import gymnasium as gym
from sb3_contrib import TQC
from stable_baselines3.common.evaluation import evaluate_policy
import os

# Custom Reward Wrapper
class CustomBipedalRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.last_leg_position = None  # Track the previous leading leg
        self.last_vertical_velocity = 0.0  # Track vertical motion

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)

        # Extract body and joint data
        hull = self.env.unwrapped.hull
        joints = self.env.unwrapped.joints

        # Get velocities
        vertical_velocity = hull.linearVelocity[1]  # Y-axis velocity
        forward_velocity = hull.linearVelocity[0]  # X-axis velocity

        # Penalize excessive vertical motion (jumping)
        if abs(vertical_velocity) > 1.2:  # Allow small vertical movement
            reward -= 5.0

        # Encourage smooth movement (low vertical acceleration)
        vertical_acceleration = abs(vertical_velocity - self.last_vertical_velocity)
        reward -= vertical_acceleration * 2.0  # Penalize sharp changes in vertical velocity

        # Reward forward movement (scaled to avoid overshooting)
        reward += forward_velocity * 1.5  # Strong reward for forward motion

        # Penalize torso tilting
        torso_angle = hull.angle
        reward -= abs(torso_angle) * 0.5  # Penalize large tilts

        # Encourage alternating leg movement
        left_leg_angle = joints[0].angle
        right_leg_angle = joints[1].angle
        leading_leg = "left" if left_leg_angle > right_leg_angle else "right"

        if self.last_leg_position is not None:
            if self.last_leg_position != leading_leg:
                reward += 3.0  # Strong reward for alternating legs
            else:
                reward -= 1.0  # Penalize sticking with one leg

        # Penalize asymmetry in leg movement
        leg_angle_diff = abs(left_leg_angle - right_leg_angle)
        reward -= leg_angle_diff * 0.5  # Penalize imbalance between legs

        # Update tracked variables
        self.last_leg_position = leading_leg
        self.last_vertical_velocity = vertical_velocity

        return obs, reward, done, truncated, info


# Directories for saving models and logs
models_dir = "models/TQC_wrapped"
model_path = "models/TQC_wrapped/.zip"
logdir = "logs"

os.makedirs(models_dir, exist_ok=True)
os.makedirs(logdir, exist_ok=True)

env = gym.make("BipedalWalker-v3", hardcore=True)
env = CustomBipedalRewardWrapper(env)

# Initialize the model
# model = TRPO(
#     "MlpPolicy", 
#     env, 
#     verbose=1, 
#     tensorboard_log=logs_dir, 
# )

model = TQC.load(model_path, env, verbose =1, tensorboard_log=logdir, device= "cpu")

# Training loop
TIMESTEPS = 100000
for ep in range(1, 51):  # Train for 5M steps
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="TRPO_wrapped")
    model.save(f"{models_dir}/hardcore{TIMESTEPS * ep}")  # Save the model at each timestep

# Evaluate the trained model
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, render=False)
print(f"Mean reward: {mean_reward} ± {std_reward}")

env.close()
