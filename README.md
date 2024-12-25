# Bipedal Walker v3 Project

The Bipedal Walker project trains a robot to walk using reinforcement learning. The environment provides continuous states, percepts (body state, joint states, and LIDAR readings), and actions (joint velocities). Rewards incentivize forward motion while penalizing falls.

---
## Project Overview
Multiple algorithms were trained and then evaluated for their efficiency (performance vs. training time). 

In hardcore mode, TRPO and PPO were further trained in a wrapped environment to successfully navigate, while TQC was trained to achieve a natural, balanced bipedal walk.

More information can be found in the "presentation.pdf" file.

---
## Results 
These are the results of training TQC for 5 million steps and PPO and TRPO for about 25 million steps:
|                    TQC                        |                   PPO                     |                  TRPO      |
|:---------------------------------------------:|:-------------------------------------------:|:---------------------------------------------:|
|  ![TQC ](media/TQChardcore.gif)                      | ![PPO](media/PPOhardcore.gif)                      | ![TRPO](media/TRPOhardcore.gif) | 

These are the results after implementing the Reward Wrappers:

|                    TQC Wrapped                        |                   PPO Wrapped                     |                  TRPO Wrapped                     |
|:---------------------------------------------:|:-------------------------------------------:|:---------------------------------------------:|
|  ![TQC ](media/TQCwrapped.gif)                      | ![PPO](media/PPOwrapped.gif)                      | ![TRPO](media/TRPOwrapped.gif) | 

---


## Requirements

We used WSL as the operating system and Python 3.11.0 as the coding language.

Install the requirements:
   ```bash
    pip install -r requirements.txt
   ```



---

## Authors
- Nuno Moreira
- Ricardo Ribeiro
- Rui Coelho