# Improving Sample Efficiency of Deep Reinforcement Learning for Bipedal Walking
The code to the article of the same title submitted to ICRA 2022. 

### Abstract
Reinforcement learning holds a great promise of
enabling bipedal walking in humanoid robots. However, despite
encouraging recent results, training still requires significant
amounts of time and resources, precluding fast iteration cycles
of the control development. Therefore, faster training methods
are needed. In this paper, we investigate a number of techniques for improving sample efficiency of on-policy actor-critic
algorithms and show that a significant reduction in training
time is achievable with a few straightforward modifications of
the common algorithms, such as PPO and DeepMimic, tailored
specifically towards the problem of bipedal walking. Action
space representation, symmetry prior induction, and cliprange
scheduling proved effective at reducing sample complexity by
a factor of 4.5. These results indicate that domain-specific
knowledge can be readily utilized to reduce training times and
thereby enable faster development cycles in challenging robotic
applications.

### Installation

1. Install _CUDA 10.1_ following [this medium post](https://medium.com/@exesse/cuda-10-1-installation-on-ubuntu-18-04-lts-d04f89287130). 
2. Follow [these instructions](https://phoenixnap.com/kb/how-to-install-anaconda-ubuntu-18-04-or-20-04) to install _anaconda_.
3. Create a conda environment from the .yml file located in the repository with
`conda env create -f path/to/conda_env_22dec20.yml`
4. Install _MuJoCo_ and _mujoco-py_ following [these instructions](https://github.com/openai/mujoco-py#install-mujoco).

### Main scripts, files and folders

- `scripts/config_light.py` specifies the simulation environment, as well as main hyperparameters and main experimental/training settings 
- `scripts/common/config.py` allows detailed control over all hyperparameters and experimental/training settings
- `scripts/train.py` trains a policy on the specified environment
- `scripts/run.py` loads a policy from a specified path and executes it on the environment defined in the config files. The execution can be rendered.
- `mujoco/gym_mimic_envs/mimic_env.py` implements the Base class to use a MuJoCo environment in the context of imitation learning
- `mujoco/gym_mimic_envs/mujoco/mimic_walker3d.py` is the main environment used during our experiments to train policies to generate stable human-like walking
   - `mujoco/gym_mimic_envs/mujoco/assets/walker3d_flat_feet.xml` defines the morphology and inertial properties of the walker  
- `scripts/mocap/ref_trajecs.py` loads the post-processed mocap data from `assets/ref_trajecs` and prepares it for usage with an RL environment.
- `graphs/` contains the processed monitoring data of different policies during training that were logged to Weights & Biases. 


### Supplementary videos

- Main video for the ICRA 2022 Submission is located in the `media` folder
- Videos of the walking gait recorded using different action spaces can be found in `media/videos_action_spaces` and in the [following Google Drive Folder](https://drive.google.com/drive/folders/1m-A7gxOcjN1_ZeDBMGs1AZ6Khp1Ebrjv?usp=sharing)

### Questions?

Please contact [Rustam Galljamov](mailto:rustam.galljamov@gmail.com) in case you have any questions regarding the code.
