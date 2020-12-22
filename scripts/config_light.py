# -----------------------------
# Experiment Specification
# -----------------------------



# -----------------------------
# Simulation Environment
# -----------------------------

# the registered gym environment id, e.g. 'Walker2d-v2'
ENV_ID = None
# simulation frequency... overwrite the frequency specified in the xml file
SIM_FREQ = 1000
# control frequency in Hz
CTRL_FREQ = 200
# does the model uses joint torques (True) or target angles (False)?
ENV_OUT_TORQUE = True
# peak joint torques [hip_sag, hip_front, knee_sag, ank_sag], same for both sides
PEAK_JOINT_TORQUES = [300, 300, 300, 300]



# -----------------------------
# Algorithm Hyperparameters
# -----------------------------
