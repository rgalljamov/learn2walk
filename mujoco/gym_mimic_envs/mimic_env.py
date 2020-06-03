'''
Interface for environments using reference trajectories.
'''

class MimicEnv:
    def __init__(self, ref_trajecs):
        self.refs = ref_trajecs


    def get_qpos(self, timestep):
        raise NotImplementedError

