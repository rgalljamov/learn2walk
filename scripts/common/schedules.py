import numpy as np

class Schedule(object):
    def value(self, fraction_timesteps_left):
        """
        Value of the schedule for a given timestep

        :param fraction_timesteps_left:
            (float) PPO2 does not pass a step count in to the schedule functions
             but instead a number between 0 to 1.0 indicating how much timesteps are left
        :return: (float) the output value for the given timestep
        """
        raise NotImplementedError

class LinearSchedule(Schedule):
    def __init__(self, start_value, final_value):
        self.start = start_value
        self.end = final_value
        self.slope = final_value - start_value

    def value(self, fraction_timesteps_left):
        fraction_passed = 1 - fraction_timesteps_left
        val = self.start + fraction_passed * self.slope
        return val

    def __str__(self):
        return f'LinearSchedule: {self.start} -> {self.end}'

    def __repr__(self):
        return f'LinearSchedule: {self.start} -> {self.end}'