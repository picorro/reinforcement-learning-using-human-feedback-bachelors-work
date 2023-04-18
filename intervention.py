import numpy as np


def get_intervention_step_array(steps, human_interuptions, exponent=4):
    return np.logspace(exponent, np.log10(steps), human_interuptions, dtype=int)
