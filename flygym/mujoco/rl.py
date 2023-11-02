import numpy as np
import os
from pathlib import Path
import logging
from tqdm import trange
from flygym.mujoco.arena import OdorsArena
import matplotlib.pyplot as plt
from flygym.mujoco import Parameters
from flygym.mujoco.examples.turning_controller import HybridTurningNMF

logging.basicConfig(level=logging.INFO)

odor_source = np.array([[24, 0, 1.5]])
peak_intensity = np.array([[1, 0]])
odor_valence = [10]


def make_arena():
    arena = OdorsArena(
        odor_source=odor_source,
        peak_intensity=peak_intensity,
        odor_valence=odor_valence,
        diffuse_func=lambda x: x**-2,
        marker_size=0.3,
    )
    return arena
