import numpy as np
import random
import logging
from tqdm import trange
from typing import Tuple, List, Optional, Callable, Dict
from dm_control import mjcf

from flygym.mujoco.util import load_config
from .food_sources import FoodSource
from .sensory_environment import OdorArena

logging.basicConfig(level=logging.INFO)


class OdorArenaEnriched(OdorArena):
    """Flat terrain with food sources.
    Attributes
    ----------
    root_element : mjcf.RootElement
        The root MJCF element of the arena.
    friction : Tuple[float, float, float]
        The sliding, torsional, and rolling friction coefficients of the
        ground, by default (1, 0.005, 0.0001)
    num_sensors : int
        The number of odor sensors, by default 4: 2 antennae + 2 maxillary
        palps.
    food_sources : np.ndarray
        An array containing all of the arena's food sources as FoodSource objects.
    num_odor_sources : int
        Number of odor sources.
    odor_dimensions : int
        Dimension of the odor space.
    diffuse_func : Callable
        The function that, given a distance from the odor source, returns
        the relative intensity of the odor. By default, this is a inverse
        square relationship.
    birdeye_cam : dm_control.mujoco.Camera
        MuJoCo camera that gives a birdeye view of the arena.
    birdeye_cam_zoom : dm_control.mujoco.Camera
         MuJoCo camera that gives a birdeye view of the arena, zoomed in
         toward the fly.
    valence_dictionary : dictionary
        Dictionary used to track the valence associated to each smell.
        For each smell, a value for the key of the dictionary is computed
        to which the valence of the smell is associated in the dictionary.
    """

    def __init__(
        self,
        size: Tuple[float, float] = (300, 300),
        friction: Tuple[float, float, float] = (1, 0.005, 0.0001),
        num_sensors: int = 4,
        food_sources: list = [
            FoodSource(np.array([[10, 0, 0]]), np.array([[1]]), np.array([[0]]))
        ],
        diffuse_func: Callable = lambda x: x**-2,
        marker_colors: Optional[List[Tuple[float, float, float, float]]] = None,
        marker_size: float = 0.25,
        key_angle: bool = False,
    ):
        """First initializer with list of food sources."""
        super().__init__(
            size,
            friction,
            num_sensors,
            np.array([source.position for source in food_sources]),
            np.array([source.peak_intensity for source in food_sources]),
            np.array([source.odor_valence for source in food_sources]),
            diffuse_func,
            marker_colors,
            marker_size,
            key_angle,
        )
        self.food_sources = food_sources

    def __init__(
        self,
        size: Tuple[float, float] = (300, 300),
        friction: Tuple[float, float, float] = (1, 0.005, 0.0001),
        num_sensors: int = 4,
        odor_source: np.ndarray = np.array([[10, 0, 0]]),
        peak_intensity: np.ndarray = np.array([[1]]),
        odor_valence: np.ndarray = np.array([[0]]),
        diffuse_func: Callable = lambda x: x**-2,
        marker_colors: Optional[List[Tuple[float, float, float, float]]] = None,
        marker_size: float = 0.25,
        key_angle: bool = False,
    ):
        """Second initializer with separate position, intensity and valence variable lists."""
        super().__init__(
            size,
            friction,
            num_sensors,
            odor_source,
            peak_intensity,
            odor_valence,
            diffuse_func,
            marker_colors,
            marker_size,
            key_angle,
        )
        self.food_sources = [
            FoodSource(position, intensity, valence)
            for position, intensity, valence in zip(
                odor_source, peak_intensity, odor_valence
            )
        ]
