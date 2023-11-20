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
        odor_source: np.ndarray = None,
        peak_intensity: np.ndarray = None,
        odor_valence: np.ndarray = None,
        food_sources: list = None,
        diffuse_func: Callable = lambda x: x**-2,
        marker_colors: Optional[List[Tuple[float, float, float, float]]] = None,
        marker_size: float = 0.25,
        key_angle: bool = False,
    ):
        """
        Initializer allowing addition of food sources in two ways :
        - by giving a list of the food sources initialized through the FoodSource class
        - by giving the food sources' variables directly
        """
        if((food_sources is None) and (any([odor_source is None, peak_intensity is None, odor_valence is None]))):
            raise ValueError("OdorArenaEnriched has to be initialized with either food source list or individual arrays of the food sources' variables, cannot both be None.")
        if((food_sources is not None) and (all([odor_source is not None, peak_intensity is not None, odor_valence is not None]))):
            raise ValueError("OdorArenaEnriched has to be initialized with either food source list or individual arrays of the food sources' variables, not both.")
        
        if(food_sources is not None):
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
        else:
            super().__init__(size, friction, num_sensors, 
                        odor_source, 
                        peak_intensity,
                        odor_valence,
                        diffuse_func, marker_colors, marker_size, key_angle
                        )
            self.food_sources = [FoodSource(position, intensity, valence) for position, intensity, valence in zip(odor_source, peak_intensity, odor_valence)]

    def move_source(self, source_index, new_pos):
        self.food_sources[source_index].move_source(new_pos)

    def add_source(self, new_source):
        self.food_sources.append(new_source)

    def consume(self, source_index):
        self.food_sources[source_index].consume()