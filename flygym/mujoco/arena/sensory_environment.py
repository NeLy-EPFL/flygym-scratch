import numpy as np
import logging
from tqdm import trange
from typing import Tuple, List, Optional, Callable, Dict
from dm_control import mjcf
from gymnasium.core import ObsType

from flygym.mujoco.util import load_config
from .base import BaseArena

logging.basicConfig(level=logging.INFO)


class OdorArena(BaseArena):
    """Flat terrain with an odor source.

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
    odor_source : np.ndarray
        The position of the odor source in (x, y, z) coordinates. The shape
        of the array is (n_sources, 3).
    peak_intensity : np.ndarray
        The peak intensity of the odor source. The shape of the array is
        (n_sources, n_dimensions). Note that the odor intensity can be
        multidimensional.
    odor_valence : np.ndarray
        The valence that is associated to its smell in a learning and memory context.
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
        Dictionary used to track the valence associated to each odor source.
        For each source, a value for the key of the dictionary is computed
        to which its valence is associated.

    Parameters
    ----------
    size : Tuple[float, float], optional
        The size of the arena in mm, by default (300, 300).
    friction : Tuple[float, float, float], optional
        The sliding, torsional, and rolling friction coefficients of the
        ground, by default (1, 0.005, 0.0001).
    num_sensors : int, optional
        The number of odor sensors, by default 4: 2 antennae + 2 maxillary
        palps.
    odor_source : np.ndarray, optional
        The position of the odor source in (x, y, z) coordinates. The shape
        of the array is (n_sources, 3).
    peak_intensity : np.ndarray, optional
        The peak intensity of the odor source. The shape of the array is
        (n_sources, n_dimensions). Note that the odor intensity can be
        multidimensional.
    odor_valence : np.ndarray, optional
        The valence that is associated to its smell in a learning and memory context.
        The shape of the array is (n_sources, 1). Note that the odor valence can also be a float.
    diffuse_func : Callable, optional
        The function that, given a distance from the odor source, returns
        the relative intensity of the odor. By default, this is a inverse
        square relationship.
    marker_colors : List[Tuple[float, float, float, float]], optional
        A list of n_sources RGBA values (each as a tuple) indicating the
        colors of the markers indicating the positions of the odor sources.
        The RGBA values should be given in the range [0, 1]. By default,
        the matplotlib color cycle is used.
    marker_size : float, optional
        The size of the odor source markers, by default 0.25.
    key_angle : bool
        This booleans decides how to compute the key
        used in the valence dictionary for the different sources.
        If true, the key is computed considering the different odor sources
        (a smell is characterized by the angle its odor_peak intensity components form
        in the complex plane) else it is computed for the different food sources
    """

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
        self.root_element = mjcf.RootElement()
        ground_size = [*size, 1]
        chequered = self.root_element.asset.add(
            "texture",
            type="2d",
            builtin="checker",
            width=300,
            height=300,
            rgb1=(0.4, 0.4, 0.4),
            rgb2=(0.5, 0.5, 0.5),
        )
        grid = self.root_element.asset.add(
            "material",
            name="grid",
            texture=chequered,
            texrepeat=(30, 30),  # (60,60) for grid (300,300)
            reflectance=0.1,
        )
        self.root_element.worldbody.add(
            "geom",
            type="plane",
            name="ground",
            material=grid,
            size=ground_size,
            friction=friction,
        )
        self.friction = friction
        self.num_sensors = num_sensors
        self.odor_source = np.array(odor_source)
        self.peak_odor_intensity = np.array(peak_intensity)
        self.odor_valence = np.array(odor_valence)
        self.num_odor_sources = self.odor_source.shape[0]
        # Define the dictionary
        self.valence_dictionary = {}

        if self.odor_valence.shape[0] != self.odor_source.shape[0]:
            raise ValueError(
                "Number of valence values and peak intensities must match."
            )

        if self.odor_source.shape[0] != self.peak_odor_intensity.shape[0]:
            raise ValueError(
                "Number of odor source locations and peak intensities must match."
            )
        self.diffuse_func = diffuse_func

        # Add birdeye camera
        self.birdeye_cam = self.root_element.worldbody.add(
            "camera",
            name="birdeye_cam",
            mode="fixed",
            pos=(self.odor_source[:, 0].max() / 2, 0, 35),
            euler=(0, 0, 0),
            fovy=70,
        )
        self.birdeye_cam_zoom = self.root_element.worldbody.add(
            "camera",
            name="birdeye_cam_zoom",
            mode="fixed",
            pos=(11, 0, 29),
            euler=(0, 0, 0),
            fovy=45,
        )

        # Add markers at the odor sources
        # If no colors are given we will assign colors depending on the intensities of the peaks,
        # which we will use to change the color
        # Aversive : red, attractive : blue, neutral : green
        if marker_colors is None:
            marker_colors = []
            intensity_norms = []
            norm_intensities = self.peak_odor_intensity

            # First normalize the peak intensities
            for i in range(self.num_odor_sources):
                intensity_norms.append(np.linalg.norm(norm_intensities[i]))
            # If any norms are over 100, normalize all intensities by the same factor
            max_norm = np.max(intensity_norms)
            if max_norm > 100:
                norm_intensities = norm_intensities / (max_norm / 100)

            # Then assign colors depending on the valence of the odor
            for i in range(self.num_odor_sources):
                curr_intensity = norm_intensities[i]
                # white marker for (quasi-)nonexistant odor
                if curr_intensity[0] < 1e-2 and curr_intensity[1] < 1e-2:
                    marker_colors.append([255, 255, 255, 1])
                # red marker for purely attractive odor
                elif curr_intensity[0] > 1e-2 and curr_intensity[1] < 1e-2:
                    alpha = 0.1 + (0.9 * curr_intensity[0] / 100)
                    marker_colors.append([255, 0, 0, alpha])
                # blue marker for purely aversive odor
                elif curr_intensity[0] < 1e-2 and curr_intensity[1] > 1e-2:
                    alpha = 0.1 + (0.9 * curr_intensity[1] / 100)
                    marker_colors.append([0, 0, 255, alpha])
                # mixed odors have a color depending on the ratio between the attractive and aversive components
                else:
                    comp_attractive = curr_intensity[0]
                    comp_aversive = curr_intensity[1]
                    total = comp_attractive + comp_aversive
                    alpha = 0.1 + (0.9 * np.linalg.norm(curr_intensity) / 100)
                    if comp_attractive > comp_aversive:
                        ratio = comp_aversive / total
                        if ratio < 0.25:
                            marker_colors.append([0, int(ratio * 255), 255, alpha])
                        else:
                            marker_colors.append(
                                [0, 255, int((ratio - 0.25) * 255), alpha]
                            )
                    else:
                        ratio = comp_attractive / total
                        if ratio < 0.25:
                            marker_colors.append([255, int(ratio * 255), 0, alpha])
                        else:
                            marker_colors.append(
                                [int((ratio - 0.25) * 255), 255, 0, alpha]
                            )

        for i, (pos, rgba) in enumerate(zip(self.odor_source, marker_colors)):
            marker_body = self.root_element.worldbody.add(
                "body", name=f"odor_source_marker_{i}", pos=pos, mocap=True
            )
            marker_body.add(
                "geom", type="capsule", size=(marker_size, marker_size), rgba=rgba
            )

        if key_angle:
            # Compute the angle key for each odor and update the internal valence dictionary
            # (two food source on the arena can have the same odor)
            for i in range(self.num_odor_sources):
                smell_key_value = self.compute_smell_angle_value(
                    self.peak_odor_intensity[i]
                )
                self.valence_dictionary.update({smell_key_value: self.odor_valence[i]})
        else:
            # Compute the key for each food source and update the internal valence dictionary
            for i in range(self.num_odor_sources):
                smell_key_value = self.compute_smell_key_value(
                    self.peak_odor_intensity[i]
                )
                self.valence_dictionary.update({smell_key_value: self.odor_valence[i]})

        # Reshape odor source and peak intensity arrays to simplify future claculations
        _odor_source_repeated = self.odor_source[:, np.newaxis, np.newaxis, :]
        _odor_source_repeated = np.repeat(
            _odor_source_repeated, self.odor_dimensions, axis=1
        )
        _odor_source_repeated = np.repeat(
            _odor_source_repeated, self.num_sensors, axis=2
        )
        self._odor_source_repeated = _odor_source_repeated
        _peak_intensity_repeated = self.peak_odor_intensity[:, :, np.newaxis]
        _peak_intensity_repeated = np.repeat(
            _peak_intensity_repeated, self.num_sensors, axis=2
        )
        self._peak_intensity_repeated = _peak_intensity_repeated

    def get_spawn_position(
        self, rel_pos: np.ndarray, rel_angle: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        return rel_pos, rel_angle

    def get_olfaction(self, antennae_pos: np.ndarray) -> np.ndarray:
        """
        Notes
        -----
        w = 4: number of sensors (2x antennae + 2x max. palps)
        3: spatial dimensionality
        k: data dimensionality
        n: number of odor sources

        Input - odor source position: [n, 3]
        Input - sensor positions: [w, 3]
        Input - peak intensity: [n, k]
        Input - difusion function: f(dist)

        Reshape sources to S = [n, k*, w*, 3] (* means repeated)
        Reshape sensor position to A = [n*, k*, w, 3] (* means repeated)
        Subtract, getting an Delta = [n, k, w, 3] array of rel difference
        Calculate Euclidean disctance: D = [n, k, w]

        Apply pre-integrated difusion function: S = f(D) -> [n, k, w]
        Reshape peak intensities to P = [n, k, w*]
        Apply scaling: I = P * S -> [n, k, w] element wise

        Output - Sum over the first axis: [k, w]
        """
        antennae_pos_repeated = antennae_pos[np.newaxis, np.newaxis, :, :]
        dist_3d = antennae_pos_repeated - self._odor_source_repeated  # (n, k, w, 3)
        dist_euc = np.linalg.norm(dist_3d, axis=3)  # (n, k, w)
        scaling = self.diffuse_func(dist_euc)  # (n, k, w)
        intensity = self._peak_intensity_repeated * scaling  # (n, k, w)
        return intensity.sum(axis=0)  # (k, w)

    @property
    def odor_dimensions(self) -> int:
        return self.peak_odor_intensity.shape[1]

    def compute_smell_key_value(self, peak_intensity) -> float:
        """
        Method to compute the key used to store the food sources into the valence_dictionary.
        The attractive component I1 of the smell is multiplied by 1,
        the aversive I2 by -1 and we choose to take max(|I1|, |-I2|) as key.

        Input - peak intensity: [n, k]
        Output - float: the key value
        """
        weights = np.array([[1, 0], [0, -1]])
        key_value_array = np.dot(peak_intensity, weights)
        key_value = key_value_array.flat[np.abs(key_value_array).argmax()]
        return key_value

    def normalize_peak_intensity(self, peak_intensity) -> np.array:
        """
        This function is used to normalize the different peak intensities.

        Input - peak intensity: [n, k]
        Output - normalized peak intensity: [n, k]
        """
        return peak_intensity / np.linalg.norm(peak_intensity)

    def compute_smell_angle_value(self, peak_intensity) -> float:
        """
        Method to compute the key for the different peak intensity
        by choosing as key the angle in the complex plane
        generated by the [x,y] components of peak_odor_intensity of
        the smell source.

        Input - peak intensity: [n, k]
        Output - the key value for the odor source
        """
        normalized_peak_intensity = self.normalize_peak_intensity(peak_intensity)
        angle = np.arctan(normalized_peak_intensity[0] / normalized_peak_intensity[1])
        return angle

    def generate_random_gains(self, explore=True) -> Tuple:
        """
        Method to compute random numbers of opposite signed assigned
        for the attractive/aversive gain for basic odor taxis.
        The range of gains is [0, 500] and [0,300]. The gain with the highest
        absolute value has negative sign. The fly will be walking towards
        the source whose gain has the highest absolute value.


        Parameters
        ----------
        explore : bool
            Whether the fly is free to explore or it is guided to a specific source, by default True

        Returns
        -------
        attractive_gain, aversive_gain : tuple
            The gains needed to guide the fly around the arena
        """

        attractive_gain = 0
        aversive_gain = 0

        x = np.random.randint(500)
        y = np.random.randint(500)
        if explore:
            if x > y:
                attractive_gain = -x
                aversive_gain = 0
            else:
                attractive_gain = 0
                aversive_gain = -y
        else:
            # the fly will fo to the source with highest reward
            max_key = max(self.valence_dictionary, key=self.valence_dictionary.get)
            if max_key > 0:
                attractive_gain = 0
                aversive_gain = -max(x, y)
            else:
                attractive_gain = -max(x, y)
                aversive_gain = 0
        return attractive_gain, aversive_gain

    def generate_random_gains_food(
        self, internal_state="satiated", fly_pos=np.array([0, 0, 0])
    ) -> Tuple:
        """
        Method to compute random gains for the odor sources.
        The range of gains is [0, 500].
        The fly will have different behaviors depending on its internal state.
        If it is satiated, it will explore.
        If it is hungry, it will exploit and
        go to the source with the highest reward.
        If it is starving, it will go to the closest source.

        Parameters
        ----------
        internal_state : str
            The internal (food) state of the fly
        fly_pos : np.array
            The current position of the fly, by default [0,0,0]

        Returns
        -------
        attractive_gain, aversive_gain : tuple
            The gains needed to guide the fly around the arena

        """
        x = np.random.randint(500)
        y = np.random.randint(500)

        assert internal_state in ["satiated", "hungry", "starving"]

        attractive_gain = 0
        aversive_gain = 0

        match internal_state:
            # If the fly is satiated it can explore
            case "satiated":
                highest = np.argmax([x, y])
                if highest == 0:
                    attractive_gain = -max(x, y)
                    aversive_gain = 0
                else:
                    attractive_gain = 0
                    aversive_gain = -max(x, y)
            # If fly is hungry it will exploit and go to food source with highest reward
            case "hungry":
                max_key = max(self.valence_dictionary, key=self.valence_dictionary.get)
                if max_key > 0:
                    attractive_gain = 0
                    aversive_gain = -max(x, y)
                else:
                    attractive_gain = -max(x, y)
                    aversive_gain = 0
            # If fly is starving it will go to closest food source
            case "starving":
                distance = np.inf
                index_source = 0
                # Find the closest source
                for i in len(self.odor_source):
                    tmp_distance = np.linalg.norm(fly_pos - self.odor_source[i, :2])
                    if tmp_distance < distance:
                        distance = tmp_distance
                        index_source = i
                # If food source is attractive
                if (
                    self.peak_odor_intensity[index_source][0]
                    > self.peak_odor_intensity[index_source][1]
                ):
                    attractive_gain = -max(x, y)
                    aversive_gain = 0
                # Else it's aversive
                else:
                    attractive_gain = 0
                    aversive_gain = -max(x, y)

        return attractive_gain, aversive_gain

    def generate_random_gains_food_internal_state(
        self,
        internal_state="satiated",
        mating_state="virgin",
    ) -> Tuple:
        """
        Method to compute random gains for the odor sources.
        The range of gains is [0, 500].
        The fly will have different behaviors depending on its internal state.
        If it is virgin and satiated, it will prefer exploring towards the sucrose source.
        If virgin and hungry, it will go to the source of yeast with the highest reward.
        If virgin and starving, it will go to the closest source of yeast.
        Else, if it is mated and satiated, it will prefer exploring towards the yeast source.
        If mated and hungry it will go to the yeast source with the highest reward.
        If mated and starving, it will go to the closest source of yeast.

        Yeast sources: the [x,y] peak_odor_intensity components are such that x>y; otherwise,
        it is a sucrose source.

        Parameters
        ----------
        internal_state : str
            The internal (food, AAs stock level) state of the fly
        mating_state : str
            The mating state of the fly

        Returns
        -------
        attractive_gain, aversive_gain : tuple
            The gains needed to guide the fly around the arena

        """
        x = np.random.randint(500)
        y = np.random.randint(500)

        assert internal_state in ["satiated", "hungry", "starving"]

        attractive_gain = 0
        aversive_gain = 0

        match internal_state:
            # If the fly is satiated it can explore
            case "satiated":
                if mating_state == "virgin":
                    # Look for sucrose sources
                    attractive_gain = 0
                    aversive_gain = -max(x, y)
                else:
                    # Look for yeast sources
                    attractive_gain = -max(x, y)
                    aversive_gain = 0
            # If fly is hungry it will exploit and go to yeast source with highest reward
            case "hungry":
                # Look for yeast sources
                attractive_gain = -max(x, y)
                aversive_gain = 0
            # If fly is starving it will go to closest food source
            case "starving":
                # Look for yeast sources
                attractive_gain = -max(x, y)
                aversive_gain = 0

        return attractive_gain, aversive_gain

    def generate_curiosity_rdm_gains_food_internal_state(self) -> Tuple:
        """
        Method to compute random gains for the odor sources.
        The range of gains is [0, 500].
        In this case, all sources are considered attractive, we therefore
        generate a negative attractive gains with a big absolute value,
        the aversive gain is set to 0 to let the fly walks faster to the
        chosen source. This could be set to min(x,y) if we would like
        the fly to explore around the arena a bit more.

        Returns
        -------
        attractive_gain, aversive_gain : tuple
            The gains needed to guide the fly around the arena
        """
        x = np.random.randint(500)
        y = np.random.randint(500)

        attractive_gain = -max(x, y)
        aversive_gain = 0  # min(x,y)

        return attractive_gain, aversive_gain


    def is_yeast(self, source_index) -> bool:
        """
        This function returns whether the food
        source is yeast or sucrose

        Parameters:
        ----------
        source_index : float
            The source that needs to be checked

        Returns:
        -------
        bool value : whether the food source is yeast or sucrose
        """
        if (
            self.peak_odor_intensity[source_index][0]
            > self.peak_odor_intensity[source_index][1]
        ):
            return True
        else:
            return False

    def compute_richest_yeast_source(self) -> float:
        """
        This function find the reachest source of yeast
        (the one that has the highest reward)

        Returns:
        -------
        float : the index of the reachest source of yeast
        """
        arena_valence_dict = self.valence_dictionary
        found_key = True
        while found_key:
            max_key = max(arena_valence_dict, key=arena_valence_dict.get)
            if max_key < 0:
                arena_valence_dict.pop(max_key)
            else:
                found_key = False
        for el in range(len(self.peak_odor_intensity)):
            if max_key == self.compute_smell_key_value(self.peak_odor_intensity[el]):
                return el

    def compute_richest_source(self) -> float:
        """
        This function find the reachest source
        (the one that has the highest reward)

        Returns:
        -------
        float : the index of the reachest source
        """
        arena_valence_dict = self.valence_dictionary
        max_key = max(arena_valence_dict, key=arena_valence_dict.get)
        for el in range(len(self.peak_odor_intensity)):
            if max_key == self.compute_smell_key_value(self.peak_odor_intensity[el]):
                return el

    def get_specific_olfaction(self, index_source, sim) -> np.array:
        """
        This function is needed when the fly wants
        to reach a specific source.
        The odor_obs are computed as if in the arena
        there was just the source the fly wants to reach

        Parameters
        ----------
        index_source : float
            The index of the source that fly needs to reach
        sim : Mujoco Hybrid Turning Controller
            The fly that needs to reach the source

        Output - Sum over the first axis: [k, w]
        """
        odors = self.odor_source[index_source]
        odors = np.expand_dims(odors, axis=0)
        _odor_source_repeated = odors[:, np.newaxis, np.newaxis, :]
        _odor_source_repeated = np.repeat(
            _odor_source_repeated, self.odor_dimensions, axis=1
        )
        _odor_source_repeated = np.repeat(
            _odor_source_repeated, self.num_sensors, axis=2
        )

        peak_odor_intesity = self.peak_odor_intensity[index_source]
        peak_odor_intesity = np.expand_dims(peak_odor_intesity, axis=0)
        _peak_intensity_repeated = peak_odor_intesity[:, :, np.newaxis]
        _peak_intensity_repeated = np.repeat(
            _peak_intensity_repeated, self.num_sensors, axis=2
        )
        _peak_intensity_repeated = _peak_intensity_repeated
        antennae_pos = sim.physics.bind(sim._antennae_sensors).sensordata
        antennae_pos = antennae_pos.reshape(4, 3)
        antennae_pos_repeated = antennae_pos[np.newaxis, np.newaxis, :, :]
        dist_3d = antennae_pos_repeated - _odor_source_repeated  # (n, k, w, 3)
        dist_euc = np.linalg.norm(dist_3d, axis=3)  # (n, k, w)
        scaling = self.diffuse_func(dist_euc)  # (n, k, w)
        intensity = _peak_intensity_repeated * scaling  # (n, k, w)
        return intensity.sum(axis=0)  # (k, w)

    def generate_exploration_turning_control(
        self, attractive_gain, aversive_gain, obs
    ) -> np.array:
        """
        This functions is used to computer
        the control signal used to make the fly walk
        around the arena.
        The fly is here exploring freely both sucrose/yeast
        or attractive/aversive sources given the value of the gains.

        Parameters:
        ---------
        attractive_gain : float
            The gain for the attractive sources
        aversive_gain : float
            The gain for the aversive sources
        obs : ObsType
            The observation as defined by the environment.

        Returns:
        -------
        control_signal : np.array
            The signal needed to make the fly walk
        """
        # Compute bias from odor intensity
        attractive_intensities = np.average(
            obs["odor_intensity"][0, :].reshape(2, 2), axis=0, weights=[9, 1]
        )
        aversive_intensities = np.average(
            obs["odor_intensity"][1, :].reshape(2, 2), axis=0, weights=[10, 0]
        )
        attractive_bias = (
            attractive_gain
            * (attractive_intensities[0] - attractive_intensities[1])
            / attractive_intensities.mean()
        )
        aversive_bias = (
            aversive_gain
            * (aversive_intensities[0] - aversive_intensities[1])
            / aversive_intensities.mean()
        )
        effective_bias = aversive_bias + attractive_bias
        effective_bias_norm = np.tanh(effective_bias**2) * np.sign(effective_bias)
        assert np.sign(effective_bias_norm) == np.sign(effective_bias)

        # Compute control signal
        control_signal = np.ones((2,))
        side_to_modulate = int(effective_bias_norm > 0)
        modulation_amount = np.abs(effective_bias_norm) * 0.8
        control_signal[side_to_modulate] -= modulation_amount
        return control_signal

    def generate_specific_turning_control(
        self, index_source, sim, attractive_gain=-500
    ) -> np.array:
        """
        This functions is used to compute
        the control signal used to make the fly walk
        around the arena.
        It computed the bias from the odor intensity knowing that
        the fly needs to reach a source specified
        by the index_source. Here all sources are by default considered attractive.
        Parameters:
        ---------

        index_source : float
            The index of the source that needs to be reached
        sim : Mujoco Hybrid Turning Controller
            The fly that will explore the arena.
        attractive_gain : float
            The gain for the attractive sources, by deafult -500

        Returns:
        -------
        control_signal : np.array
            The signal needed to make the fly walk
        """

        obs = self.get_specific_olfaction(index_source, sim)

        attractive_intensities = np.average(
            obs[0, :].reshape(2, 2), axis=0, weights=[9, 1]
        )
        attractive_bias = (
            attractive_gain
            * (attractive_intensities[0] - attractive_intensities[1])
            / attractive_intensities.mean()
        )
        aversive_bias = 0
        effective_bias = aversive_bias + attractive_bias
        effective_bias_norm = np.tanh(effective_bias**2) * np.sign(effective_bias)
        assert np.sign(effective_bias_norm) == np.sign(effective_bias)

        # Compute control signal
        control_signal = np.ones((2,))
        side_to_modulate = int(effective_bias_norm > 0)
        modulation_amount = np.abs(effective_bias_norm) * 0.8
        control_signal[side_to_modulate] -= modulation_amount
        return control_signal

    def get_odor_intensities(self) -> np.array:
        """
        This function returns the odor
        intensity of the arena

        Outputs: np.array - the peak_odor_intensity vector
        """
        return self.peak_odor_intensity

    def compute_richest_closest_source(self, obs, food_source=False) -> float:
        """
        This method is used to compute the closest source to the fly that has the highest reward.
        A reward can be shared between different sources of the same smell.
        Therefore with this function the fly computes the highest possible reward
        and then looks for the closest source with this reward.

        Parameters
        ----------
        obs: ObsType
            The observation as defined by the environment.
        food_source : bool
            Whether the arena is a OdorArenaEnriched or not.

        Returns
        -------
        index_source : float
            the index of the richest closest source

        """
        max_key = max(self.valence_dictionary, key=self.valence_dictionary.get)
        possible_sources = []
        for el in range(len(self.peak_odor_intensity)):
            if max_key == self.compute_smell_angle_value(self.peak_odor_intensity[el]):
                possible_sources.append(el)
        distance = np.inf
        index_source = 0
        if food_source:
            for i in possible_sources:
                tmp_distance = np.linalg.norm(
                    obs["fly"][0, :2] - self.food_sources[i].position[:2]
                )
                if tmp_distance < distance:
                    distance = tmp_distance
                    index_source = i
        else:
            for i in possible_sources:
                tmp_distance = np.linalg.norm(
                    obs["fly"][0, :2] - self.odor_source[i, :2]
                )
                if tmp_distance < distance:
                    distance = tmp_distance
                    index_source = i
        return index_source
