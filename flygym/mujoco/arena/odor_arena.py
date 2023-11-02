import numpy as np
import random
import logging
from tqdm import trange
from typing import Tuple, List, Optional, Callable, Dict
from dm_control import mjcf

from flygym.mujoco.util import load_config
from .base import BaseArena

logging.basicConfig(level=logging.INFO)


class OdorsArena(BaseArena):
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
        Dictionary used to track the valence associated to each smell.
        For each smell, a value for the key of the dictionary is computed
        to which the valence of the smell is associated in the dictionary.

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
            texrepeat=(60, 60),
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
            fovy=90,
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
            num_odor_sources = self.odor_source.shape[0]
            intensity_norms = []
            norm_intensities = self.peak_odor_intensity

            # First normalize the peak intensities
            for i in range(num_odor_sources):
                intensity_norms.append(np.linalg.norm(norm_intensities[i]))
            # If any norms are over 100, normalize all intensities by the same factor
            max_norm = np.max(intensity_norms)
            if max_norm > 100:
                norm_intensities = norm_intensities / (max_norm / 100)

            # Then assign colors depending on the valence of the odor
            for i in range(num_odor_sources):
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
        self.marker_colors = marker_colors
        self._odor_marker_geoms = []

        for i, (pos, rgba) in enumerate(zip(self.odor_source, marker_colors)):
            marker_body = self.root_element.worldbody.add(
                "body", name=f"odor_source_marker_{i}", pos=pos, mocap=True
            )
            geom = marker_body.add(
                "geom", type="capsule", size=(marker_size, marker_size), rgba=rgba
            )
            self._odor_marker_geoms.append(geom)

        # Compute the key for each smell and update the dictionary
        for i in range(self.num_odor_sources):
            smell_key_value = self.compute_smell_key_value(self.peak_odor_intensity[i])
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

    def compute_smell_key_value(self, peak_intensity):
        """Method to compute the key used to store the smell into the valence_dictionary.
        The attractive component I1 of the smell is multiplied by 1,
        the aversive I2 by -1 and we choose to take max(|I1|, |-I2|)"""
        weights = np.array([[1, 0], [0, -1]])
        key_value_array = np.dot(peak_intensity, weights)
        key_value = key_value_array.flat[np.abs(key_value_array).argmax()]
        return key_value

    def pre_visual_render_hook(self, physics):
        for geom, rgba in zip(self._odor_marker_geoms, self.marker_colors):
            physics.bind(geom).rgba = np.array([*rgba[:3], 0])

    def post_visual_render_hook(self, physics):
        for geom, rgba in zip(self._odor_marker_geoms, self.marker_colors):
            physics.bind(geom).rgba = np.array([*rgba[:3], 1])
