import numpy as np

class FoodSource():
    """
    Class for defining food sources present in the arena.

    Attributes
    ----------
    position : 3D np.ndarray
        The position of the odor source in (x, y, z) coordinates.
    peak_intensity : list
        The peak intensity (or intensities) of the odor source. It can be multidimensional.
    odor_valence : float
        The valence that is associated to the smell of the source in a learning and 
        memory context.

    Parameters
    ----------
    position : 3D np.ndarray, optional
        The position of the odor source in (x, y, z) coordinates. By default it is [10, 0, 0].
    peak_intensity : np.ndarray, optional
        The peak intensity of the odor source, can be multidimensional. By default it is [1].
    odor_valence : float, optional
        The valence that is associated to the smell of the source in a learning and memory 
        context. Can be multidimensional if there are multiple different food types. 
        By default it is [0].
    """

    def __init__(
            self, 
            position : np.ndarray = np.array([10, 0, 0]),
            peak_intensity : np.ndarray = np.array([1]),
            odor_valence: float = 0.0
    ):
        self.position = position
        self.peak_intensity = peak_intensity
        self.odor_valence = odor_valence

    def get_odor_dimension(self):
        return self.peak_intensity.shape[0]
    
    def get_valence_dimension(self):
        return self.odor_valence.shape[0]

    