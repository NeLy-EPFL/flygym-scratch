import numpy as np


class FoodSource:
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
    marker_color : 4D np.ndarray
        The RGBA color of the source on the image frames.
    stock : int
        The number of times the fly can still visit the source before its stock runs out.
    stock_init : int
        The initial number of times a fly can visit each source before its stock runs out.

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
    marker_color : 4D np.ndarray, optional
        The RGBA color of the source on the image frames. By default it is [255, 0, 0, 1]
    stock : int, optional
        The number of times the fly can still visit the source before its stock runs out. By
        default it is 5.
    """

    def __init__(
        self,
        position: np.ndarray = np.array([10, 0, 0]),
        peak_intensity: np.ndarray = np.array([1]),
        odor_valence: float = 0.0,
        marker_color: np.ndarray = np.array([255, 0, 0, 1]),
        stock: int = 5,
    ):
        self.position = position
        self.peak_intensity = peak_intensity
        self.odor_valence = odor_valence
        self.marker_color = marker_color
        self.stock = stock
        self.stock_init = stock

    def move_source(self, new_pos=np.empty(0)) -> None:
        """
        This method is used to move the food source in the OdorArenaEnriched.
        If the fly has already visited the source more than a certain treshold (here 5)
        the food source is moved to a new position, representing a new
        source of the same food.

        Parameters
        new_pos : array
            The new position of the food source
        """

        if np.shape(new_pos) == (0,):
            x_pos = np.random.randint(0, 30, 1)[0]
            y_pos = np.random.randint(0, 23, 1)[0]
            new_pos = [x_pos, y_pos, 1.5]
            self.position = new_pos
        else:
            self.position = new_pos

    def consume(self) -> None:
        """
        Everytime the fly approaches a food source, the source's stock decreases
        (the fly eats the food decreasing its available quantity). If the food
        stock reaches zero, then the food source dissapears and a new one appears
        (the food source is restocked and its location is changed).
        """
        to_be_moved = False
        self.stock -= 1
        print(self.stock)
        if self.stock == 0:
            self.stock = self.stock_init
            self.move_source()
            to_be_moved = True
        return to_be_moved
