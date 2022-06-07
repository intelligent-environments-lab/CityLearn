from typing import Any, List, Union
import numpy as np

class Encoder:
    def __init__(self):
        r"""Initialize base `Encoder` class.

        Use to transform observation values in the replay buffer.
        """

        pass

    def __mul__(self, x: Any):
        raise NotImplementedError

    def __rmul__(self, x: Any):
        raise NotImplementedError

class NoNormalization(Encoder):
    def __init__(self):
        r"""Initialize `NoNormalization` encoder class.

        Use to return observation value as-is i.e. without any transformation.
        """

        super().__init__()

    def __mul__(self, x: Union[float, int]):
        return x
        
    def __rmul__(self, x: Union[float, int]):
        return x
        
class PeriodicNormalization(Encoder):
    def __init__(self, x_max: Union[float, int]):
        r"""Initialize `PeriodicNormalization` encoder class.

        Use to transform observations that are cyclical/periodic e.g. hour-of-day, day-of-week, e.t.c.

        Parameters
        ----------
        x_max : Union[float, int]
            Maximum observation value.

        Notes
        -----
        The transformation returns two values :math:`x_{sin}` and :math:`x_{sin}` defined as:
        
        .. math:: 
            x_{sin} = sin(\frac{2 \cdot \pi \cdot x}{x_{max}})
            
            x_{cos} = cos(\frac{2 \cdot \pi \cdot x}{x_{max}})

        Examples
        --------
        >>> x_max = 24
        >>> encoder = PeriodicNormalization(x_max)
        >>> encoder*2
        array([0.75, 0.9330127])
        """

        super().__init__()
        self.x_max = x_max

    def __mul__(self, x: Union[float, int]):
        x = 2 * np.pi * x / self.x_max
        x_sin = np.sin(x)
        x_cos = np.cos(x)
        return np.array([(x_sin+1)/2.0, (x_cos+1)/2.0])

    def __rmul__(self, x: Union[float, int]):
        x = 2 * np.pi * x / self.x_max
        x_sin = np.sin(x)
        x_cos = np.cos(x)
        return np.array([(x_sin+1)/2.0, (x_cos+1)/2.0])

class OnehotEncoding(Encoder):
    r"""Initialize `PeriodicNormalization` encoder class.

    Use to transform unordered categorical observations e.g. boolean daylight savings e.t.c.

    Parameters
    ----------
    classes : Union[List[float], List[int], List[str]]
        Observation categories.

    Examples
    --------
    >>> classes = [1, 2, 3, 4]
    >>> encoder = OnehotEncoding(classes)
    # identity_matrix = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    >>> encoder*2
    [0, 1, 0, 0]
    """

    def __init__(self, classes: Union[List[float], List[int], List[str]]):
        super().__init__()
        self.classes = classes

    def __mul__(self, x: Union[float, int, str]):
        identity_mat = np.eye(len(self.classes))
        return identity_mat[np.array(self.classes) == x][0]

    def __rmul__(self, x: Union[float, int, str]):
        identity_mat = np.eye(len(self.classes))
        return identity_mat[np.array(self.classes) == x][0]
    
class Normalize(Encoder):
    def __init__(self, x_min: Union[float, int], x_max: Union[float, int]):
        super().__init__()
        self.x_min = x_min
        self.x_max = x_max

    def __mul__(self, x: Union[float, int]):
        if self.x_min == self.x_max:
            return 0
        else:
            return (x - self.x_min)/(self.x_max - self.x_min)

    def __rmul__(self, x: Union[float, int]):
        if self.x_min == self.x_max:
            return 0
        else:
            return (x - self.x_min)/(self.x_max - self.x_min)
        
class RemoveFeature(Encoder):
    def __init__(self):
        super().__init__()
        pass

    def __mul__(self, x: Any):
        return None

    def __rmul__(self, x: Any):
        return None