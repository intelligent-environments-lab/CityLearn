from typing import Iterable, Union
import numpy as np

class Noise:
    @staticmethod
    def generate_gaussian_noise(input_data: Union[np.ndarray, Iterable[float]], noise_std: float) -> np.ndarray:
        """Generates Gaussian noise matching input shape.
        
        Parameters
        ----------
        input_data : Union[np.ndarray, Iterable[float]]
            Time series to add noise to.
        noise_std : float
            Noise standard deviation (ignored if <= 0)
            
        Returns
        -------
            noise: np.ndarray
                Zero-mean noise array with same shape as input
        """

        arr = np.asarray(input_data)  # Handles both ndarray and Iterable
        if noise_std <= 0:
            return np.zeros(arr.shape)
        return np.random.normal(loc=0, scale=noise_std, size=arr.shape)

    @staticmethod
    def generate_scaled_noise(input_data: Union[np.ndarray, Iterable[float]], noise_std: float, scale: float = 1.0) -> np.ndarray:
        """Generates pre-scaled noise (e.g., for percentage values)."""
        
        return Noise.generate_gaussian_noise(input_data, noise_std) * scale