import numpy as np
from typing import Union, Iterable

class NoiseUtils:
    @staticmethod
    def generate_gaussian_noise(input_data: Union[np.ndarray, Iterable[float]], 
                              noise_std: float) -> np.ndarray:
        """Generates Gaussian noise matching input shape.
        
        Args:
            input_data: numpy array or iterable (list/tuple of numbers)
            noise_std: Noise standard deviation (ignored if <= 0)
            
        Returns:
            Zero-mean noise array with same shape as input
        """
        arr = np.asarray(input_data)  # Handles both ndarray and Iterable
        if noise_std <= 0:
            return np.zeros(arr.shape)
        return np.random.normal(loc=0, scale=noise_std, size=arr.shape)

    @staticmethod
    def generate_scaled_noise(input_data: Union[np.ndarray, Iterable[float]], 
                            noise_std: float, 
                            scale: float = 1.0) -> np.ndarray:
        """Generates pre-scaled noise (e.g., for percentage values)."""
        return NoiseUtils.generate_gaussian_noise(input_data, noise_std) * scale