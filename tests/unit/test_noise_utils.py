import numpy as np

from citylearn.utilities import NoiseUtils
from citylearn.utils.noise import Noise


def test_noise_helpers_return_float32_arrays():
    values = [1, 2, 3]

    assert NoiseUtils.generate_gaussian_noise(values, 0.0).dtype == np.float32
    assert NoiseUtils.generate_gaussian_noise(values, 0.1).dtype == np.float32
    assert Noise.generate_gaussian_noise(values, 0.0).dtype == np.float32
    assert Noise.generate_gaussian_noise(values, 0.1).dtype == np.float32
