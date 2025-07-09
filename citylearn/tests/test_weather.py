import unittest
import numpy as np

from citylearn.data import Weather

class TestWeatherStochasticity(unittest.TestCase):

    def setUp(self):
        self.length = 10
        self.base_data = {
            'outdoor_dry_bulb_temperature': [20.0]*self.length,
            'outdoor_relative_humidity': [50.0]*self.length,
            'diffuse_solar_irradiance': [200.0]*self.length,
            'direct_solar_irradiance': [800.0]*self.length,
            'outdoor_dry_bulb_temperature_predicted_1': [21.0]*self.length,
            'outdoor_dry_bulb_temperature_predicted_2': [22.0]*self.length,
            'outdoor_dry_bulb_temperature_predicted_3': [23.0]*self.length,
            'outdoor_relative_humidity_predicted_1': [55.0]*self.length,
            'outdoor_relative_humidity_predicted_2': [56.0]*self.length,
            'outdoor_relative_humidity_predicted_3': [57.0]*self.length,
            'diffuse_solar_irradiance_predicted_1': [210.0]*self.length,
            'diffuse_solar_irradiance_predicted_2': [220.0]*self.length,
            'diffuse_solar_irradiance_predicted_3': [230.0]*self.length,
            'direct_solar_irradiance_predicted_1': [810.0]*self.length,
            'direct_solar_irradiance_predicted_2': [820.0]*self.length,
            'direct_solar_irradiance_predicted_3': [830.0]*self.length,
        }

    def test_no_noise_returns_identical_values(self):
        weather = Weather(**self.base_data, noise_std=0.0)

        for key in self.base_data:
            original = np.array(self.base_data[key], dtype='float32')
            actual = getattr(weather, key)
            np.testing.assert_array_equal(original, actual, err_msg=f"{key} should be unchanged with noise_std=0.0")

    def test_noise_introduces_variation(self):
        weather = Weather(**self.base_data, noise_std=1.0)

        changes = []
        for key in self.base_data:
            original = np.array(self.base_data[key], dtype='float32')
            actual = getattr(weather, key)
            diff = np.abs(original - actual)
            changes.append(np.any(diff > 0.001))

        self.assertTrue(any(changes), "At least one array should differ when noise is applied.")

    def test_shapes_remain_consistent(self):
        weather = Weather(**self.base_data, noise_std=1.0)
        for key in self.base_data:
            expected_len = len(self.base_data[key])
            actual = getattr(weather, key)
            self.assertEqual(len(actual), expected_len, f"{key} should have same length as input.")

if __name__ == '__main__':
    unittest.main()
