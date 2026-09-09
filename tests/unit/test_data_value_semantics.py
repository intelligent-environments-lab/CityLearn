import numpy as np

from citylearn.data import CarbonIntensity, Pricing


def test_pricing_preserves_negative_and_above_one_values():
    pricing = Pricing(
        electricity_pricing=[-0.25, 0.0, 2.5],
        electricity_pricing_predicted_1=[-0.2, 0.1, 2.0],
        electricity_pricing_predicted_2=[-0.1, 0.2, 1.5],
        electricity_pricing_predicted_3=[0.0, 0.3, 1.25],
        noise_std=0.0,
    )

    assert np.allclose(pricing.electricity_pricing, [-0.25, 0.0, 2.5])
    assert np.allclose(pricing.electricity_pricing_predicted_1, [-0.2, 0.1, 2.0])
    assert np.allclose(pricing.electricity_pricing_predicted_2, [-0.1, 0.2, 1.5])
    assert np.allclose(pricing.electricity_pricing_predicted_3, [0.0, 0.3, 1.25])


def test_carbon_intensity_preserves_raw_values_without_encoding():
    carbon = CarbonIntensity([0.2, 1.4, -0.1], noise_std=0.0)

    assert np.allclose(carbon.carbon_intensity, [0.2, 1.4, -0.1])
