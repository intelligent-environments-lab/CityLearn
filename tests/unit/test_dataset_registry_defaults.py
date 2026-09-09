from citylearn import __version__
from citylearn.data import DataSet


def test_dataset_registry_defaults_to_matching_official_release():
    dataset = DataSet()

    assert dataset.github_account == "citylearn-project"
    assert dataset.repository == "CityLearn"
    assert dataset.tag == f"v{__version__}"
