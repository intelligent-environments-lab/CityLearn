import pytest

from citylearn.base import EpisodeTracker


def test_episode_time_steps_larger_than_simulation_window_raises_clear_error():
    tracker = EpisodeTracker(0, 9)

    with pytest.raises(ValueError, match='exceeds available simulation window'):
        tracker.next_episode(
            episode_time_steps=11,
            rolling_episode_split=False,
            random_episode_split=False,
            random_seed=0,
        )


def test_random_episode_split_with_single_split_is_handled():
    tracker = EpisodeTracker(0, 9)
    tracker.next_episode(
        episode_time_steps=10,
        rolling_episode_split=False,
        random_episode_split=True,
        random_seed=1,
    )

    assert tracker.episode_start_time_step == 0
    assert tracker.episode_end_time_step == 9


def test_non_positive_episode_time_steps_raises_value_error():
    tracker = EpisodeTracker(0, 9)

    with pytest.raises(ValueError, match='must be >= 1'):
        tracker.next_episode(
            episode_time_steps=0,
            rolling_episode_split=False,
            random_episode_split=False,
            random_seed=0,
        )
