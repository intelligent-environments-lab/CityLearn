import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from citylearn.base import EpisodeTracker
from citylearn.data import DeferrableApplianceSimulation
from citylearn.energy_model import DeferrableAppliance


def _simulation(
    *,
    profile=None,
    earliest=1,
    latest=2,
    deadline=None,
    priority=0.5,
    cycle_id="cycle_1",
):
    profile = [0.1, 0.2] if profile is None else profile
    deadline = latest + len(profile) - 1 if deadline is None else deadline
    return DeferrableApplianceSimulation.from_dataframes(
        cycle_profiles=pd.DataFrame(
            [
                {
                    "profile_id": "normal",
                    "duration_steps": len(profile),
                    "total_energy_kwh": sum(profile),
                    "load_profile": json.dumps(profile),
                }
            ]
        ),
        flexibility_schedule=pd.DataFrame(
            [
                {
                    "cycle_id": cycle_id,
                    "profile_id": "normal",
                    "earliest_start_time_step": earliest,
                    "latest_start_time_step": latest,
                    "deadline_time_step": deadline,
                    "priority": priority,
                    "must_run": True,
                }
            ]
        ),
    )


def _appliance(
    simulation=None,
    episode_time_steps=8,
    seconds_per_time_step=3600,
    trigger_threshold=None,
    nominal_power=2.0,
):
    tracker = EpisodeTracker(0, episode_time_steps - 1)
    tracker.next_episode(
        episode_time_steps,
        rolling_episode_split=False,
        random_episode_split=False,
        random_seed=0,
    )
    appliance = DeferrableAppliance(
        deferrable_appliance_simulation=simulation or _simulation(),
        name="washer_1",
        episode_tracker=tracker,
        seconds_per_time_step=seconds_per_time_step,
        nominal_power=nominal_power,
        **({} if trigger_threshold is None else {"trigger_threshold": trigger_threshold}),
    )
    appliance.reset()
    return appliance


def test_parser_validates_profile_reference_and_load_profile_safely():
    marker = Path(tempfile.gettempdir()) / "citylearn_deferrable_eval_should_not_run"
    marker.unlink(missing_ok=True)

    with pytest.raises(ValueError, match="unknown profile_id"):
        DeferrableApplianceSimulation.from_dataframes(
            cycle_profiles=pd.DataFrame(
                [
                    {
                        "profile_id": "normal",
                        "duration_steps": 1,
                        "total_energy_kwh": 0.1,
                        "load_profile": "[0.1]",
                    }
                ]
            ),
            flexibility_schedule=pd.DataFrame(
                [
                    {
                        "cycle_id": "cycle_1",
                        "profile_id": "missing",
                        "earliest_start_time_step": 0,
                        "latest_start_time_step": 0,
                        "deadline_time_step": 0,
                        "priority": 1.0,
                        "must_run": True,
                    }
                ]
            ),
        )

    with pytest.raises(ValueError, match="empty load_profile"):
        DeferrableApplianceSimulation.from_dataframes(
            cycle_profiles=pd.DataFrame(
                [
                    {
                        "profile_id": "unsafe",
                        "duration_steps": 1,
                        "total_energy_kwh": 0.0,
                        "load_profile": f"__import__('pathlib').Path({str(marker)!r}).write_text('x')",
                    }
                ]
            ),
            flexibility_schedule=pd.DataFrame(
                columns=[
                    "cycle_id",
                    "profile_id",
                    "earliest_start_time_step",
                    "latest_start_time_step",
                    "deadline_time_step",
                    "priority",
                    "must_run",
                ]
            ),
        )
    assert not marker.exists()


def test_valid_start_applies_cycle_energy_in_kwh_per_step_and_completes():
    appliance = _appliance(_simulation(profile=[0.1, 0.2], earliest=1, latest=2))

    appliance.start_cycle(1.0)
    assert appliance.cycle_state["cycle_1"] == "pending"
    np.testing.assert_allclose(appliance.electricity_consumption[:3], [0.0, 0.0, 0.0])

    appliance.next_time_step()
    appliance.start_cycle(1.0)

    assert appliance.cycle_state["cycle_1"] == "running"
    np.testing.assert_allclose(appliance.electricity_consumption[:4], [0.0, 0.1, 0.2, 0.0])
    assert appliance.observations()["current_step_energy_kwh"] == pytest.approx(0.1)

    appliance.next_time_step()
    assert appliance.observations()["remaining_energy_kwh"] == pytest.approx(0.2)
    appliance.next_time_step()
    assert appliance.cycle_state["cycle_1"] == "completed"

    summary = appliance.service_summary()
    assert summary["completed_cycles"] == 1.0
    assert summary["missed_cycles"] == 0.0
    assert summary["served_energy_kwh"] == pytest.approx(0.3)
    assert summary["service_level_ratio"] == 1.0


def test_default_trigger_threshold_is_binary_and_rejects_small_actions():
    appliance = _appliance(_simulation(profile=[0.2], earliest=0, latest=0), episode_time_steps=3)

    assert appliance.trigger_threshold == pytest.approx(0.5)
    assert appliance.observations()["can_start"] == 1.0

    appliance.start_cycle(0.5)
    assert appliance.cycle_state["cycle_1"] == "pending"
    np.testing.assert_allclose(appliance.electricity_consumption[:2], [0.0, 0.0])

    appliance.start_cycle(0.5001)
    assert appliance.cycle_state["cycle_1"] == "running"
    np.testing.assert_allclose(appliance.electricity_consumption[:2], [0.2, 0.0], atol=1e-9)


def test_cycle_cannot_start_if_it_would_finish_on_terminal_index():
    appliance = _appliance(_simulation(profile=[0.1, 0.2], earliest=0, latest=0), episode_time_steps=2)

    assert appliance.observations()["can_start"] == 0.0

    appliance.start_cycle(1.0)

    assert appliance.cycle_state["cycle_1"] == "pending"
    np.testing.assert_allclose(appliance.electricity_consumption, np.zeros(2))


def test_non_finite_start_action_is_treated_as_off():
    appliance = _appliance(_simulation(profile=[0.2], earliest=0, latest=0), episode_time_steps=3)

    appliance.start_cycle(np.nan)
    assert appliance.cycle_state["cycle_1"] == "pending"
    np.testing.assert_allclose(appliance.electricity_consumption[:2], [0.0, 0.0])


def test_start_before_window_or_too_late_is_rejected_and_missed():
    appliance = _appliance(_simulation(profile=[0.3], earliest=2, latest=2))

    appliance.start_cycle(1.0)
    assert appliance.cycle_state["cycle_1"] == "pending"
    np.testing.assert_allclose(appliance.electricity_consumption, np.zeros(8))

    appliance.next_time_step()
    appliance.next_time_step()
    appliance.next_time_step()
    appliance.start_cycle(1.0)

    assert appliance.cycle_state["cycle_1"] == "missed"
    assert appliance.service_summary()["unserved_energy_kwh"] == pytest.approx(0.3)
    assert appliance.observations()["deadline_missed"] == 1.0


def test_requests_expired_before_dynamic_activation_are_not_missed_service():
    appliance = _appliance(
        _simulation(profile=[0.3], earliest=1, latest=2, deadline=2),
        episode_time_steps=8,
    )

    appliance.skip_cycles_before(global_time_step=3)

    assert appliance.cycle_state["cycle_1"] == "expired_before_activation"
    assert appliance.service_summary()["completed_cycles"] == 0.0
    assert appliance.service_summary()["missed_cycles"] == 0.0
    assert appliance.service_summary()["unserved_energy_kwh"] == 0.0


def test_request_with_closed_start_window_is_excluded_even_before_deadline():
    appliance = _appliance(
        _simulation(profile=[0.1, 0.2], earliest=1, latest=2, deadline=4),
        episode_time_steps=8,
    )

    appliance.skip_cycles_before(global_time_step=3)

    assert appliance.cycle_state["cycle_1"] == "expired_before_activation"
    assert appliance.service_summary()["completed_cycles"] == 0.0
    assert appliance.service_summary()["missed_cycles"] == 0.0
    assert appliance.service_summary()["unserved_energy_kwh"] == 0.0


def test_15_second_cycle_keeps_small_kwh_values_exact():
    appliance = _appliance(
        _simulation(profile=[0.001, 0.002], earliest=0, latest=0),
        episode_time_steps=4,
        seconds_per_time_step=15,
    )

    appliance.start_cycle(1.0)

    np.testing.assert_allclose(appliance.electricity_consumption[:2], [0.001, 0.002], atol=1e-9)
    obs = appliance.observations()
    assert obs["cycle_energy_kwh"] == pytest.approx(0.003)
    assert obs["hours_until_deadline"] == pytest.approx(15 / 3600)


def test_15_second_profile_peak_cannot_exceed_nominal_power():
    with pytest.raises(ValueError, match="peak power"):
        _appliance(
            _simulation(profile=[0.1], earliest=0, latest=0),
            episode_time_steps=3,
            seconds_per_time_step=15,
            nominal_power=2.0,
        )
