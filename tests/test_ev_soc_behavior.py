import os
import sys
import unittest

PARENT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)

from pathlib import Path

import numpy as np

from citylearn.citylearn import CityLearnEnv


SCHEMA = Path(PARENT) / 'data' / 'datasets' / 'citylearn_challenge_2022_phase_all_plus_evs' / 'schema.json'


class TestEVBatterySOC(unittest.TestCase):
    """Regression tests for EV SOC management behaviour."""

    def setUp(self):
        self.env = CityLearnEnv(
            str(SCHEMA),
            central_agent=True,
            episode_time_steps=96,
            random_seed=0,
        )
        self.env.reset()

    def tearDown(self):
        self.env = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _zero_actions(self):
        """Return a zero-action list matching the environment action space."""
        return [
            np.zeros(space.shape, dtype='float32')
            for space in self.env.action_space
        ]

    def _ev_action_index(self, building, charger_id):
        """Return the flat action index for a given building charger."""
        offset = 0
        for b in self.env.buildings:
            if b is building:
                break
            offset += b.action_space.shape[0]
        else:
            raise AssertionError('Target building not found while computing action index.')

        action_key = f'electric_vehicle_storage_{charger_id}'
        if action_key not in building.active_actions:
            raise AssertionError(f'{action_key} not active for building {building.name}')

        local_index = building.active_actions.index(action_key)
        return offset + local_index

    @staticmethod
    def _get_charger(building, charger_id):
        for charger in building.electric_vehicle_chargers or []:
            if charger.charger_id == charger_id:
                return charger

        raise AssertionError(f'Charger {charger_id} not found in building {building.name}')

    def _step_zero(self):
        """Advance environment one step with zero actions."""
        return self.env.step(self._zero_actions())

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------

    def test_soc_persists_while_connected(self):
        """Charging an already connected EV should increase SOC and keep it."""
        for _ in range(self.env.episode_time_steps - 2):
            t = self.env.time_step

            for building in self.env.buildings:
                if not building.electric_vehicle_chargers:
                    continue

                for charger in building.electric_vehicle_chargers:
                    ev = charger.connected_electric_vehicle
                    if ev is None:
                        continue

                    sim = charger.charger_simulation
                    if t + 1 >= len(sim.electric_vehicle_charger_state):
                        continue

                    connected_now = (
                        isinstance(sim.electric_vehicle_id[t], str)
                        and sim.electric_vehicle_id[t] == ev.name
                        and sim.electric_vehicle_charger_state[t] == 1
                    )
                    connected_next = sim.electric_vehicle_charger_state[t + 1] == 1

                    if connected_now and connected_next:
                        initial_soc = float(ev.battery.soc[t])
                        actions = self._zero_actions()
                        action_idx = self._ev_action_index(building, charger.charger_id)
                        actions[0][action_idx] = 1.0
                        self.env.step(actions)

                        charger_after = self._get_charger(building, charger.charger_id)
                        ev_after = charger_after.connected_electric_vehicle
                        self.assertIsNotNone(ev_after)

                        soc_after = float(ev_after.battery.soc[self.env.time_step])
                        self.assertGreater(
                            soc_after,
                            initial_soc + 1e-3,
                            'EV SOC did not reflect charging action while connected.',
                        )
                        return

            self._step_zero()

        self.fail('No EV found that remains connected for consecutive steps.')

    def test_soc_resets_on_arrival(self):
        """EV arriving at a charger should adopt dataset arrival SOC."""
        for _ in range(self.env.episode_time_steps - 2):
            t = self.env.time_step

            for building in self.env.buildings:
                if not building.electric_vehicle_chargers:
                    continue

                for charger in building.electric_vehicle_chargers:
                    sim = charger.charger_simulation

                    if t + 1 >= len(sim.electric_vehicle_charger_state):
                        continue

                    curr_state = sim.electric_vehicle_charger_state[t]
                    next_state = sim.electric_vehicle_charger_state[t + 1]
                    next_id = sim.electric_vehicle_id[t + 1] if t + 1 < len(sim.electric_vehicle_id) else ''

                    if not isinstance(next_id, str) or next_id.strip() == '':
                        continue

                    if curr_state == 1 or next_state != 1:
                        continue

                    if curr_state == 2:
                        if t >= len(sim.electric_vehicle_estimated_soc_arrival):
                            continue
                        expected_soc = float(sim.electric_vehicle_estimated_soc_arrival[t])
                    else:
                        if t + 1 >= len(sim.electric_vehicle_estimated_soc_arrival):
                            continue
                        expected_soc = float(sim.electric_vehicle_estimated_soc_arrival[t + 1])

                    if not (0 <= expected_soc <= 1):
                        continue

                    self._step_zero()

                    charger_after = self._get_charger(building, charger.charger_id)
                    ev_after = charger_after.connected_electric_vehicle
                    if ev_after is None or ev_after.name != next_id:
                        continue

                    soc_after = float(ev_after.battery.soc[self.env.time_step])
                    self.assertAlmostEqual(
                        soc_after,
                        expected_soc,
                        places=5,
                        msg='EV SOC on arrival does not match dataset expectation.',
                    )
                    return

            self._step_zero()

        self.fail('No EV reconnection event detected during simulation.')


if __name__ == '__main__':
    unittest.main()
