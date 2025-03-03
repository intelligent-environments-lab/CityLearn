import logging
from typing import List, Mapping, Tuple
from gymnasium import spaces
import numpy as np
from citylearn.base import Environment, EpisodeTracker
from citylearn.data import ElectricVehicleSimulation
from citylearn.energy_model import Battery
from citylearn.preprocessing import Normalize, PeriodicNormalization

ZERO_DIVISION_PLACEHOLDER = 0.000001
LOGGER = logging.getLogger()

class ElectricVehicle(Environment):

    def __init__(self, electric_vehicle_simulation: ElectricVehicleSimulation, episode_tracker: EpisodeTracker,
                 battery: Battery = None, name: str = None, **kwargs):
        """
        Initialize the EVCar class.

        Parameters
        ----------
        electric_vehicle_simulation : ElectricVehicleSimulation
            Temporal features, locations, predicted SOCs and more.
        battery : Battery
            An instance of the Battery class.
        name : str, optional
            Unique Electric_Vehicle name.

        Other Parameters
        ----------------
        **kwargs : dict
            Other keyword arguments used to initialize super class.
        """

        self.electric_vehicle_simulation = electric_vehicle_simulation
        self.name = name

        super().__init__(
            seconds_per_time_step=kwargs.get('seconds_per_time_step'),
            random_seed=kwargs.get('random_seed'),
            episode_tracker=episode_tracker
        )

        self.battery = battery
        self.__observation_epsilon = 0.0  # to avoid out of bound observations


    @property
    def electric_vehicle_simulation(self) -> ElectricVehicleSimulation:
        """Return the Electric_Vehicle simulation data."""

        return self.__electric_vehicle_simulation

    @property
    def name(self) -> str:
        """Unique building name."""

        return self.__name

    @property
    def battery(self) -> Battery:
        """Battery for Electric_Vehicle."""

        return self.__battery
    
    @electric_vehicle_simulation.setter
    def electric_vehicle_simulation(self, electric_vehicle_simulation: ElectricVehicleSimulation):
        self.__electric_vehicle_simulation = electric_vehicle_simulation

    @name.setter
    def name(self, name: str):
        self.__name = name
    
    @battery.setter
    def battery(self, battery: Battery):
        self.__battery = Battery(0.0, 0.0) if battery is None else battery

    def next_time_step(self) -> Mapping[int, str]:
        LOGGER.debug(f"[{self.name}] next_time_step: Starting new time step {self.time_step}")
        LOGGER.debug(
            f"[{self.name}] characteristics: Current battery SoC (% 0 to 1): {self.battery.soc[-1]}, Battery Capacity {self.battery.capacity}")

        # Check if the next time step exists in the charger state array
        if self.time_step + 1 < len(self.electric_vehicle_simulation.electric_vehicle_charger_state):
            current_charger_state = self.electric_vehicle_simulation.electric_vehicle_charger_state[self.time_step]
            next_charger_state = self.electric_vehicle_simulation.electric_vehicle_charger_state[self.time_step + 1]
            if (current_charger_state in [2, 3]) and (next_charger_state == 1):
                soc_arrival = self.electric_vehicle_simulation.electric_vehicle_soc_arrival[self.time_step]
                self.battery.force_set_soc(soc_arrival)

        if current_charger_state in [2, 3] and next_charger_state != 1 and self.time_step > 0:
            last_soc = self.battery.soc[self.time_step - 1]
            # Generate a variability factor from a normal distribution centered at 1 with std 0.2.
            variability_factor = np.random.normal(loc=1.0, scale=0.2)
            # Clip the factor to ensure it doesn't deviate by more than 40% (i.e., factor in [0.6, 1.4])
            variability_factor = np.clip(variability_factor, 0.6, 1.4)
            new_soc = np.clip(last_soc * variability_factor, 0.0, 1.0)
            self.battery.force_set_soc(new_soc)

        self.battery.next_time_step()
        super().next_time_step()

    def reset(self):
        """
        Reset the EVCar to its initial state.
        """
        super().reset()

        self.battery.reset()


    def observations(self, include_all: bool = None, normalize: bool = None, periodic_normalization: bool = None) -> \
            Mapping[str, float]:
        r"""Observations at current time step.

        Parameters
        ----------
        include_all: bool, default: False,
            Whether to estimate for all observations as listed in `observation_metadata` or only those that are active.
        normalize : bool, default: False
            Whether to apply min-max normalization bounded between [0, 1].
        periodic_normalization: bool, default: False
            Whether to apply sine-cosine normalization to cyclic observations.

        Returns
        -------
        observation_space : spaces.Box
            Observation low and high limits.
        """

        unwanted_keys = ['month', 'hour', 'day_type', "electric_vehicle_charger_state", "charger"]

        normalize = False if normalize is None else normalize
        periodic_normalization = False if periodic_normalization is None else periodic_normalization
        include_all = False if include_all is None else include_all

        data = {
            **{
                k.lstrip('_'): self.electric_vehicle_simulation.__getattr__(k.lstrip('_'))[self.time_step]
                for k, v in vars(self.electric_vehicle_simulation).items() if isinstance(v, np.ndarray) and k not in unwanted_keys
            },
            'electric_vehicle_soc': self.battery.soc[self.time_step]
        }


        if include_all:
            valid_observations = list(self.observation_metadata.keys())
        else:
            valid_observations = self.active_observations

        observations = {k: data[k] for k in valid_observations if k in data.keys()}
        unknown_observations = list(set(valid_observations).difference(observations.keys()))
        assert len(unknown_observations) == 0, f'Unknown observations: {unknown_observations}'

        low_limit, high_limit = self.periodic_normalized_observation_space_limits
        periodic_observations = self.get_periodic_observation_metadata()

        if periodic_normalization:
            observations_copy = {k: v for k, v in observations.items()}
            observations = {}
            pn = PeriodicNormalization(x_max=0)

            for k, v in observations_copy.items():
                if k in periodic_observations:
                    pn.x_max = max(periodic_observations[k])
                    sin_x, cos_x = v * pn
                    observations[f'{k}_cos'] = cos_x
                    observations[f'{k}_sin'] = sin_x
                else:
                    observations[k] = v
        else:
            pass

        if normalize:
            nm = Normalize(0.0, 1.0)

            for k, v in observations.items():
                nm.x_min = low_limit[k]
                nm.x_max = high_limit[k]
                observations[k] = v * nm
        else:
            pass
        return observations


    @staticmethod
    def get_periodic_observation_metadata() -> dict[str, range]:
        r"""Get periodic observation names and their minimum and maximum values for periodic/cyclic normalization.

        Returns
        -------
        periodic_observation_metadata: Mapping[str, int]
            Observation low and high limits.
        """

        return {
            'hour': range(1, 25),
            'day_type': range(1, 9),
            'month': range(1, 13)
        }

    def estimate_observation_space(self, include_all: bool = None, normalize: bool = None,
                                   periodic_normalization: bool = None) -> spaces.Box:
        r"""Get estimate of observation spaces.
        Parameters
        ----------
        include_all: bool, default: False,
            Whether to estimate for all observations as listed in `observation_metadata` or only those that are active.
        normalize : bool, default: False
            Whether to apply min-max normalization bounded between [0, 1].
        periodic_normalization: bool, default: False
            Whether to apply sine-cosine normalization to cyclic observations including hour, day_type and month.
        Returns
        -------
        observation_space : spaces.Box
            Observation low and high limits.
        """
        normalize = False if normalize is None else normalize
        normalized_observation_space_limits = self.estimate_observation_space_limits(
            include_all=include_all, periodic_normalization=True
        )
        unnormalized_observation_space_limits = self.estimate_observation_space_limits(
            include_all=include_all, periodic_normalization=False
        )
        if normalize:
            low_limit, high_limit = normalized_observation_space_limits
            low_limit = [0.0] * len(low_limit)
            high_limit = [1.0] * len(high_limit)
        else:
            low_limit, high_limit = unnormalized_observation_space_limits
            low_limit = list(low_limit.values())
            high_limit = list(high_limit.values())
        return spaces.Box(low=np.array(low_limit, dtype='float32'), high=np.array(high_limit, dtype='float32'))

    def estimate_observation_space_limits(self, include_all: bool = None, periodic_normalization: bool = None) -> Tuple[
        Mapping[str, float], Mapping[str, float]]:
        r"""Get estimate of observation space limits.
        Find minimum and maximum possible values of all the observations, which can then be used by the RL agent to scale the observations and train any function approximators more effectively.
        Parameters
        ----------
        include_all: bool, default: False,
            Whether to estimate for all observations as listed in `observation_metadata` or only those that are active.
        periodic_normalization: bool, default: False
            Whether to apply sine-cosine normalization to cyclic observations including hour, day_type and month.
        Returns
        -------
        observation_space_limits : Tuple[Mapping[str, float], Mapping[str, float]]
            Observation low and high limits.
        Notes
        -----
        Lower and upper bounds of net electricity consumption are rough estimates and may not be completely accurate hence,
        scaling this observation-variable using these bounds may result in normalized values above 1 or below 0.
        """
        include_all = False if include_all is None else include_all
        observation_names = list(self.observation_metadata.keys()) if include_all else self.active_observations
        periodic_normalization = False if periodic_normalization is None else periodic_normalization
        periodic_observations = self.get_periodic_observation_metadata()
        low_limit, high_limit = {}, {}
        for key in observation_names:
            if key in "electric_vehicle_departure_time" or key in "electric_vehicle_estimated_arrival_time":
                    low_limit[key] = -1
                    high_limit[key] = 24
            elif key in "electric_vehicle_required_soc_departure" or key in "electric_vehicle_estimated_soc_arrival"  or key in "electric_vehicle_soc":
                    low_limit[key] = -0.1
                    high_limit[key] = 1.0
        low_limit = {k: v - 0.05 for k, v in low_limit.items()}
        high_limit = {k: v + 0.05 for k, v in high_limit.items()}
        return low_limit, high_limit

    def estimate_action_space(self) -> spaces.Box:
        r"""Get estimate of action spaces.
        Find minimum and maximum possible values of all the actions, which can then be used by the RL agent to scale the selected actions.
        Returns
        -------
        action_space : spaces.Box
            Action low and high limits.
        Notes
        -----
        The lower and upper bounds for the `cooling_storage`, `heating_storage` and `dhw_storage` actions are set to (+/-) 1/maximum_demand for each respective end use,
        as the energy storage device can't provide the building with more energy than it will ever need for a given time step. .
        For example, if `cooling_storage` capacity is 20 kWh and the maximum `cooling_demand` is 5 kWh, its actions will be bounded between -5/20 and 5/20.
        These boundaries should speed up the learning process of the agents and make them more stable compared to setting them to -1 and 1.
        """
        low_limit, high_limit = [], []
        for key in self.active_actions:
            if key == 'electric_vehicle_storage':
                limit = self.electrical_storage.nominal_power/max(self.electrical_storage.capacity, ZERO_DIVISION_PLACEHOLDER)
                limit = min(limit, 1.0)
                low_limit.append(-limit)
                high_limit.append(limit)
        return spaces.Box(low=np.array(low_limit, dtype='float32'), high=np.array(high_limit, dtype='float32'))

    @staticmethod
    def observations_length() -> Mapping[str, int]:
        r"""Get periodic observation names and their minimum and maximum values for periodic/cyclic normalization.

        Returns
        -------
        periodic_observation_metadata: Mapping[str, int]
            Observation low and high limits.
        """

        return {
            'hour': range(1, 25),
            'day_type': range(1, 9),
            'month': range(1, 13)
        }
