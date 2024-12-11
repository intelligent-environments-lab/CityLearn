from typing import List, Mapping, Tuple
from gymnasium import spaces
import numpy as np
from citylearn.base import Environment, EpisodeTracker
from citylearn.data import ElectricVehicleSimulation
from citylearn.energy_model import Battery
from citylearn.preprocessing import Normalize, PeriodicNormalization

class ElectricVehicle(Environment):

    def __init__(self, electric_vehicle_simulation: ElectricVehicleSimulation,episode_tracker: EpisodeTracker, observation_metadata: Mapping[str, bool],
                 action_metadata: Mapping[str, bool], battery: Battery = None, min_battery_soc: int = None, name: str = None, **kwargs):
        """
        Initialize the EVCar class.

        Parameters
        ----------
        electric_vehicle_simulation : ElectricVehicleSimulation
            Temporal features, locations, predicted SOCs and more.
        battery : Battery
            An instance of the Battery class.
        observation_metadata : dict
            Mapping of active and inactive observations.
        action_metadata : dict
            Mapping od active and inactive actions.
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
        self.observation_metadata = observation_metadata
        self.action_metadata = action_metadata
        self.non_periodic_normalized_observation_space_limits = None
        self.periodic_normalized_observation_space_limits = None
        self.observation_space = self.estimate_observation_space()
        self.action_space = self.estimate_action_space()
        self.min_battery_soc = min_battery_soc
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
    def min_battery_soc(self) -> int:
        """Min battery soc percentage."""

        return self.__min_battery_soc

    @property
    def observation_metadata(self) -> Mapping[str, bool]:
        """Mapping of active and inactive observations."""

        return self.__observation_metadata

    @property
    def action_metadata(self) -> Mapping[str, bool]:
        """Mapping od active and inactive actions."""

        return self.__action_metadata

    @property
    def battery(self) -> Battery:
        """Battery for Electric_Vehicle."""

        return self.__battery

    @property
    def observation_space(self) -> spaces.Box:
        """Agent observation space."""

        return self.__observation_space

    @property
    def action_space(self) -> spaces.Box:
        """Agent action spaces."""

        return self.__action_space

    @property
    def active_observations(self) -> List[str]:
        """Observations in `observation_metadata` with True value i.e. obeservable."""

        return [k for k, v in self.observation_metadata.items() if v]

    @property
    def active_actions(self) -> List[str]:
        """Actions in `action_metadata` with True value i.e.
        indicates which storage systems are to be controlled during simulation."""

        return [k for k, v in self.action_metadata.items() if v]
    
    @electric_vehicle_simulation.setter
    def electric_vehicle_simulation(self, electric_vehicle_simulation: ElectricVehicleSimulation):
        self.__electric_vehicle_simulation = electric_vehicle_simulation

    @name.setter
    def name(self, name: str):
        self.__name = name
    
    @min_battery_soc.setter
    def min_battery_soc(self, min_battery_soc: str):
        if min_battery_soc is None:
            self.__min_battery_soc = 10.0
        else:
            self.__min_battery_soc = min_battery_soc

    @observation_metadata.setter
    def observation_metadata(self, observation_metadata: Mapping[str, bool]):
        self.__observation_metadata = observation_metadata
    
    @action_metadata.setter
    def action_metadata(self, action_metadata: Mapping[str, bool]):
        self.__action_metadata = action_metadata
    
    @battery.setter
    def battery(self, battery: Battery):
        self.__battery = Battery(0.0, 0.0) if battery is None else battery
        
    @observation_space.setter
    def observation_space(self, observation_space: spaces.Box):
        self.__observation_space = observation_space
        self.non_periodic_normalized_observation_space_limits = self.estimate_observation_space_limits(
            include_all=True, periodic_normalization=False
        )
        self.periodic_normalized_observation_space_limits = self.estimate_observation_space_limits(
            include_all=True, periodic_normalization=True
        )
        
    @action_space.setter
    def action_space(self, action_space: spaces.Box):
        self.__action_space = action_space


    def adjust_electric_vehicle_soc_on_system_connection(self, soc_system_connection : float):
        """
        Adjusts the state of charge (SoC) of an electric vehicle's (Electric_Vehicle's) battery upon connection to the system.

        When an Electric_Vehicle is in transit, the system "loses" the connection and does not know how much battery
        has been used during travel. As such, when an Electric_Vehicle enters an incoming or connected state, its battery
        SoC is updated to be close to the predicted SoC at arrival present in the Electric_Vehicle dataset.

        However, predictions sometimes fail, so this method introduces variability for the simulation by
        randomly creating a discrepancy between the predicted value and a "real-world inspired" value. This discrepancy
        is generated using a normal (Gaussian) distribution, which is more likely to produce values near 0 and less
        likely to produce extreme values.

        The range of potential variation is between -30% to +30% of the predicted SoC, with most of the values
        being close to 0 (i.e., the prediction). The exact amount of variation is calculated by taking a random
        value from the normal distribution and scaling it by the predicted SoC. This value is then added to the
        predicted SoC to get the actual SoC, which can be higher or lower than the prediction.

        The difference between the actual SoC and the initial SoC (before the adjustment) is passed to the
        battery's charge method. If the difference is positive, the battery is charged; if the difference is negative,
        the battery is discharged.

        For example, if the Electric_Vehicle dataset has a predicted SoC at arrival of 20% (of the battery's total capacity),
        this method can randomly adjust the Electric_Vehicle's battery to 22% or 19%, or even by a larger margin such as 40%.

        Args:
        soc_system_connection (float): The predicted SoC at system connection, expressed as a percentage of the
        battery's total capacity.
        """

        # Get the SoC in kWh from the battery
        soc_init_kwh = self.battery.initial_soc

        # Calculate the system connection SoC in kWh
        soc_system_connection_kwh = self.battery.capacity * (soc_system_connection / 100)

        # Determine the range for random variation.
        # Here we use a normal distribution centered at 0 and a standard deviation of 0.1.
        # We also make sure that the values are truncated at -20% and +20%.
        variation_percentage = np.clip(np.random.normal(0, 0.1), -0.2, 0.2)

        # Apply the variation
        variation_kwh = variation_percentage * soc_system_connection_kwh

        # Calculate the final SoC in kWh
        soc_final_kwh = soc_system_connection_kwh + variation_kwh

        # Charge or discharge the battery to the new SoC.
        self.battery.set_ad_hoc_charge(soc_final_kwh - soc_init_kwh)

    def next_time_step(self) -> Mapping[int, str]:
        """
        Advance Electric_Vehicle to the next `time_step`s
        """
        self.battery.next_time_step()
        super().next_time_step()

        if self.electric_vehicle_simulation.electric_vehicle_charger_state[self.time_step] == 2:
            self.adjust_electric_vehicle_soc_on_system_connection(self.electric_vehicle_simulation.electric_vehicle_estimated_soc_arrival[self.time_step])

        elif self.electric_vehicle_simulation.electric_vehicle_charger_state[self.time_step] == 3:
            self.adjust_electric_vehicle_soc_on_system_connection((self.battery.soc[-1] / self.battery.capacity)*100)
            #ToDo here might be better to add only Nan as the vehicle is disconnencted


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
            'electric_vehicle_soc': self.battery.soc[self.time_step] / self.battery.capacity
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
                    low_limit[key] = 0
                    high_limit[key] = 24
            elif key in "electric_vehicle_required_soc_departure" or key in "electric_vehicle_estimated_soc_arrival"  or key in "electric_vehicle_soc":
                    low_limit[key] = 0.0
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
                limit = self.battery.nominal_power / self.battery.capacity
                low_limit.append(-limit)
                high_limit.append(limit)
        return spaces.Box(low=np.array(low_limit, dtype='float32'), high=np.array(high_limit, dtype='float32'))

    def autosize_battery(self, **kwargs):
        """Autosize `Battery` for a typical Electric_Vehicle.

        Other Parameters
        ----------------
        **kwargs : dict
            Other keyword arguments parsed to `electrical_storage` `autosize` function.
        """

        self.battery.autosize_for_EV()

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
