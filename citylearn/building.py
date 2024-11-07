import logging
from typing import Any, List, Mapping, Tuple, Union
from gymnasium import spaces
import numpy as np
import pandas as pd
import torch
from citylearn.base import Environment, EpisodeTracker
from citylearn.data import CarbonIntensity, EnergySimulation, Pricing, TOLERANCE, Weather, ZERO_DIVISION_PLACEHOLDER
from citylearn.dynamics import Dynamics, LSTMDynamics
from citylearn.electric_vehicle_charger import Charger
from citylearn.energy_model import Battery, ElectricDevice, ElectricHeater, HeatPump, PV, StorageTank
from citylearn.occupant import LogisticRegressionOccupant, Occupant
from citylearn.power_outage import PowerOutage
from citylearn.preprocessing import Normalize, PeriodicNormalization

LOGGER = logging.getLogger()
logging.basicConfig(level=logging.INFO)

class Building(Environment):
    r"""Base class for building.

    Parameters
    ----------
    energy_simulation : EnergySimulation
        Temporal features, cooling, heating, dhw and plug loads, solar generation and indoor environment time series.
    weather : Weather
        Outdoor weather conditions and forecasts time sereis.
    observation_metadata : dict
        Mapping of active and inactive observations.
    action_metadata : dict
        Mapping od active and inactive actions.
    episode_tracker: EpisodeTracker, optional
        :py:class:`citylearn.base.EpisodeTracker` object used to keep track of current episode time steps 
        for reading observations from data files.
    carbon_intensity : CarbonIntensity, optional
        Carbon dioxide emission rate time series.
    pricing : Pricing, optional
        Energy pricing and forecasts time series.
    dhw_storage : StorageTank, optional
        Hot water storage object for domestic hot water.
    cooling_storage : StorageTank, optional
        Cold water storage object for space cooling.
    heating_storage : StorageTank, optional
        Hot water storage object for space heating.
    electrical_storage : Battery, optional
        Electric storage object for meeting electric loads.
    dhw_device : Union[HeatPump, ElectricHeater], optional
        Electric device for meeting hot domestic hot water demand and charging `dhw_storage`.
    cooling_device : HeatPump, optional
        Electric device for meeting space cooling demand and charging `cooling_storage`.
    heating_device : Union[HeatPump, ElectricHeater], optional
        Electric device for meeting space heating demand and charging `heating_storage`.
    pv : PV, optional
        PV object for offsetting electricity demand from grid.
    name : str, optional
        Unique building name.
    observation_space_limit_delta: float, default: 0.0
        +/- buffer for observation space limits after they have been dynamically calculated.
    maximum_temperature_delta: float, default: 20.0
        Expected maximum absolute temperature delta above and below indoor dry-bulb temperature in [C].
    demand_observation_limit_factor: float, default: 1.15
        Multiplier for maximum cooling/heating/dhw demand observations when setting observation limits.
    simulate_power_outage: bool, default: False
        Whether to allow time steps when the grid is unavailable and loads must be met using only the 
        building's downward flexibility resources.
    stochastic_power_outage: bool, default: False
        Whether to use a stochastic function to determine outage time steps otherwise, 
        :py:class:`citylearn.building.Building.energy_simulation.power_outage` time series is used.
    stochastic_power_outage_model: PowerOutage, optional
        Power outage model class used to generate stochastic power outage signals.
    carbon_intensity : CarbonIntensity, optional
        Carbon dioxide emission rate time series.
    electric_vehicle_chargers : Charger, optional
        Electric Vehicle Chargers associated with the building.

    Other Parameters
    ----------------
    **kwargs : Any
        Other keyword arguments used to initialize super class.
    """

    def __init__(
            self, energy_simulation: EnergySimulation, weather: Weather, observation_metadata: Mapping[str, bool],
            action_metadata: Mapping[str, bool], episode_tracker: EpisodeTracker,
            carbon_intensity: CarbonIntensity = None,
            pricing: Pricing = None, dhw_storage: StorageTank = None, cooling_storage: StorageTank = None,
            heating_storage: StorageTank = None, electrical_storage: Battery = None,
            dhw_device: Union[HeatPump, ElectricHeater] = None, cooling_device: HeatPump = None,
            heating_device: Union[HeatPump, ElectricHeater] = None, pv: PV = None, name: str = None,
            maximum_temperature_delta: float = None, observation_space_limit_delta: float = None,
            demand_observation_limit_factor: float = None, simulate_power_outage: bool = None,
            stochastic_power_outage: bool = None, stochastic_power_outage_model: PowerOutage = None,
            electric_vehicle_chargers: List[Charger] = None, **kwargs: Any
    ):
        self.name = name
        self.dhw_storage = dhw_storage
        self.cooling_storage = cooling_storage
        self.heating_storage = heating_storage
        self.electrical_storage = electrical_storage
        self.dhw_device = dhw_device
        self.cooling_device = cooling_device
        self.heating_device = heating_device
        self.__non_shiftable_load_device = ElectricDevice(0.0)
        self.pv = pv
        super().__init__(
            seconds_per_time_step=kwargs.get('seconds_per_time_step'),
            random_seed=kwargs.get('random_seed'),
            episode_tracker=episode_tracker
        )
        self.stochastic_power_outage_model = stochastic_power_outage_model
        self.electric_vehicle_chargers = electric_vehicle_chargers
        self.energy_simulation = energy_simulation
        self.weather = weather
        self.carbon_intensity = carbon_intensity
        self.pricing = pricing
        self.observation_metadata = observation_metadata
        self.action_metadata = action_metadata
        self.observation_space_limit_delta = observation_space_limit_delta
        self.maximum_temperature_delta = maximum_temperature_delta
        self.demand_observation_limit_factor = demand_observation_limit_factor
        self.simulate_power_outage = simulate_power_outage
        self.stochastic_power_outage = stochastic_power_outage
        self.non_periodic_normalized_observation_space_limits = None
        self.periodic_normalized_observation_space_limits = None
        self.observation_space = self.estimate_observation_space(include_all=False, normalize=False)
        self.action_space = self.estimate_action_space()

    @property
    def energy_simulation(self) -> EnergySimulation:
        """Temporal features, cooling, heating, dhw and plug loads, solar generation and indoor environment time series."""

        return self.__energy_simulation

    @property
    def weather(self) -> Weather:
        """Outdoor weather conditions and forecasts time series."""

        return self.__weather

    @property
    def observation_metadata(self) -> Mapping[str, bool]:
        """Mapping of active and inactive observations."""

        return self.__observation_metadata

    @property
    def action_metadata(self) -> Mapping[str, bool]:
        """Mapping od active and inactive actions."""

        return self.__action_metadata

    @property
    def carbon_intensity(self) -> CarbonIntensity:
        """Carbon dioxide emission rate time series."""

        return self.__carbon_intensity

    @property
    def pricing(self) -> Pricing:
        """Energy pricing and forecasts time series."""

        return self.__pricing

    @property
    def dhw_storage(self) -> StorageTank:
        """Hot water storage object for domestic hot water."""

        return self.__dhw_storage

    @property
    def cooling_storage(self) -> StorageTank:
        """Cold water storage object for space cooling."""

        return self.__cooling_storage

    @property
    def heating_storage(self) -> StorageTank:
        """Hot water storage object for space heating."""

        return self.__heating_storage

    @property
    def electrical_storage(self) -> Battery:
        """Electric storage object for meeting electric loads."""

        return self.__electrical_storage

    @property
    def dhw_device(self) -> Union[HeatPump, ElectricHeater]:
        """Electric device for meeting hot domestic hot water demand and charging `dhw_storage`."""

        return self.__dhw_device

    @property
    def cooling_device(self) -> HeatPump:
        """Electric device for meeting space cooling demand and charging `cooling_storage`."""

        return self.__cooling_device

    @property
    def heating_device(self) -> Union[HeatPump, ElectricHeater]:
        """Electric device for meeting space heating demand and charging `heating_storage`."""

        return self.__heating_device

    @property
    def non_shiftable_load_device(self) -> ElectricDevice:
        """Generic electric device for meeting non_shiftable_load."""

        return self.__non_shiftable_load_device

    @property
    def pv(self) -> PV:
        """PV object for offsetting electricity demand from grid."""

        return self.__pv

    @property
    def electric_vehicle_chargers(self) -> List[Charger]:
        """Electric Vehicle Chargers associated with the building for charging connected eletric vehicles."""

        return self.__electric_vehicle_chargers

    @property
    def name(self) -> str:
        """Unique building name."""

        return self.__name

    @property
    def observation_space_limit_delta(self) -> float:
        """+/- buffer for observation space limits after they have been dynamically calculated."""

        return self.__observation_space_limit_delta

    @property
    def maximum_temperature_delta(self) -> float:
        """Expected maximum absolute temperature delta above and below indoor dry-bulb temperature in [C]."""

        return self.__maximum_temperature_delta

    @property
    def demand_observation_limit_factor(self) -> float:
        """Multiplier for maximum cooling/heating/dhw demand observations when setting observation limits."""

        return self.__demand_observation_limit_factor

    @property
    def simulate_power_outage(self) -> bool:
        """Whether to allow time steps when the grid is unavailable and loads must be met using only the 
        building's downward flexibility resources."""

        return self.__simulate_power_outage

    @property
    def stochastic_power_outage(self) -> bool:
        """Whether to use a stochastic function to determine outage time steps otherwise, 
        :py:class:`citylearn.building.Building.energy_simulation.power_outage` time series is used."""

        return self.__stochastic_power_outage

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

    @property
    def net_electricity_consumption_emission_without_storage_and_pv(self) -> np.ndarray:
        """Carbon dioxide emmission from `net_electricity_consumption_without_storage_pv` time series, in [kg_co2]."""

        return (
                self.carbon_intensity.carbon_intensity[
                0:self.time_step + 1] * self.net_electricity_consumption_without_storage_and_pv
        ).clip(min=0)

    @property
    def net_electricity_consumption_cost_without_storage_and_pv(self) -> np.ndarray:
        """net_electricity_consumption_without_storage_and_pv` cost time series, in [$]."""

        return self.pricing.electricity_pricing[
               0:self.time_step + 1] * self.net_electricity_consumption_without_storage_and_pv

    @property
    def net_electricity_consumption_without_storage_and_pv(self) -> np.ndarray:
        """Net electricity consumption in the absence of flexibility provided by storage devices, 
        and self generation time series, in [kWh]. 
        
        Notes
        -----
        net_electricity_consumption_without_storage_and_pv = 
        `net_electricity_consumption_without_storage` - `solar_generation`
        """

        return self.net_electricity_consumption_without_storage - self.solar_generation

    @property
    def net_electricity_consumption_emission_without_storage(self) -> np.ndarray:
        """Carbon dioxide emmission from `net_electricity_consumption_without_storage` time series, in [kg_co2]."""

        return (self.carbon_intensity.carbon_intensity[
                0:self.time_step + 1] * self.net_electricity_consumption_without_storage).clip(min=0)

    @property
    def net_electricity_consumption_cost_without_storage(self) -> np.ndarray:
        """`net_electricity_consumption_without_storage` cost time series, in [$]."""

        return self.pricing.electricity_pricing[0:self.time_step + 1] * self.net_electricity_consumption_without_storage

    @property
    def net_electricity_consumption_without_storage(self) -> np.ndarray:
        """net electricity consumption in the absence of flexibility provided by storage devices time series, in [kWh]. 
        
        Notes
        -----
        net_electricity_consumption_without_storage = `net_electricity_consumption` - (`cooling_storage_electricity_consumption`
        + `heating_storage_electricity_consumption` + `dhw_storage_electricity_consumption` + `electrical_storage_electricity_consumption` + `charger_electricity_consumption`)

        Regarding electric vehicles there is:
        chargers_electricity_consumption -> Sum of the electricity consumption of all electric_vehicle_chargers in the building

        So, the first one is subtracted from the net_electricity_consumption, obtaining the energy consumption as if the cars were not used at all.
        However, if there are chargers and EVs, they need to charge per usual, so that consumption is added
        This is what allows to check if the control mechanism affects the grid balancing scheme for EVs for example.
        """

        return self.net_electricity_consumption - np.sum([
            self.cooling_storage_electricity_consumption,
            self.heating_storage_electricity_consumption,
            self.dhw_storage_electricity_consumption,
            self.electrical_storage_electricity_consumption,
            self.__chargers_electricity_consumption
        ], axis=0)

    @property
    def net_electricity_consumption_emission(self) -> np.ndarray:
        """Carbon dioxide emmission from `net_electricity_consumption` time series, in [kg_co2]."""

        return self.__net_electricity_consumption_emission[:self.time_step + 1]

    @property
    def net_electricity_consumption_cost(self) -> np.ndarray:
        """`net_electricity_consumption` cost time series, in [$]."""

        return self.__net_electricity_consumption_cost[:self.time_step + 1]

    @property
    def net_electricity_consumption(self) -> np.ndarray:
        """Net electricity consumption time series, in [kWh]."""

        return self.__net_electricity_consumption[:self.time_step + 1]

    @property
    def cooling_electricity_consumption(self) -> np.ndarray:
        """`cooling_device` net electricity consumption in meeting cooling demand and `cooling_storage` energy demand time series, in [kWh]. 
        """

        return self.cooling_device.electricity_consumption[:self.time_step + 1]

    @property
    def heating_electricity_consumption(self) -> np.ndarray:
        """`heating_device` net electricity consumption in meeting heating demand and `heating_storage` energy demand time series, in [kWh]. 
        """

        return self.heating_device.electricity_consumption[:self.time_step + 1]

    @property
    def dhw_electricity_consumption(self) -> np.ndarray:
        """`dhw_device` net electricity consumption in meeting domestic hot water and `dhw_storage` energy demand time series, in [kWh]. 
        """

        return self.dhw_device.electricity_consumption[:self.time_step + 1]

    @property
    def non_shiftable_load_electricity_consumption(self) -> np.ndarray:
        """`non_shiftable_load_device` net electricity consumption in meeting `non_shiftable_load` energy demand time series, in [kWh]. 
        """

        return self.non_shiftable_load_device.electricity_consumption[:self.time_step + 1]

    @property
    def cooling_storage_electricity_consumption(self) -> np.ndarray:
        """`cooling_storage` net electricity consumption time series, in [kWh]. 
        
        Positive values indicate `cooling_device` electricity consumption to charge `cooling_storage` while negative values indicate avoided `cooling_device` 
        electricity consumption by discharging `cooling_storage` to meet `cooling_demand`.
        """

        return self.cooling_device.get_input_power(self.cooling_storage.energy_balance[:self.time_step + 1],
                                                   self.weather.outdoor_dry_bulb_temperature[:self.time_step + 1],
                                                   False)

    @property
    def heating_storage_electricity_consumption(self) -> np.ndarray:
        """`heating_storage` net electricity consumption time series, in [kWh]. 
        
        Positive values indicate `heating_device` electricity consumption to charge `heating_storage` while negative values indicate avoided `heating_device` 
        electricity consumption by discharging `heating_storage` to meet `heating_demand`.
        """

        if isinstance(self.heating_device, HeatPump):
            consumption = self.heating_device.get_input_power(self.heating_storage.energy_balance[:self.time_step + 1],
                                                              self.weather.outdoor_dry_bulb_temperature[
                                                              :self.time_step + 1], True)
        else:
            consumption = self.heating_device.get_input_power(self.heating_storage.energy_balance[:self.time_step + 1])

        return consumption

    @property
    def dhw_storage_electricity_consumption(self) -> np.ndarray:
        """`dhw_storage` net electricity consumption time series, in [kWh]. 
        
        Positive values indicate `dhw_device` electricity consumption to charge `dhw_storage` while negative values indicate avoided `dhw_device` 
        electricity consumption by discharging `dhw_storage` to meet `dhw_demand`.
        """

        if isinstance(self.dhw_device, HeatPump):
            consumption = self.dhw_device.get_input_power(self.dhw_storage.energy_balance[:self.time_step + 1],
                                                          self.weather.outdoor_dry_bulb_temperature[
                                                          :self.time_step + 1], True)
        else:
            consumption = self.dhw_device.get_input_power(self.dhw_storage.energy_balance[:self.time_step + 1])

        return consumption

    @property
    def electrical_storage_electricity_consumption(self) -> np.ndarray:
        """Energy supply from grid and/or `PV` to `electrical_storage` time series, in [kWh]."""

        return self.electrical_storage.electricity_consumption[:self.time_step + 1]

    @property
    def energy_from_cooling_device_to_cooling_storage(self) -> np.ndarray:
        """Energy supply from `cooling_device` to `cooling_storage` time series, in [kWh]."""

        return self.cooling_storage.energy_balance.clip(min=0)[:self.time_step + 1]

    @property
    def energy_from_heating_device_to_heating_storage(self) -> np.ndarray:
        """Energy supply from `heating_device` to `heating_storage` time series, in [kWh]."""

        return self.heating_storage.energy_balance.clip(min=0)[:self.time_step + 1]

    @property
    def energy_from_dhw_device_to_dhw_storage(self) -> np.ndarray:
        """Energy supply from `dhw_device` to `dhw_storage` time series, in [kWh]."""

        return self.dhw_storage.energy_balance.clip(min=0)[:self.time_step + 1]

    @property
    def energy_to_electrical_storage(self) -> np.ndarray:
        """Energy supply from `electrical_device` to building time series, in [kWh]."""

        return self.electrical_storage.energy_balance.clip(min=0)[:self.time_step + 1]

    @property
    def energy_from_cooling_device(self) -> np.ndarray:
        """Energy supply from `cooling_device` to building time series, in [kWh]."""

        return self.__energy_from_cooling_device[:self.time_step + 1]

    @property
    def energy_from_heating_device(self) -> np.ndarray:
        """Energy supply from `heating_device` to building time series, in [kWh]."""

        return self.__energy_from_heating_device[:self.time_step + 1]

    @property
    def energy_from_dhw_device(self) -> np.ndarray:
        """Energy supply from `dhw_device` to building time series, in [kWh]."""

        return self.__energy_from_dhw_device[:self.time_step + 1]

    @property
    def energy_to_non_shiftable_load(self) -> np.ndarray:
        """Energy supply from grid, PV and battery to non shiftable loads, in [kWh]."""

        return self.__energy_to_non_shiftable_load[:self.time_step + 1]

    @property
    def energy_from_cooling_storage(self) -> np.ndarray:
        """Energy supply from `cooling_storage` to building time series, in [kWh]."""

        return self.cooling_storage.energy_balance.clip(max=0)[:self.time_step + 1] * -1

    @property
    def energy_from_heating_storage(self) -> np.ndarray:
        """Energy supply from `heating_storage` to building time series, in [kWh]."""

        return self.heating_storage.energy_balance.clip(max=0)[:self.time_step + 1] * -1

    @property
    def energy_from_dhw_storage(self) -> np.ndarray:
        """Energy supply from `dhw_storage` to building time series, in [kWh]."""

        return self.dhw_storage.energy_balance.clip(max=0)[:self.time_step + 1] * -1

    @property
    def energy_from_electrical_storage(self) -> np.ndarray:
        """Energy supply from `electrical_storage` to building time series, in [kWh]."""

        return self.electrical_storage.energy_balance.clip(max=0)[:self.time_step + 1] * -1

    @property
    def indoor_dry_bulb_temperature(self) -> np.ndarray:
        """dry bulb temperature time series, in [C].
        
        This is the temperature when cooling_device and heating_device are controlled.
        """

        return self.energy_simulation.indoor_dry_bulb_temperature[0:self.time_step + 1]

    @property
    def indoor_dry_bulb_temperature_cooling_set_point(self) -> np.ndarray:
        """Dry bulb temperature cooling set point time series, in [C]."""

        return self.energy_simulation.indoor_dry_bulb_temperature_cooling_set_point[0:self.time_step + 1]

    @property
    def indoor_dry_bulb_temperature_heating_set_point(self) -> np.ndarray:
        """Dry bulb temperature heating set point time series, in [C]."""

        return self.energy_simulation.indoor_dry_bulb_temperature_heating_set_point[0:self.time_step + 1]

    @property
    def comfort_band(self) -> np.ndarray:
        """Occupant comfort band above the `indoor_dry_bulb_temperature_cooling_set_point` and below the `indoor_dry_bulb_temperature_heating_set_point`, in [C]."""

        return self.energy_simulation.comfort_band[0:self.time_step + 1]

    @property
    def occupant_count(self) -> np.ndarray:
        """Building occupant count time series, in [people]."""

        return self.energy_simulation.occupant_count[0:self.time_step + 1]

    @property
    def cooling_demand(self) -> np.ndarray:
        """Space cooling demand to be met by `cooling_device` and/or `cooling_storage` time series, in [kWh]."""

        return self.energy_simulation.cooling_demand[0:self.time_step + 1]

    @property
    def heating_demand(self) -> np.ndarray:
        """Space heating demand to be met by `heating_device` and/or `heating_storage` time series, in [kWh]."""

        return self.energy_simulation.heating_demand[0:self.time_step + 1]

    @property
    def dhw_demand(self) -> np.ndarray:
        """Domestic hot water demand to be met by `dhw_device` and/or `dhw_storage` time series, in [kWh]."""

        return self.energy_simulation.dhw_demand[0:self.time_step + 1]

    @property
    def non_shiftable_load(self) -> np.ndarray:
        """Electricity load that must be met by the grid, or `PV` and/or `electrical_storage` if available time series, in [kWh]."""

        return self.energy_simulation.non_shiftable_load[0:self.time_step + 1]

    @property
    def cooling_device_cop(self) -> np.ndarray:
        """Heat pump `cooling_device` coefficient of performance time series."""

        return self.cooling_device.get_cop(self.weather.outdoor_dry_bulb_temperature, heating=False)[
               0:self.time_step + 1]

    @property
    def heating_device_cop(self) -> np.ndarray:
        """Heat pump `heating_device` coefficient of performance or electric heater `heating_device` static technical efficiency time series."""

        return self.heating_device.get_cop(self.weather.outdoor_dry_bulb_temperature, heating=True)[
               0:self.time_step + 1] \
            if isinstance(self.heating_device, HeatPump) else np.zeros(self.time_step + 1, dtype='float32')

    @property
    def dhw_device_cop(self) -> np.ndarray:
        """Heat pump `dhw_device` coefficient of performance or electric heater `dhw_device` static technical efficiency time series."""

        return self.dhw_device.get_cop(self.weather.outdoor_dry_bulb_temperature, heating=True)[0:self.time_step + 1] \
            if isinstance(self.dhw_device, HeatPump) else np.zeros(self.time_step + 1, dtype='float32')

    @property
    def solar_generation(self) -> np.ndarray:
        """`PV` solar generation (negative value) time series, in [kWh]."""

        return self.__solar_generation[:self.time_step + 1]

    @property
    def power_outage_signal(self) -> np.ndarray:
        """Power outage signal time series, in [Yes/No]."""

        return self.__power_outage_signal[:self.time_step + 1]

    @property
    def downward_electrical_flexibility(self) -> float:
        """Available distributed energy resource capacity to satisfy electric loads while considering power outage at current time step.
        
        It is the sum of solar generation and any discharge from electrical storage, less electricity consumption by cooling, heating, 
        dhw and non-shfitable load devices as well as charging electrical storage. When there is no power outage, the returned value 
        is `np.inf`.
        """

        capacity = abs(self.solar_generation[self.time_step]) - (
                self.cooling_device.electricity_consumption[self.time_step]
                + self.heating_device.electricity_consumption[self.time_step]
                + self.dhw_device.electricity_consumption[self.time_step]
                + self.non_shiftable_load_device.electricity_consumption[self.time_step]
                + self.electrical_storage.electricity_consumption[self.time_step]
        )
        capacity = capacity if self.power_outage else np.inf

        message = 'downward_electrical_flexibility must be >= 0.0!' \
                  f'time step:, {self.time_step}, outage:, {self.power_outage}, capacity:, {capacity},' \
                  f' solar:, {abs(self.solar_generation[self.time_step])},' \
                  f' cooling:, {self.cooling_device.electricity_consumption[self.time_step]},' \
                  f' heating:, {self.heating_device.electricity_consumption[self.time_step]},' \
                  f'dhw:, {self.dhw_device.electricity_consumption[self.time_step]},' \
                  f'non-shiftable:, {self.non_shiftable_load_device.electricity_consumption[self.time_step]},' \
                  f' battery:, {self.electrical_storage.electricity_consumption[self.time_step]}'
        assert capacity >= 0.0 or abs(capacity) < TOLERANCE, message
        capacity = max(0.0, capacity)

        return capacity

    @property
    def power_outage(self) -> bool:
        """Whether there is power outage at current time step."""

        return self.simulate_power_outage and bool(self.__power_outage_signal[self.time_step])

    @property
    def stochastic_power_outage_model(self) -> PowerOutage:
        """Power outage model class used to generate stochastic power outage signals."""

        return self.__stochastic_power_outage_model

    @energy_simulation.setter
    def energy_simulation(self, energy_simulation: EnergySimulation):
        self.__energy_simulation = energy_simulation

    @weather.setter
    def weather(self, weather: Weather):
        self.__weather = weather

    @observation_metadata.setter
    def observation_metadata(self, observation_metadata: Mapping[str, bool]):
        self.__observation_metadata = observation_metadata

    @action_metadata.setter
    def action_metadata(self, action_metadata: Mapping[str, bool]):
        self.__action_metadata = action_metadata

    @carbon_intensity.setter
    def carbon_intensity(self, carbon_intensity: CarbonIntensity):
        if carbon_intensity is None:
            self.__carbon_intensity = CarbonIntensity(
                np.zeros(self.episode_tracker.simulation_time_steps, dtype='float32'))
        else:
            self.__carbon_intensity = carbon_intensity

    @pricing.setter
    def pricing(self, pricing: Pricing):
        if pricing is None:
            self.__pricing = Pricing(
                np.zeros(self.episode_tracker.simulation_time_steps, dtype='float32'),
                np.zeros(self.episode_tracker.simulation_time_steps, dtype='float32'),
                np.zeros(self.episode_tracker.simulation_time_steps, dtype='float32'),
                np.zeros(self.episode_tracker.simulation_time_steps, dtype='float32'),
            )
        else:
            self.__pricing = pricing

    @dhw_storage.setter
    def dhw_storage(self, dhw_storage: StorageTank):
        self.__dhw_storage = StorageTank(0.0) if dhw_storage is None else dhw_storage

    @cooling_storage.setter
    def cooling_storage(self, cooling_storage: StorageTank):
        self.__cooling_storage = StorageTank(0.0) if cooling_storage is None else cooling_storage

    @heating_storage.setter
    def heating_storage(self, heating_storage: StorageTank):
        self.__heating_storage = StorageTank(0.0) if heating_storage is None else heating_storage

    @electrical_storage.setter
    def electrical_storage(self, electrical_storage: Battery):
        self.__electrical_storage = Battery(0.0, 0.0) if electrical_storage is None else electrical_storage

    @dhw_device.setter
    def dhw_device(self, dhw_device: Union[HeatPump, ElectricHeater]):
        self.__dhw_device = ElectricHeater(0.0) if dhw_device is None else dhw_device

    @cooling_device.setter
    def cooling_device(self, cooling_device: HeatPump):
        self.__cooling_device = HeatPump(0.0) if cooling_device is None else cooling_device

    @heating_device.setter
    def heating_device(self, heating_device: Union[HeatPump, ElectricHeater]):
        self.__heating_device = HeatPump(0.0) if heating_device is None else heating_device

    @pv.setter
    def pv(self, pv: PV):
        self.__pv = PV(0.0) if pv is None else pv

    @electric_vehicle_chargers.setter
    def electric_vehicle_chargers(self, electric_vehicle_chargers: List[Charger]):
        self.__electric_vehicle_chargers = electric_vehicle_chargers

    @observation_space.setter
    def observation_space(self, observation_space: spaces.Box):
        self.__observation_space = observation_space
        self.non_periodic_normalized_observation_space_limits = self.estimate_observation_space_limits(include_all=True,
                                                                                                       periodic_normalization=False)
        self.periodic_normalized_observation_space_limits = self.estimate_observation_space_limits(include_all=True,
                                                                                                   periodic_normalization=True)

    @action_space.setter
    def action_space(self, action_space: spaces.Box):
        self.__action_space = action_space

    @name.setter
    def name(self, name: str):
        self.__name = self.uid if name is None else name

    @observation_space_limit_delta.setter
    def observation_space_limit_delta(self, observation_space_limit_delta: float):
        self.__observation_space_limit_delta = 0.0 if observation_space_limit_delta is None else observation_space_limit_delta

        if hasattr(self, 'observation_space') and self.observation_space is not None:
            self.observation_space = self.estimate_observation_space(include_all=False, normalize=False)

        else:
            pass

    @maximum_temperature_delta.setter
    def maximum_temperature_delta(self, maximum_temperature_delta: float):
        self.__maximum_temperature_delta = 20.0 if maximum_temperature_delta is None else maximum_temperature_delta

        if hasattr(self, 'observation_space') and self.observation_space is not None:
            self.observation_space = self.estimate_observation_space(include_all=False, normalize=False)

        else:
            pass

    @demand_observation_limit_factor.setter
    def demand_observation_limit_factor(self, demand_observation_limit_factor: float):
        self.__demand_observation_limit_factor = 2.0 if demand_observation_limit_factor is None else demand_observation_limit_factor

        if hasattr(self, 'observation_space') and self.observation_space is not None:
            self.observation_space = self.estimate_observation_space(include_all=False, normalize=False)

        else:
            pass

    @stochastic_power_outage_model.setter
    def stochastic_power_outage_model(self, stochastic_power_outage_model: PowerOutage):
        self.__stochastic_power_outage_model = PowerOutage() if stochastic_power_outage_model is None else stochastic_power_outage_model

    @simulate_power_outage.setter
    def simulate_power_outage(self, simulate_power_outage: bool):
        self.__simulate_power_outage = False if simulate_power_outage is None else simulate_power_outage

    @stochastic_power_outage.setter
    def stochastic_power_outage(self, stochastic_power_outage: bool):
        self.__stochastic_power_outage = False if stochastic_power_outage is None else stochastic_power_outage

    @Environment.random_seed.setter
    def random_seed(self, seed: int):
        Environment.random_seed.fset(self, seed)
        self.cooling_device.random_seed = self.random_seed
        self.heating_device.random_seed = self.random_seed
        self.dhw_device.random_seed = self.random_seed
        self.cooling_storage.random_seed = self.random_seed
        self.heating_storage.random_seed = self.random_seed
        self.electrical_storage.random_seed = self.random_seed
        self.pv.random_seed = self.random_seed

    @Environment.episode_tracker.setter
    def episode_tracker(self, episode_tracker: EpisodeTracker):
        Environment.episode_tracker.fset(self, episode_tracker)
        self.cooling_device.episode_tracker = self.episode_tracker
        self.heating_device.episode_tracker = self.episode_tracker
        self.dhw_device.episode_tracker = self.episode_tracker
        self.cooling_storage.episode_tracker = self.episode_tracker
        self.heating_storage.episode_tracker = self.episode_tracker
        self.dhw_storage.episode_tracker = self.episode_tracker
        self.electrical_storage.episode_tracker = self.episode_tracker
        self.non_shiftable_load_device.episode_tracker = self.episode_tracker
        self.pv.episode_tracker = self.episode_tracker

    def get_metadata(self) -> Mapping[str, Any]:
        n_years = max(1, (self.episode_tracker.episode_time_steps * self.seconds_per_time_step) / (8760 * 3600))
        return {
            **super().get_metadata(),
            'name': self.name,
            'observation_metadata': self.observation_metadata,
            'action_metadata': self.action_metadata,
            'maximum_temperature_delta': self.maximum_temperature_delta,
            'cooling_device': self.cooling_device.get_metadata(),
            'heating_device': self.heating_device.get_metadata(),
            'dhw_device': self.dhw_device.get_metadata(),
            'non_shiftable_load_device': self.non_shiftable_load_device.get_metadata(),
            'cooling_storage': self.cooling_storage.get_metadata(),
            'heating_storage': self.heating_storage.get_metadata(),
            'dhw_storage': self.dhw_storage.get_metadata(),
            'electrical_storage': self.electrical_storage.get_metadata(),
            'pv': self.pv.get_metadata(),
            'annual_cooling_demand_estimate': self.energy_simulation.cooling_demand.sum() / n_years,
            'annual_heating_demand_estimate': self.energy_simulation.heating_demand.sum() / n_years,
            'annual_dhw_demand_estimate': self.energy_simulation.dhw_demand.sum() / n_years,
            'annual_non_shiftable_load_estimate': self.energy_simulation.non_shiftable_load.sum() / n_years,
            'annual_solar_generation_estimate': self.pv.get_generation(
                self.energy_simulation.solar_generation).sum() / n_years,
        }

    def observations(self, include_all: bool = None, normalize: bool = None, periodic_normalization: bool = None,
                     check_limits: bool = None) -> Mapping[str, float]:
        r"""Observations at current time step.

        Parameters
        ----------
        include_all: bool, default: False,
            Whether to estimate for all observations as listed in `observation_metadata` or only those that are active.
        normalize : bool, default: False
            Whether to apply min-max normalization bounded between [0, 1].
        periodic_normalization: bool, default: False
            Whether to apply sine-cosine normalization to cyclic observations including hour, day_type and month.
        check_limits: bool, default: False
            Whether to check if observations are within observation space and if not, will send output to log describing 
            out of bounds observations. Useful for agents that will fail if observations fall outside space e.g. RLlib agents.

        Returns
        -------
        observation_space : spaces.Box
            Observation low and high limits.

        Notes
        -----
        Lower and upper bounds of net electricity consumption are rough estimates and may not be completely accurate hence,
        scaling this observation-variable using these bounds may result in normalized values above 1 or below 0.
        """

        normalize = False if normalize is None else normalize
        periodic_normalization = False if periodic_normalization is None else periodic_normalization
        include_all = False if include_all is None else include_all
        check_limits = False if check_limits is None else check_limits

        observations = {}
        data = self._get_observations_data()

        if include_all:
            valid_observations = list(data.keys())
        else:
            valid_observations = self.active_observations

        observations = {k: data[k] for k in valid_observations if k in data.keys()}

        # Observations for electric_vehicle_chargers
        # Connected is 1 and disconnected is 0
        if self.electric_vehicle_chargers is not None:
            for charger in self.electric_vehicle_chargers:  # If present, itrerate to each charger
                charger_id = charger.charger_id
                charger_key_state = f'charger_{charger_id}_connected_state'  # Observations names are composed from the charger unique ID
                charger_key_incoming_state = f'charger_{charger_id}_incoming_state'

                if charger.connected_electric_vehicle:
                    observations[
                        charger_key_state] = 1  # attributes the f'charger_{charger_id}_connected_state' the value of one (connected EV)
                    obs = charger.connected_electric_vehicle.observations(include_all, normalize, periodic_normalization)
                    for k, v in obs.items():
                        observations[
                            f'charger_{charger_id}_connected_{k}'] = v  # for the connected EV several observations are added (according to the observations specified in Electric_Vehicle class
                else:  # otherwise, when not connected, 0 is given and observations are filled with -1
                    observations[charger_key_state] = 0
                    for o in self.observation_metadata:
                        if f'charger_{charger_id}_connected' in o and o != charger_key_state:
                            observations[o] = -0.1

                # same logic for incoming EV, which states if an EV is routing towards the charger
                if charger.incoming_electric_vehicle:
                    observations[charger_key_incoming_state] = 1
                    obs = charger.incoming_electric_vehicle.observations(include_all, normalize, periodic_normalization)
                    for k, v in obs.items():
                        observations[f'charger_{charger_id}_incoming_{k}'] = v
                else:
                    observations[charger_key_incoming_state] = 0
                    for o in self.observation_metadata:
                        if f'charger_{charger_id}_incoming' in o and o != charger_key_incoming_state:
                            observations[o] = -0.1

        unknown_observations = list(set(valid_observations).difference(observations.keys()))
        assert len(unknown_observations) == 0, f'Unknown observations: {unknown_observations}'

        non_periodic_low_limit, non_periodic_high_limit = self.non_periodic_normalized_observation_space_limits
        periodic_low_limit, periodic_high_limit = self.periodic_normalized_observation_space_limits
        periodic_observations = self.get_periodic_observation_metadata()

        if check_limits:
            for k in self.active_observations:
                value = observations[k]
                lower = non_periodic_low_limit[k]
                upper = non_periodic_high_limit[k]

                if not lower <= value <= upper:
                    report = {
                        'Building': self.name,
                        'episode': self.episode_tracker.episode,
                        'time_step': f'{self.time_step + 1}/{self.episode_tracker.episode_time_steps}',
                        'observation': k,
                        'value': value,
                        'lower': lower,
                        'upper': upper
                    }
                    LOGGER.debug(f'Observation outside space limit: {report}')

                else:
                    pass

            else:
                pass

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
                nm.x_min = periodic_low_limit[k]
                nm.x_max = periodic_high_limit[k]
                observations[k] = v * nm

        else:
            pass

        return observations
    
    def _get_observations_data(self) -> Mapping[str, Union[float, int]]:
        return {
            **{
                k.lstrip('_'): self.energy_simulation.__getattr__(k.lstrip('_'))[self.time_step] 
                for k, v in vars(self.energy_simulation).items() if isinstance(v, np.ndarray)
            },
            **{
                k.lstrip('_'): self.weather.__getattr__(k.lstrip('_'))[self.time_step] 
                for k, v in vars(self.weather).items() if isinstance(v, np.ndarray)
            },
            **{
                k.lstrip('_'): self.pricing.__getattr__(k.lstrip('_'))[self.time_step] 
                for k, v in vars(self.pricing).items() if isinstance(v, np.ndarray)
            },
            **{
                k.lstrip('_'): self.carbon_intensity.__getattr__(k.lstrip('_'))[self.time_step] 
                for k, v in vars(self.carbon_intensity).items() if isinstance(v, np.ndarray)
            },
            'solar_generation':abs(self.solar_generation[self.time_step]),
            **{
                'cooling_storage_soc':self.cooling_storage.soc[self.time_step],
                'heating_storage_soc':self.heating_storage.soc[self.time_step],
                'dhw_storage_soc':self.dhw_storage.soc[self.time_step],
                'electrical_storage_soc':self.electrical_storage.soc[self.time_step],
            },
            'cooling_demand': self.__energy_from_cooling_device[self.time_step] + abs(min(self.cooling_storage.energy_balance[self.time_step], 0.0)),
            'heating_demand': self.__energy_from_heating_device[self.time_step] + abs(min(self.heating_storage.energy_balance[self.time_step], 0.0)),
            'dhw_demand': self.__energy_from_dhw_device[self.time_step] + abs(min(self.dhw_storage.energy_balance[self.time_step], 0.0)),
            'net_electricity_consumption': self.net_electricity_consumption[self.time_step],
            'cooling_electricity_consumption': self.cooling_electricity_consumption[self.time_step],
            'heating_electricity_consumption': self.heating_electricity_consumption[self.time_step],
            'dhw_electricity_consumption': self.dhw_electricity_consumption[self.time_step],
            'cooling_storage_electricity_consumption': self.cooling_storage_electricity_consumption[self.time_step],
            'heating_storage_electricity_consumption': self.heating_storage_electricity_consumption[self.time_step],
            'dhw_storage_electricity_consumption': self.dhw_storage_electricity_consumption[self.time_step],
            'electrical_storage_electricity_consumption': self.electrical_storage_electricity_consumption[self.time_step],
            'cooling_device_efficiency': self.cooling_device.get_cop(self.weather.outdoor_dry_bulb_temperature[self.time_step], heating=False),
            'heating_device_efficiency': self.heating_device.get_cop(
                self.weather.outdoor_dry_bulb_temperature[self.time_step], heating=True
                    ) if isinstance(self.heating_device, HeatPump) else self.heating_device.efficiency,
            'dhw_device_efficiency': self.dhw_device.get_cop(
                self.weather.outdoor_dry_bulb_temperature[self.time_step], heating=True
                    ) if isinstance(self.dhw_device, HeatPump) else self.dhw_device.efficiency,
            'indoor_dry_bulb_temperature_cooling_set_point': self.energy_simulation.indoor_dry_bulb_temperature_cooling_set_point[self.time_step],
            'indoor_dry_bulb_temperature_heating_set_point': self.energy_simulation.indoor_dry_bulb_temperature_heating_set_point[self.time_step],
            'indoor_dry_bulb_temperature_cooling_delta': self.energy_simulation.indoor_dry_bulb_temperature[self.time_step] - self.energy_simulation.indoor_dry_bulb_temperature_cooling_set_point[self.time_step],
            'indoor_dry_bulb_temperature_heating_delta': self.energy_simulation.indoor_dry_bulb_temperature[self.time_step] - self.energy_simulation.indoor_dry_bulb_temperature_heating_set_point[self.time_step],
            'comfort_band': self.energy_simulation.comfort_band[self.time_step],
            'occupant_count': self.energy_simulation.occupant_count[self.time_step],
            'power_outage': self.__power_outage_signal[self.time_step],
        }
    
    @staticmethod
    def get_periodic_observation_metadata() -> Mapping[str, int]:
        r"""Get periodic observation names and their minimum and maximum values for periodic/cyclic normalization.

        Returns
        -------
        periodic_observation_metadata: Mapping[str, int]
            Observation low and high limits.
        """

        return {
            'hour': range(1, 25),
            'day_type': range(1, 8),
            'month': range(1, 13)
        }

    def apply_actions(self,
                      cooling_or_heating_device_action: float = None,
                      cooling_device_action: float = None, heating_device_action: float = None,
                      cooling_storage_action: float = None, heating_storage_action: float = None,
                      dhw_storage_action: float = None, electrical_storage_action: float = None,
                      electric_vehicle_storage_actions: dict = None,
                      **kwargs
                      ):
        r"""Update cooling and heating demand for next timestep and charge/discharge storage devices.

        The order of action execution is dependent on polarity of the storage actions. If the electrical 
        storage is to be discharged, its action is executed first before all other actions. Likewise, if 
        the storage for an end-use is to be discharged, the storage action is executed before the control 
        action for the end-use electric device. Discharging the storage devices before fulfilling thermal 
        and non-shiftable loads ensures that the discharged energy is considered when allocating electricity 
        consumption to meet building loads. Likewise, meeting building loads before charging storage devices 
        ensures that comfort is met before attempting to shift loads.

        Parameters
        ----------
        cooling_or_heating_device_action : float, default: np.nan
            Fraction of `cooling_device` or `heating_device` `nominal_power` to make available. An action 
            < 0.0 is for the `cooling_device`, while an action > 0.0 is for the `heating_device`.
        cooling_device_action : float, default: np.nan
            Fraction of `cooling_device` `nominal_power` to make available for space cooling.
        heating_device_action : float, default: np.nan
            Fraction of `heating_device` `nominal_power` to make available for space heating.
        cooling_storage_action : float, default: 0.0
            Fraction of `cooling_storage` `capacity` to charge/discharge by.
        heating_storage_action : float, default: 0.0
            Fraction of `heating_storage` `capacity` to charge/discharge by.
        dhw_storage_action : float, default: 0.0
            Fraction of `dhw_storage` `capacity` to charge/discharge by.
        electrical_storage_action : float, default: 0.0
            Fraction of `electrical_storage` `capacity` to charge/discharge by.
        electric_vehicle_storage_actions : dict, default: None
            A dictionary where keys are charger IDs and values are the fraction of connected EV battery `capacity`
        **kwargs
        """

        # hvac devices
        if 'cooling_or_heating_device' in self.active_actions:
            assert 'cooling_device' not in self.active_actions and 'heating_device' not in self.active_actions, \
                'cooling_device and heating_device actions must be set to False when cooling_or_heating_device is True.' \
                    ' They will be implicitly set based on the polarity of cooling_or_heating_device.'
            cooling_device_action = abs(min(cooling_or_heating_device_action, 0.0))
            heating_device_action = abs(max(cooling_or_heating_device_action, 0.0))

        else:
            assert not ('cooling_device' in self.active_actions and 'heating_device' in self.active_actions), \
                'cooling_device and heating_device actions cannot both be set to True to avoid both actions having' \
                    ' values > 0.0 in the same time step. Use cooling_or_heating_device action instead to control' \
                        ' both cooling_device and heating_device in a building.'
            cooling_device_action = np.nan if 'cooling_device' not in self.active_actions else cooling_device_action
            heating_device_action = np.nan if 'heating_device' not in self.active_actions else heating_device_action

        # energy storage devices
        cooling_storage_action = 0.0 if 'cooling_storage' not in self.active_actions else cooling_storage_action
        heating_storage_action = 0.0 if 'heating_storage' not in self.active_actions else heating_storage_action
        dhw_storage_action = 0.0 if 'dhw_storage' not in self.active_actions else dhw_storage_action
        electrical_storage_action = 0.0 if 'electrical_storage' not in self.active_actions else electrical_storage_action

        # Initialize electric_vehicle_storage_actions dictionary if not provided
        if electric_vehicle_storage_actions is None:
            electric_vehicle_storage_actions = {charger.charger_id: 0.0 for charger in self.electric_vehicle_chargers}

        # set action priority
        actions = {
            'cooling_demand': (self.update_cooling_demand, (cooling_device_action,)),
            'heating_demand': (self.update_heating_demand, (heating_device_action,)),
            'cooling_device': (self.update_energy_from_cooling_device, ()),
            'cooling_storage': (self.update_cooling_storage, (cooling_storage_action,)),
            'heating_device': (self.update_energy_from_heating_device, ()),
            'heating_storage': (self.update_heating_storage, (heating_storage_action,)),
            'dhw_device': (self.update_energy_from_dhw_device, ()),
            'dhw_storage': (self.update_dhw_storage, (dhw_storage_action,)),
            'non_shiftable_load': (self.update_non_shiftable_load, ()),
            'electrical_storage': (self.update_electrical_storage, (electrical_storage_action,)),
        }

        priority_list = list(actions.keys())

        for charger in self.electric_vehicle_chargers:
            action_key = f'electric_vehicle_storage_{charger.charger_id}'
            electric_vehicle_action_value = electric_vehicle_storage_actions.get(charger.charger_id, 0.0)
            actions[action_key] = (charger.update_connected_electric_vehicle_soc, (electric_vehicle_action_value,))

        electric_vehicle_priority_list = [f'electric_vehicle_storage_{charger.charger_id}' for charger in self.electric_vehicle_chargers]
        priority_list = priority_list + electric_vehicle_priority_list  # the priority lists are merged

        if electrical_storage_action < 0.0:
            key = 'electrical_storage'
            priority_list.remove(key)
            priority_list = [key] + priority_list

        else:
            pass

        for key in ['cooling', 'heating', 'dhw']:
            storage = f'{key}_storage'
            device = f'{key}_device'

            if actions[storage][1][0] < 0.0:
                storage_ix = priority_list.index(storage)
                device_ix = priority_list.index(device)
                priority_list[storage_ix] = device
                priority_list[device_ix] = storage

            else:
                pass

        for k in priority_list:
            func, args = actions[k]

            try:
                func(*args)

            except NotImplementedError:
                pass

        self.update_variables()

    def update_cooling_demand(self, action: float):
        """Update space cooling demand for current time step."""

        raise NotImplementedError

    def update_energy_from_cooling_device(self):
        r"""Update cooling device electricity consumption and energy tranfer for current time step's cooling demand."""

        demand = self.cooling_demand[self.time_step]
        temperature = self.weather.outdoor_dry_bulb_temperature[self.time_step]
        storage_output = self.energy_from_cooling_storage[self.time_step]
        max_electric_power = self.downward_electrical_flexibility
        max_device_output = self.cooling_device.get_max_output_power(temperature, heating=False,
                                                                     max_electric_power=max_electric_power)
        self.___demand_limit_check('cooling', demand, max_device_output)
        device_output = min(demand - storage_output, max_device_output)
        self.__energy_from_cooling_device[self.time_step] = device_output
        electricity_consumption = self.cooling_device.get_input_power(device_output, temperature, heating=False)
        # LOGGER.debug(
        #     'timestep:', self.time_step, 'bldg:', self.name, 'demand:', demand, 'temperature:', temperature, 
        #     'storage_capacity:', self.cooling_storage.capacity, 'prev_soc:', self.cooling_storage.soc[self.time_step - 1], 
        #     'curr_soc:', self.cooling_storage.soc[self.time_step], 'storage_output:', storage_output, 
        #     'max_electric_power:', max_electric_power, 'max_device_output:', max_device_output, 'device_output:', 
        #     device_output, 'consumption:', electricity_consumption
        # )
        self.___electricity_consumption_polarity_check('cooling', device_output, electricity_consumption)
        self.cooling_device.update_electricity_consumption(max(0.0, electricity_consumption))

    def update_cooling_storage(self, action: float):
        r"""Charge/discharge `cooling_storage` for current time step.

        Parameters
        ----------
        action: float
            Fraction of `cooling_storage` `capacity` to charge/discharge by.
        """

        energy = action * self.cooling_storage.capacity
        temperature = self.weather.outdoor_dry_bulb_temperature[self.time_step]

        if energy > 0.0:
            max_electric_power = self.downward_electrical_flexibility
            max_output = self.cooling_device.get_max_output_power(temperature, heating=False,
                                                                  max_electric_power=max_electric_power)
            energy = min(max_output, energy)

        else:
            demand = self.cooling_demand[self.time_step]
            energy = max(-demand, energy)

        self.cooling_storage.charge(energy)
        charged_energy = max(self.cooling_storage.energy_balance[self.time_step], 0.0)
        electricity_consumption = self.cooling_device.get_input_power(charged_energy, temperature, heating=False)
        self.cooling_device.update_electricity_consumption(electricity_consumption)

    def update_heating_demand(self, action: float):
        """Update space heating demand for current time step."""

        raise NotImplementedError

    def update_energy_from_heating_device(self):
        r"""Update heating device electricity consumption and energy tranfer for current time step's heating demand."""

        demand = self.heating_demand[self.time_step]
        temperature = self.weather.outdoor_dry_bulb_temperature[self.time_step]
        storage_output = self.energy_from_heating_storage[self.time_step]
        max_electric_power = self.downward_electrical_flexibility
        max_device_output = self.heating_device.get_max_output_power(temperature, heating=True,
                                                                     max_electric_power=max_electric_power) \
            if isinstance(self.heating_device, HeatPump) else self.heating_device.get_max_output_power(
            max_electric_power=max_electric_power)
        self.___demand_limit_check('heating', demand, max_device_output)
        device_output = min(demand - storage_output, max_device_output)
        self.__energy_from_heating_device[self.time_step] = device_output
        electricity_consumption = self.heating_device.get_input_power(device_output, temperature, heating=True) \
            if isinstance(self.heating_device, HeatPump) else self.heating_device.get_input_power(device_output)
        self.___electricity_consumption_polarity_check('heating', device_output, electricity_consumption)
        self.heating_device.update_electricity_consumption(max(0.0, electricity_consumption))

    def update_heating_storage(self, action: float):
        r"""Charge/discharge `heating_storage` for current time step.

        Parameters
        ----------
        action: float
            Fraction of `heating_storage` `capacity` to charge/discharge by.
        """

        energy = action * self.heating_storage.capacity
        temperature = self.weather.outdoor_dry_bulb_temperature[self.time_step]

        if energy > 0.0:
            max_electric_power = self.downward_electrical_flexibility
            max_output = self.heating_device.get_max_output_power(temperature, heating=True,
                                                                  max_electric_power=max_electric_power) \
                if isinstance(self.heating_device, HeatPump) else self.heating_device.get_max_output_power(
                max_electric_power=max_electric_power)
            energy = min(max_output, energy)

        else:
            demand = self.heating_demand[self.time_step]
            energy = max(-demand, energy)

        self.heating_storage.charge(energy)
        charged_energy = max(self.heating_storage.energy_balance[self.time_step], 0.0)
        electricity_consumption = self.heating_device.get_input_power(charged_energy, temperature, heating=True) \
            if isinstance(self.heating_device, HeatPump) else self.heating_device.get_input_power(charged_energy)
        self.heating_device.update_electricity_consumption(electricity_consumption)

    def update_energy_from_dhw_device(self):
        r"""Update dhw device electricity consumption and energy tranfer for current time step's dhw demand."""

        demand = self.dhw_demand[self.time_step]
        temperature = self.weather.outdoor_dry_bulb_temperature[self.time_step]
        storage_output = self.energy_from_dhw_storage[self.time_step]
        max_electric_power = self.downward_electrical_flexibility
        max_device_output = self.dhw_device.get_max_output_power(temperature, heating=True,
                                                                 max_electric_power=max_electric_power) \
            if isinstance(self.dhw_device, HeatPump) else self.dhw_device.get_max_output_power(
            max_electric_power=max_electric_power)
        self.___demand_limit_check('dhw', demand, max_device_output)
        device_output = min(demand - storage_output, max_device_output)
        self.__energy_from_dhw_device[self.time_step] = device_output
        electricity_consumption = self.dhw_device.get_input_power(device_output, temperature, heating=True) \
            if isinstance(self.dhw_device, HeatPump) else self.dhw_device.get_input_power(device_output)
        self.___electricity_consumption_polarity_check('dhw', device_output, electricity_consumption)
        self.dhw_device.update_electricity_consumption(max(0.0, electricity_consumption))

    def update_dhw_storage(self, action: float):
        r"""Charge/discharge `dhw_storage` for current time step.

        Parameters
        ----------
        action: float
            Fraction of `dhw_storage` `capacity` to charge/discharge by.
        """

        energy = action * self.dhw_storage.capacity
        temperature = self.weather.outdoor_dry_bulb_temperature[self.time_step]

        if energy > 0.0:
            max_electric_power = self.downward_electrical_flexibility
            max_output = self.dhw_device.get_max_output_power(temperature, heating=True,
                                                              max_electric_power=max_electric_power) \
                if isinstance(self.dhw_device, HeatPump) else self.dhw_device.get_max_output_power(
                max_electric_power=max_electric_power)
            energy = min(max_output, energy)

        else:
            demand = self.dhw_demand[self.time_step]
            energy = max(-demand, energy)

        self.dhw_storage.charge(energy)
        charged_energy = max(self.dhw_storage.energy_balance[self.time_step], 0.0)
        electricity_consumption = self.dhw_device.get_input_power(charged_energy, temperature, heating=True) \
            if isinstance(self.dhw_device, HeatPump) else self.dhw_device.get_input_power(charged_energy)
        self.dhw_device.update_electricity_consumption(electricity_consumption)

    def update_non_shiftable_load(self):
        r"""Update non shiftable loads electricity consumption for current time step non shiftable load."""

        demand = min(self.non_shiftable_load[self.time_step], self.downward_electrical_flexibility)
        self.__energy_to_non_shiftable_load[self.time_step] = demand
        self.non_shiftable_load_device.update_electricity_consumption(demand)
        # ToDo might need to add chargers here

    def update_electrical_storage(self, action: float):
        r"""Charge/discharge `electrical_storage` for current time step.

        Parameters
        ----------
        action : float
            Fraction of `electrical_storage` `capacity` to charge/discharge by.
        """

        energy = min(action * self.electrical_storage.capacity, self.downward_electrical_flexibility)
        self.electrical_storage.charge(energy)

    def ___demand_limit_check(self, end_use: str, demand: float, max_device_output: float):
        message = f'timestep: {self.time_step}, building: {self.name}, outage: {self.power_outage}, demand: {demand},' \
                  f'output: {max_device_output}, difference: {demand - max_device_output}, check: {demand <= max_device_output},'
        assert self.power_outage or demand <= max_device_output or abs(demand - max_device_output) < TOLERANCE, \
            f'demand is greater than {end_use}_device max output | {message}'

    def ___electricity_consumption_polarity_check(self, end_use: str, device_output: float,
                                                  electricity_consumption: float):
        message = f'timestep: {self.time_step}, building: {self.name}, device_output: {device_output}, electricity_consumption: {electricity_consumption}'
        assert electricity_consumption >= 0.0 or abs(electricity_consumption) < TOLERANCE, \
            f'negative electricity consumption for {end_use} demand | {message}'

    def estimate_observation_space(self, include_all: bool = None, normalize: bool = None) -> spaces.Box:
        r"""Get estimate of observation spaces.

        Parameters
        ----------
        include_all: bool, default: False,
            Whether to estimate for all observations as listed in `observation_metadata` or only those that are active.
        normalize : bool, default: False
            Whether to apply min-max normalization bounded between [0, 1].

        Returns
        -------
        observation_space : spaces.Box
            Observation low and high limits.
        """

        normalize = False if normalize is None else normalize
        normalized_observation_space_limits = self.estimate_observation_space_limits(include_all=include_all,
                                                                                     periodic_normalization=True)
        unnormalized_observation_space_limits = self.estimate_observation_space_limits(include_all=include_all,
                                                                                       periodic_normalization=False)

        if normalize:
            low_limit, high_limit = normalized_observation_space_limits
            low_limit = [0.0] * len(low_limit)
            high_limit = [1.0] * len(high_limit)
        else:
            low_limit, high_limit = unnormalized_observation_space_limits
            low_limit = list(low_limit.values())
            high_limit = list(high_limit.values())

        return spaces.Box(low=np.array(low_limit, dtype='float32'), high=np.array(high_limit, dtype='float32'),
                          dtype='float32')

    def estimate_observation_space_limits(self, include_all: bool = None, periodic_normalization: bool = None) -> Tuple[
        Mapping[str, float], Mapping[str, float]]:
        r"""Get estimate of observation space limits.

        Find minimum and maximum possible values of all the observations, which can then be used by the RL agent to scale the observations 
        and train any function approximators more effectively.

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
        scaling this observation-variable using these bounds may result in normalized values above 1 or below 0. It is also
        assumed that devices and storage systems have been sized.
        """

        include_all = False if include_all is None else include_all
        internal_limit_observations = [
            'net_electricity_consumption_without_storage',
            'net_electricity_consumption_without_storage_and_partial_load',
            'net_electricity_consumption_without_storage_and_partial_load_and_pv'
        ]
        observation_names = list(
            self.observation_metadata.keys()) + internal_limit_observations if include_all else self.active_observations
        periodic_normalization = False if periodic_normalization is None else periodic_normalization
        periodic_observations = self.get_periodic_observation_metadata()
        low_limit, high_limit = {}, {}
        data = self._get_observation_space_limits_data()

        for key in observation_names:
            if key == 'net_electricity_consumption':
                # assumes devices and storages have been sized
                low_limits = data['non_shiftable_load'] - (
                        + self.electrical_storage.nominal_power
                        + data['solar_generation']
                )
                high_limits = data['non_shiftable_load'] \
                              + self.cooling_device.nominal_power \
                              + self.heating_device.nominal_power \
                              + self.dhw_device.nominal_power \
                              + self.electrical_storage.nominal_power \
                              - data['solar_generation']
                low_limit[key] = min(low_limits.min(), 0.0)
                high_limit[key] = high_limits.max()

            elif key == 'net_electricity_consumption_without_storage':
                low_limit[key] = min(low_limit['net_electricity_consumption'] + self.electrical_storage.nominal_power,
                                     0.0)
                high_limit[key] = high_limit['net_electricity_consumption'] - self.electrical_storage.nominal_power

            elif key == 'net_electricity_consumption_without_storage_and_partial_load':
                low_limit[key] = low_limit['net_electricity_consumption_without_storage']
                high_limit[key] = high_limit['net_electricity_consumption_without_storage']

            elif key == 'net_electricity_consumption_without_storage_and_partial_load_and_pv':
                low_limit[key] = 0.0
                high_limits = data['non_shiftable_load'] \
                              + self.cooling_device.nominal_power \
                              + self.heating_device.nominal_power \
                              + self.dhw_device.nominal_power
                high_limit[key] = high_limits.max()

            elif key in ['cooling_storage_soc', 'heating_storage_soc', 'dhw_storage_soc', 'electrical_storage_soc',
                         'electric_vehicle_charger_state', 'electric_vehicle_soc']:
                low_limit[key] = 0.0
                high_limit[key] = 1.0

            elif key in ['electric_vehicle_required_soc_departure', 'electric_vehicle_estimated_soc_arrival']:
                low_limit[key] = -0.1
                high_limit[key] = 1.0

            elif key in ['cooling_device_efficiency']:
                cop = self.cooling_device.get_cop(data['outdoor_dry_bulb_temperature'], heating=False)
                low_limit[key] = min(cop)
                high_limit[key] = max(cop)

            elif key in ['heating_device_efficiency']:
                if isinstance(self.heating_device, HeatPump):
                    cop = self.heating_device.get_cop(data['outdoor_dry_bulb_temperature'], heating=True)
                    low_limit[key] = min(cop)
                    high_limit[key] = max(cop)
                else:
                    low_limit[key] = self.heating_device.efficiency
                    high_limit[key] = self.heating_device.efficiency

            elif 'charger' in key:
                if self.electric_vehicle_chargers is not None:
                    for charger in self.electric_vehicle_chargers:
                        if key == f'charger_{charger.charger_id}_connected_state' or key == f'charger_{charger.charger_id}_incoming_state':
                            low_limit[key] = 0
                            high_limit[key] = 3
                        elif 'electric_vehicle_charger_state' in key:
                            low_limit[key] = 0
                            high_limit[key] = 1
                        elif any(value in key for value in
                                 ['electric_vehicle_departure_time', 'electric_vehicle_estimated_arrival_time']):
                            low_limit[key] = -0.1
                            high_limit[key] = 24
                        elif any(value in key for value in
                                 ['electric_vehicle_required_soc_departure', 'electric_vehicle_estimated_soc_arrival',
                                  'electric_vehicle_soc']):
                            low_limit[key] = -0.1
                            high_limit[key] = 1.0
            elif key in ['dhw_device_efficiency']:
                if isinstance(self.dhw_device, HeatPump):
                    cop = self.dhw_device.get_cop(data['outdoor_dry_bulb_temperature'], heating=True)
                    low_limit[key] = min(cop)
                    high_limit[key] = max(cop)
                else:
                    low_limit[key] = self.dhw_device.efficiency
                    high_limit[key] = self.dhw_device.efficiency

            elif key == 'indoor_dry_bulb_temperature':
                low_limit[key] = data['indoor_dry_bulb_temperature'].min() - self.maximum_temperature_delta
                high_limit[key] = data['indoor_dry_bulb_temperature'].max() + self.maximum_temperature_delta

            elif key in ['indoor_dry_bulb_temperature_cooling_delta', 'indoor_dry_bulb_temperature_heating_delta']:
                low_limit[key] = -self.maximum_temperature_delta
                high_limit[key] = self.maximum_temperature_delta

            elif key == 'comfort_band':
                low_limit[key] = 0
                high_limit[key] = max(data[key])

            elif key in ['cooling_demand', 'heating_demand', 'dhw_demand']:
                low_limit[key] = 0.0
                max_demand = data[key].max()
                high_limit[key] = max_demand * self.demand_observation_limit_factor

            elif key == 'cooling_electricity_consumption':
                low_limit[key] = 0.0
                high_limit[key] = self.cooling_device.nominal_power

            elif key == 'heating_electricity_consumption':
                low_limit[key] = 0.0
                high_limit[key] = self.heating_device.nominal_power

            elif key == 'dhw_electricity_consumption':
                low_limit[key] = 0.0
                high_limit[key] = self.dhw_device.nominal_power

            elif key == 'cooling_storage_electricity_consumption':
                demand = self.energy_simulation.__getattr__(
                    f'cooling_demand',
                    start_time_step=self.episode_tracker.simulation_start_time_step,
                    end_time_step=self.episode_tracker.simulation_end_time_step
                )
                electricity_consumption = self.cooling_device.get_input_power(demand,
                                                                              data['outdoor_dry_bulb_temperature'],
                                                                              False)
                low_limit[key] = -max(electricity_consumption)
                high_limit[key] = self.cooling_device.nominal_power

            elif key == 'heating_storage_electricity_consumption':
                demand = self.energy_simulation.__getattr__(
                    f'heating_demand',
                    start_time_step=self.episode_tracker.simulation_start_time_step,
                    end_time_step=self.episode_tracker.simulation_end_time_step
                )
                electricity_consumption = self.heating_device.get_input_power(demand,
                                                                              data['outdoor_dry_bulb_temperature'],
                                                                              True) \
                    if isinstance(self.heating_device, HeatPump) else self.heating_device.get_input_power(demand)
                low_limit[key] = -max(electricity_consumption)
                high_limit[key] = self.heating_device.nominal_power

            elif key == 'dhw_storage_electricity_consumption':
                demand = self.energy_simulation.__getattr__(
                    f'dhw_demand',
                    start_time_step=self.episode_tracker.simulation_start_time_step,
                    end_time_step=self.episode_tracker.simulation_end_time_step
                )
                electricity_consumption = self.dhw_device.get_input_power(demand, data['outdoor_dry_bulb_temperature'],
                                                                          True) \
                    if isinstance(self.dhw_device, HeatPump) else self.dhw_device.get_input_power(demand)
                low_limit[key] = -max(electricity_consumption)
                high_limit[key] = self.dhw_device.nominal_power

            elif key == 'electrical_storage_electricity_consumption':
                low_limit[key] = -self.electrical_storage.nominal_power
                high_limit[key] = self.electrical_storage.nominal_power

            elif key == 'power_outage':
                low_limit[key] = 0.0
                high_limit[key] = 1.0

            elif periodic_normalization and key in periodic_observations:
                pn = PeriodicNormalization(max(periodic_observations[key]))
                x_sin, x_cos = pn * np.array(list(periodic_observations[key]))
                low_limit[f'{key}_cos'], high_limit[f'{key}_cos'] = min(x_cos), max(x_cos)
                low_limit[f'{key}_sin'], high_limit[f'{key}_sin'] = min(x_sin), max(x_sin)

            elif key == 'occupant_interaction_indoor_dry_bulb_temperature_set_point_delta':
                # will get set in the overriding  LogisticRegressionOccupantInteractionBuilding._get_observation_space_limits_data
                pass

            else:
                low_limit[key] = min(data[key])
                high_limit[key] = max(data[key])

        low_limit = {k: v - self.observation_space_limit_delta for k, v in low_limit.items()}
        high_limit = {k: v + self.observation_space_limit_delta for k, v in high_limit.items()}

        return low_limit, high_limit
    
    def _get_observation_space_limits_data(self) -> Mapping[str, List[Union[float, int]]]:
        # Use entire dataset length for space limit estimation
        return {
            **{k.lstrip('_'): self.energy_simulation.__getattr__(
                k.lstrip('_'), 
                start_time_step=self.episode_tracker.simulation_start_time_step, 
                end_time_step=self.episode_tracker.simulation_end_time_step
            ) for k in vars(self.energy_simulation)},
            'solar_generation':np.array(self.pv.get_generation(self.energy_simulation.__getattr__(
                'solar_generation', 
                start_time_step=self.episode_tracker.simulation_start_time_step, 
                end_time_step=self.episode_tracker.simulation_end_time_step
            ))),
            **{k.lstrip('_'): self.weather.__getattr__(
                k.lstrip('_'), 
                start_time_step=self.episode_tracker.simulation_start_time_step, 
                end_time_step=self.episode_tracker.simulation_end_time_step
            ) for k in vars(self.weather)},
            **{k.lstrip('_'): self.carbon_intensity.__getattr__(
                k.lstrip('_'), 
                start_time_step=self.episode_tracker.simulation_start_time_step, 
                end_time_step=self.episode_tracker.simulation_end_time_step
            ) for k in vars(self.carbon_intensity)},
            **{k.lstrip('_'): self.pricing.__getattr__(
                k.lstrip('_'), 
                start_time_step=self.episode_tracker.simulation_start_time_step, 
                end_time_step=self.episode_tracker.simulation_end_time_step
            ) for k in vars(self.pricing)},
        }
    
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
            if key == 'cooling_or_heating_device':
                if self.cooling_device.nominal_power > ZERO_DIVISION_PLACEHOLDER:
                    low_limit.append(-1.0)

                else:
                    low_limit.append(0.0)

                if self.heating_device.nominal_power > ZERO_DIVISION_PLACEHOLDER:
                    high_limit.append(1.0)

                else:
                    high_limit.append(0.0)

            elif key in ['cooling_device', 'heating_device']:
                low_limit.append(0.0)
                high_limit.append(1.0)

            elif 'electric_vehicle_storage' in key:
                if self.electric_vehicle_chargers is not None:
                    for c in self.electric_vehicle_chargers:
                        if key == f'electric_vehicle_storage_{c.charger_id}':
                            discharging_limit = 0 if c.max_discharging_power == 0 else -1

                            high_limit.append(1.0)  # For discharging limit
                            low_limit.append(discharging_limit)  # For charging limit

            elif 'storage' in key:
                if key == 'electrical_storage':
                    limit = self.electrical_storage.nominal_power / max(self.electrical_storage.capacity,
                                                                        ZERO_DIVISION_PLACEHOLDER)

                else:
                    if key == 'cooling_storage':
                        capacity = self.cooling_storage.capacity
                        power = self.cooling_device.nominal_power

                    elif key == 'heating_storage':
                        capacity = self.heating_storage.capacity
                        power = self.heating_device.nominal_power

                    elif key == 'dhw_storage':
                        capacity = self.dhw_storage.capacity
                        power = self.dhw_device.nominal_power

                    else:
                        raise Exception(f'Unknown action: {key}')

                    limit = power / max(capacity, ZERO_DIVISION_PLACEHOLDER)

                limit = min(limit, 1.0)
                low_limit.append(-limit)
                high_limit.append(limit)

            else:
                if key == 'cooling_storage':
                    capacity = self.cooling_storage.capacity
                    cooling_demand = self.energy_simulation.__getattr__(
                        'cooling_demand',
                        start_time_step=self.episode_tracker.simulation_start_time_step,
                        end_time_step=self.episode_tracker.simulation_end_time_step
                    )
                    maximum_demand = cooling_demand.max()

                elif key == 'heating_storage':
                    capacity = self.heating_storage.capacity
                    heating_demand = self.energy_simulation.__getattr__(
                        'heating_demand',
                        start_time_step=self.episode_tracker.simulation_start_time_step,
                        end_time_step=self.episode_tracker.simulation_end_time_step
                    )
                    maximum_demand = heating_demand.max()

                elif key == 'dhw_storage':
                    capacity = self.dhw_storage.capacity
                    dhw_demand = self.energy_simulation.__getattr__(
                        'dhw_demand',
                        start_time_step=self.episode_tracker.simulation_start_time_step,
                        end_time_step=self.episode_tracker.simulation_end_time_step
                    )
                    maximum_demand = dhw_demand.max()

                else:
                    raise Exception(f'Unknown action: {key}')

                maximum_demand_ratio = maximum_demand / max(capacity, ZERO_DIVISION_PLACEHOLDER)

                try:
                    low_limit.append(max(-maximum_demand_ratio, -1.0))
                    high_limit.append(min(maximum_demand_ratio, 1.0))
                except ZeroDivisionError:
                    low_limit.append(-1.0)
                    high_limit.append(1.0)

        return spaces.Box(low=np.array(low_limit, dtype='float32'), high=np.array(high_limit, dtype='float32'),
                          dtype='float32')

    def autosize_cooling_device(self, **kwargs):
        """Autosize `cooling_device` `nominal_power` to minimum power needed to always meet `cooling_demand`.
        
        Other Parameters
        ----------------
        **kwargs : dict
            Other keyword arguments parsed to `cooling_device` `autosize` function.
        """

        demand = self.energy_simulation.__getattr__(
            'cooling_demand',
            start_time_step=self.episode_tracker.simulation_start_time_step,
            end_time_step=self.episode_tracker.simulation_end_time_step
        )
        temperature = self.weather.__getattr__(
            'outdoor_dry_bulb_temperature',
            start_time_step=self.episode_tracker.simulation_start_time_step,
            end_time_step=self.episode_tracker.simulation_end_time_step
        )
        self.cooling_device.nominal_power = self.cooling_device.autosize(temperature, cooling_demand=demand, **kwargs)

    def autosize_heating_device(self, **kwargs):
        """Autosize `heating_device` `nominal_power` to minimum power needed to always meet `heating_demand`.
        
        Other Parameters
        ----------------
        **kwargs : dict
            Other keyword arguments parsed to `heating_device` `autosize` function.
        """

        demand = self.energy_simulation.__getattr__(
            'heating_demand',
            start_time_step=self.episode_tracker.simulation_start_time_step,
            end_time_step=self.episode_tracker.simulation_end_time_step
        )
        temperature = self.weather.__getattr__(
            'outdoor_dry_bulb_temperature',
            start_time_step=self.episode_tracker.simulation_start_time_step,
            end_time_step=self.episode_tracker.simulation_end_time_step
        )

        if isinstance(self.heating_device, HeatPump):
            self.heating_device.nominal_power = self.heating_device.autosize(temperature, heating_demand=demand,
                                                                             **kwargs)

        else:
            self.heating_device.nominal_power = self.heating_device.autosize(demand, **kwargs)

    def autosize_dhw_device(self, **kwargs):
        """Autosize `dhw_device` `nominal_power` to minimum power needed to always meet `dhw_demand`.
        
        Other Parameters
        ----------------
        **kwargs : dict
            Other keyword arguments parsed to `dhw_device` `autosize` function.
        """

        demand = self.energy_simulation.__getattr__(
            'dhw_demand',
            start_time_step=self.episode_tracker.simulation_start_time_step,
            end_time_step=self.episode_tracker.simulation_end_time_step
        )
        temperature = self.weather.__getattr__(
            'outdoor_dry_bulb_temperature',
            start_time_step=self.episode_tracker.simulation_start_time_step,
            end_time_step=self.episode_tracker.simulation_end_time_step
        )

        if isinstance(self.dhw_device, HeatPump):
            self.dhw_device.nominal_power = self.dhw_device.autosize(temperature, heating_demand=demand, **kwargs)

        else:
            self.dhw_device.nominal_power = self.dhw_device.autosize(demand, **kwargs)

    def autosize_cooling_storage(self, **kwargs):
        """Autosize `cooling_storage` `capacity` to minimum capacity needed to always meet `cooling_demand`.
        
        Other Parameters
        ----------------
        **kwargs : dict
            Other keyword arguments parsed to `cooling_storage` `autosize` function.
        """

        demand = self.energy_simulation.__getattr__(
            'cooling_demand',
            start_time_step=self.episode_tracker.simulation_start_time_step,
            end_time_step=self.episode_tracker.simulation_end_time_step
        )
        self.cooling_storage.capacity = self.cooling_storage.autosize(demand, **kwargs)

    def autosize_heating_storage(self, **kwargs):
        """Autosize `heating_storage` `capacity` to minimum capacity needed to always meet `heating_demand`.
        
        Other Parameters
        ----------------
        **kwargs : dict
            Other keyword arguments parsed to `heating_storage` `autosize` function.
        """

        demand = self.energy_simulation.__getattr__(
            'heating_demand',
            start_time_step=self.episode_tracker.simulation_start_time_step,
            end_time_step=self.episode_tracker.simulation_end_time_step
        )
        self.heating_storage.capacity = self.heating_storage.autosize(demand, **kwargs)

    def autosize_dhw_storage(self, **kwargs):
        """Autosize `dhw_storage` `capacity` to minimum capacity needed to always meet `dhw_demand`.
        
        Other Parameters
        ----------------
        **kwargs : dict
            Other keyword arguments parsed to `dhw_storage` `autosize` function.
        """

        demand = self.energy_simulation.__getattr__(
            'dhw_demand',
            start_time_step=self.episode_tracker.simulation_start_time_step,
            end_time_step=self.episode_tracker.simulation_end_time_step
        )
        self.dhw_storage.capacity = self.dhw_storage.autosize(demand, **kwargs)

    def autosize_electrical_storage(self, **kwargs):
        """Autosize `electrical_storage` `capacity`, `nominal_power`, `depth_of_discharge`, `efficiency`, 
        `loss_coefficient`, and `capacity_loss_coefficient` to meet an estimated average peak demand.
        
        Other Parameters
        ----------------
        **kwargs : dict
            Other keyword arguments parsed to `electrical_storage` `autosize` function.
        """

        demand = pd.DataFrame(self._estimate_baseline_electricity_consumption(), columns=['value'])
        demand['day'] = (demand.index / (self.seconds_per_time_step * 24 / self.seconds_per_time_step)).astype(int)
        demand = demand.groupby('day')['value'].max().mean()

        self.electrical_storage.capacity, \
            self.electrical_storage.nominal_power, \
            self.electrical_storage.depth_of_discharge, \
            self.electrical_storage.efficiency, \
            self.electrical_storage.loss_coefficient, \
            self.electrical_storage.capacity_loss_coefficient = self.electrical_storage.autosize(
            demand, **kwargs
        )

    def autosize_pv(self, **kwargs):
        """Autosize `pv` `nominal_power` and set `energy_simulation.solar_generation` using sampled PV data from
        LBNL's Tracking The Sun dataset.
        
        Other Parameters
        ----------------
        **kwargs : dict
            Other keyword arguments parsed to `electrical_storage` `autosize` function.
        """

        demand = pd.DataFrame(self._estimate_baseline_electricity_consumption(), columns=['value'])
        demand['year'] = (demand.index / (self.seconds_per_time_step * 24 * 365 / self.seconds_per_time_step)).astype(
            int)
        demand = demand.groupby('year')['value'].sum().mean()
        epw_filepath = kwargs.pop('epw_filepath')
        self.pv.nominal_power, solar_generation = self.pv.autosize(demand, epw_filepath, **kwargs)
        self.energy_simulation.__setattr__('solar_generation', np.array(solar_generation, dtype='float32'))

    def _estimate_baseline_electricity_consumption(self) -> np.ndarray:
        """Returns estimated baseline electricity consumption time series for entire simulation period.
        
        The estimate is the sum of estimated cooling, heating, domestic hot water and non-shiftable 
        load consumption without storage and self-generation flexibility.

        Returns
        -------
        baseline_electricity_consumption: np.ndarray
            Estimate time series for simulation period which may be equal to or longer than the current
            episode's number of time steps.
        """

        cooling_demand = self.energy_simulation.__getattr__(
            'cooling_demand',
            start_time_step=self.episode_tracker.simulation_start_time_step,
            end_time_step=self.episode_tracker.simulation_end_time_step
        )
        heating_demand = self.energy_simulation.__getattr__(
            'heating_demand',
            start_time_step=self.episode_tracker.simulation_start_time_step,
            end_time_step=self.episode_tracker.simulation_end_time_step
        )
        dhw_demand = self.energy_simulation.__getattr__(
            'dhw_demand',
            start_time_step=self.episode_tracker.simulation_start_time_step,
            end_time_step=self.episode_tracker.simulation_end_time_step
        )
        non_shiftable_electricity_consumption = self.energy_simulation.__getattr__(
            'non_shiftable_load',
            start_time_step=self.episode_tracker.simulation_start_time_step,
            end_time_step=self.episode_tracker.simulation_end_time_step
        )
        outdoor_dry_bulb_temperature = self.weather.__getattr__(
            'outdoor_dry_bulb_temperature',
            start_time_step=self.episode_tracker.simulation_start_time_step,
            end_time_step=self.episode_tracker.simulation_end_time_step
        )
        cooling_electricity_consumption = self.cooling_device.get_input_power(cooling_demand,
                                                                              outdoor_dry_bulb_temperature,
                                                                              heating=False)

        if isinstance(self.heating_device, HeatPump):
            heating_electricity_consumption = self.heating_device.get_input_power(heating_demand,
                                                                                  outdoor_dry_bulb_temperature,
                                                                                  heating=True)

        else:
            heating_electricity_consumption = self.heating_device.get_input_power(heating_demand)

        if isinstance(self.dhw_device, HeatPump):
            dhw_electricity_consumption = self.dhw_device.get_input_power(dhw_demand, outdoor_dry_bulb_temperature,
                                                                          heating=True)

        else:
            dhw_electricity_consumption = self.dhw_device.get_input_power(dhw_demand)

        electricity_consumption = cooling_electricity_consumption + heating_electricity_consumption + dhw_electricity_consumption + non_shiftable_electricity_consumption

        return electricity_consumption

    def next_time_step(self):
        r"""Advance all energy storage and electric devices and, PV to next `time_step`."""

        self.cooling_device.next_time_step()
        self.heating_device.next_time_step()
        self.dhw_device.next_time_step()
        self.non_shiftable_load_device.next_time_step()
        self.cooling_storage.next_time_step()
        self.heating_storage.next_time_step()
        self.dhw_storage.next_time_step()
        self.electrical_storage.next_time_step()
        self.pv.next_time_step()

        if self.electric_vehicle_chargers is not None:
            for c in self.electric_vehicle_chargers:
                pass
                c.next_time_step()
        super().next_time_step()

    def reset(self):
        r"""Reset `Building` to initial state."""

        # object reset
        super().reset()
        self.cooling_storage.reset()
        self.heating_storage.reset()
        self.dhw_storage.reset()
        self.electrical_storage.reset()
        self.cooling_device.reset()
        self.heating_device.reset()
        self.dhw_device.reset()
        self.non_shiftable_load_device.reset()
        self.pv.reset()

        if self.electric_vehicle_chargers is not None:
            for c in self.electric_vehicle_chargers:
                c.reset()
        else:
            pass

        # variable reset
        self.reset_dynamic_variables()
        self.reset_data_sets()
        self.__solar_generation = self.pv.get_generation(self.energy_simulation.solar_generation) * -1
        self.__energy_from_cooling_device = self.energy_simulation.cooling_demand.copy()
        self.__energy_from_heating_device = self.energy_simulation.heating_demand.copy()
        self.__energy_from_dhw_device = self.energy_simulation.dhw_demand.copy()
        self.__energy_to_non_shiftable_load = self.energy_simulation.non_shiftable_load.copy()
        self.__net_electricity_consumption = np.zeros(self.episode_tracker.episode_time_steps, dtype='float32')
        self.__net_electricity_consumption_emission = np.zeros(self.episode_tracker.episode_time_steps, dtype='float32')
        self.__net_electricity_consumption_cost = np.zeros(self.episode_tracker.episode_time_steps, dtype='float32')
        self.__power_outage_signal = self.reset_power_outage_signal()
        self.__chargers_electricity_consumption = np.zeros(self.episode_tracker.episode_time_steps, dtype='float32')

        self.update_variables()

    def reset_power_outage_signal(self) -> np.ndarray:
        """Resets power outage signal time series.
        
        Resets to zeros if `simulate_power_outage` is `False` otherwise, resets to a stochastic time series 
        if `stochastic_power_outage` is `True` or the  time series defined in `energy_simulation.power_outage`.

        Returns
        -------
        power_outage_signal: np.ndarray
            Power outage signal time series.
        """

        power_outage_signal = np.zeros(self.episode_tracker.episode_time_steps, dtype='float32')

        if self.simulate_power_outage:
            if self.stochastic_power_outage:
                power_outage_signal = self.stochastic_power_outage_model.get_signals(
                    self.episode_tracker.episode_time_steps,
                    seconds_per_time_step=self.seconds_per_time_step,
                    weather=self.weather
                )

            else:
                power_outage_signal = self.energy_simulation.power_outage.copy()

        else:
            pass

        return power_outage_signal

    def reset_dynamic_variables(self):
        """Resets data file variables that change during control to their initial values."""

        pass

    def reset_data_sets(self):
        """Resets time series data `start_time_step` and `end_time_step` with respect to current episode's time step settings."""

        start_time_step = self.episode_tracker.episode_start_time_step
        end_time_step = self.episode_tracker.episode_end_time_step
        self.energy_simulation.start_time_step = start_time_step
        self.weather.start_time_step = start_time_step
        self.pricing.start_time_step = start_time_step
        self.carbon_intensity.start_time_step = start_time_step
        self.energy_simulation.end_time_step = end_time_step
        self.weather.end_time_step = end_time_step
        self.pricing.end_time_step = end_time_step
        self.carbon_intensity.end_time_step = end_time_step

    def update_variables(self):
        """Update cooling, heating, dhw and net electricity consumption as well as net electricity consumption cost and carbon emissions."""

        if self.time_step == 0:
            temperature = self.weather.outdoor_dry_bulb_temperature[self.time_step]

            # cooling electricity consumption
            cooling_demand = self.__energy_from_cooling_device[self.time_step] + self.cooling_storage.energy_balance[
                self.time_step]
            cooling_electricity_consumption = self.cooling_device.get_input_power(cooling_demand, temperature,
                                                                                  heating=False)
            self.cooling_device.update_electricity_consumption(cooling_electricity_consumption)

            # heating electricity consumption
            heating_demand = self.__energy_from_heating_device[self.time_step] + self.heating_storage.energy_balance[
                self.time_step]

            if isinstance(self.heating_device, HeatPump):
                heating_electricity_consumption = self.heating_device.get_input_power(heating_demand, temperature,
                                                                                      heating=True)
            else:
                heating_electricity_consumption = self.dhw_device.get_input_power(heating_demand)

            self.heating_device.update_electricity_consumption(heating_electricity_consumption)

            # dhw electricity consumption
            dhw_demand = self.__energy_from_dhw_device[self.time_step] + self.dhw_storage.energy_balance[self.time_step]

            if isinstance(self.dhw_device, HeatPump):
                dhw_electricity_consumption = self.dhw_device.get_input_power(dhw_demand, temperature, heating=True)
            else:
                dhw_electricity_consumption = self.dhw_device.get_input_power(dhw_demand)

            self.dhw_device.update_electricity_consumption(dhw_electricity_consumption)

            # non shiftable load electricity consumption
            non_shiftable_load_electricity_consumption = self.__energy_to_non_shiftable_load[self.time_step]
            self.non_shiftable_load_device.update_electricity_consumption(non_shiftable_load_electricity_consumption)

            # electrical storage
            electrical_storage_electricity_consumption = self.electrical_storage.energy_balance[self.time_step]
            self.electrical_storage.update_electricity_consumption(electrical_storage_electricity_consumption,
                                                                   enforce_polarity=False)

        else:
            pass

        building_chargers_total_electricity_consumption = 0

        if self.electric_vehicle_chargers is not None:

            for c in self.electric_vehicle_chargers:
                building_chargers_total_electricity_consumption = building_chargers_total_electricity_consumption + c.electricity_consumption[
                    self.time_step - 1]
        else:
            pass

        self.__chargers_electricity_consumption[self.time_step] = building_chargers_total_electricity_consumption

        # net electricity consumption
        net_electricity_consumption = 0.0

        if not self.power_outage:
            net_electricity_consumption = self.cooling_device.electricity_consumption[self.time_step] \
                                          + self.heating_device.electricity_consumption[self.time_step] \
                                          + self.dhw_device.electricity_consumption[self.time_step] \
                                          + self.non_shiftable_load_device.electricity_consumption[self.time_step] \
                                          + self.electrical_storage.electricity_consumption[self.time_step] \
                                          + self.solar_generation[self.time_step] \
                                          + self.__chargers_electricity_consumption[self.time_step]
        else:
            pass

        self.__net_electricity_consumption[self.time_step] = net_electricity_consumption

        # net electriciy consumption cost
        self.__net_electricity_consumption_cost[self.time_step] = net_electricity_consumption * \
                                                                  self.pricing.electricity_pricing[self.time_step]

        # net electriciy consumption emission
        self.__net_electricity_consumption_emission[self.time_step] = max(0.0, net_electricity_consumption *
                                                                          self.carbon_intensity.carbon_intensity[
                                                                              self.time_step])


class DynamicsBuilding(Building):  # ToDo
    r"""Base class for temperature dynamic building.

    Parameters
    ----------
    *args: Any
        Positional arguments in :py:class:`citylearn.building.Building`.
    dynamics: Dynamics
        Indoor dry-bulb temperature dynamics model.
    ignore_dynamics: bool, default: False
        Wether to simulate temperature dynamics at any time step.

    Other Parameters
    ----------------
    **kwargs : Any
        Other keyword arguments used to initialize :py:class:`citylearn.building.Building` super class.
    """

    def __init__(self, *args: Any, dynamics: Dynamics, ignore_dynamics: bool = None, **kwargs: Any):
        """Intialize `DynamicsBuilding`"""

        self.dynamics = dynamics
        self.ignore_dynamics = False if ignore_dynamics is None else ignore_dynamics
        super().__init__(*args, **kwargs)

    @property
    def simulate_dynamics(self) -> bool:
        """Whether to predict indoor dry-bulb temperature at current `time_step`."""

        return not self.ignore_dynamics

    @property
    def net_electricity_consumption_emission_without_storage_and_partial_load_and_pv(self) -> np.ndarray:
        """Carbon dioxide emmission from `net_electricity_consumption_without_storage_and_partial_load_pv` time series, in [kg_co2]."""

        return (
                self.carbon_intensity.carbon_intensity[
                0:self.time_step + 1] * self.net_electricity_consumption_without_storage_and_partial_load_and_pv
        ).clip(min=0)

    @property
    def net_electricity_consumption_cost_without_storage_and_partial_load_and_pv(self) -> np.ndarray:
        """net_electricity_consumption_without_storage_and_partial_load_and_pv` cost time series, in [$]."""

        return self.pricing.electricity_pricing[
               0:self.time_step + 1] * self.net_electricity_consumption_without_storage_and_partial_load_and_pv

    @property
    def net_electricity_consumption_without_storage_and_partial_load_and_pv(self) -> np.ndarray:
        """Net electricity consumption in the absence of flexibility provided by storage devices, 
        partial load cooling and heating devices and self generation time series, in [kWh]. 
        
        Notes
        -----
        net_electricity_consumption_without_storage_and_partial_load_and_pv = 
        `net_electricity_consumption_without_storage_and_partial_load` - `solar_generation`
        """

        return self.net_electricity_consumption_without_storage_and_partial_load - self.solar_generation

    @property
    def net_electricity_consumption_emission_without_storage_and_partial_load(self) -> np.ndarray:
        """Carbon dioxide emmission from `net_electricity_consumption_without_storage_and_partial_load` time series, in [kg_co2]."""

        return (self.carbon_intensity.carbon_intensity[
                0:self.time_step + 1] * self.net_electricity_consumption_without_storage_and_partial_load).clip(min=0)

    @property
    def net_electricity_consumption_cost_without_storage_and_partial_load(self) -> np.ndarray:
        """`net_electricity_consumption_without_storage_and_partial_load` cost time series, in [$]."""

        return self.pricing.electricity_pricing[
               0:self.time_step + 1] * self.net_electricity_consumption_without_storage_and_partial_load

    @property
    def net_electricity_consumption_without_storage_and_partial_load(self):
        """Net electricity consumption in the absence of flexibility provided by 
        storage devices and partial load cooling and heating devices time series, in [kWh]."""

        # cooling electricity consumption
        cooling_demand_difference = self.cooling_demand_without_partial_load - self.cooling_demand
        cooling_electricity_consumption_difference = self.cooling_device.get_input_power(
            cooling_demand_difference,
            self.weather.outdoor_dry_bulb_temperature[0:self.time_step + 1],
            heating=False
        )

        # heating electricity consumption
        heating_demand_difference = self.heating_demand_without_partial_load - self.heating_demand

        if isinstance(self.heating_device, HeatPump):
            heating_electricity_consumption_difference = self.heating_device.get_input_power(
                heating_demand_difference,
                self.weather.outdoor_dry_bulb_temperature[self.time_step],
                heating=True
            )
        else:
            heating_electricity_consumption_difference = self.dhw_device.get_input_power(heating_demand_difference)

        # net electricity consumption without storage and partial load
        return self.net_electricity_consumption_without_storage + np.sum([
            cooling_electricity_consumption_difference,
            heating_electricity_consumption_difference,
        ], axis=0)

    @property
    def heating_demand_without_partial_load(self) -> np.ndarray:
        """Total building space ideal heating demand time series in [kWh].
        
        This is the demand when heating_device is not controlled and always supplies ideal load.
        """

        return self.energy_simulation.heating_demand_without_control[0:self.time_step + 1]

    @property
    def cooling_demand_without_partial_load(self) -> np.ndarray:
        """Total building space ideal cooling demand time series in [kWh].
        
        This is the demand when cooling_device is not controlled and always supplies ideal load.
        """

        return self.energy_simulation.cooling_demand_without_control[0:self.time_step + 1]

    @property
    def indoor_dry_bulb_temperature_without_partial_load(self) -> np.ndarray:
        """Ideal load dry bulb temperature time series in [C].
        
        This is the temperature when cooling_device and heating_device
        are not controlled and always supply ideal load.
        """

        return self.energy_simulation.indoor_dry_bulb_temperature_without_control[0:self.time_step + 1]

    def apply_actions(self, **kwargs):
        super().apply_actions(**kwargs)
        self._update_dynamics_input()

        if self.simulate_dynamics:
            self.update_indoor_dry_bulb_temperature()
        else:
            pass

    def update_indoor_dry_bulb_temperature(self):
        raise NotImplementedError

    def get_dynamics_input(self):
        raise NotImplementedError

    def _update_dynamics_input(self):
        raise NotImplementedError

    def reset_dynamic_variables(self):
        """Resets data file variables that change during control to their initial values.
        
        Resets cooling demand, heating deamand and indoor temperature time series to their initial value 
        at the beginning of an episode.
        """

        start_ix = 0
        end_ix = self.episode_tracker.episode_time_steps
        self.energy_simulation.cooling_demand[
        start_ix:end_ix] = self.energy_simulation.cooling_demand_without_control.copy()[start_ix:end_ix]
        self.energy_simulation.heating_demand[
        start_ix:end_ix] = self.energy_simulation.heating_demand_without_control.copy()[start_ix:end_ix]
        self.energy_simulation.indoor_dry_bulb_temperature[
        start_ix:end_ix] = self.energy_simulation.indoor_dry_bulb_temperature_without_control.copy()[start_ix:end_ix]

    def reset(self):
        """Reset Building to initial state and resets `dynamics`."""

        super().reset()
        self.dynamics.reset()


class LSTMDynamicsBuilding(DynamicsBuilding):
    r"""Class for building with LSTM temperature dynamics model.

    Parameters
    ----------
    *args: Any
        Positional arguments in :py:class:`citylearn.building.Building`.
    dynamics: Dynamics
        Indoor dry-bulb temperature dynamics model.

    Other Parameters
    ----------------
    **kwargs : Any
        Other keyword arguments used to initialize :py:class:`citylearn.building.Building` super class.
    """

    def __init__(self, *args, dynamics: LSTMDynamics, **kwargs):
        super().__init__(*args, dynamics=dynamics, **kwargs)
        self.dynamics: LSTMDynamics

    @DynamicsBuilding.simulate_dynamics.getter
    def simulate_dynamics(self) -> bool:
        return super().simulate_dynamics and self.dynamics._model_input[0][0] is not None

    def update_indoor_dry_bulb_temperature(self):
        """Predict and update indoor dry-bulb temperature for current `time_step`.

        This method will first apply min-max normalization to the model input data where the input data
        is made up of building and district level observations including the predicted 
        :py:attr:`citylearn.building.Building.energy_simulation.indoor_dry_bulb_temperature` 
        with all input variables having a length of :py:attr:`citylearn.dynamics.LSTMDynamics.lookback`.
        asides the `indoor_dry_bulb_temperature` whose input includes all values from
        `time_step` - (`lookback` + 1) to `time_step` - 1, other input variables have values from
        `time_step` - `lookback` to `time_step`. The `indoor_dry_bulb_temperature` for the current `time_step`
        is then predicted using the input data and current `hidden_state` and the predicted values replaces the
        current `time_step` value in :py:attr:`citylearn.building.Building.energy_simulation.indoor_dry_bulb_temperature`.

        Notes
        -----
        LSTM model only uses either cooling/heating demand not both as input variable. 
        Use :py:attr:`citylearn.building.Building.energy_simulation.hvac_mode` to specify whether to consider cooling 
        or heating demand at each `time_step`.
        """

        # predict
        model_input_tensor = torch.tensor(self.get_dynamics_input().T)
        model_input_tensor = model_input_tensor[np.newaxis, :, :]
        hidden_state = tuple([h.data for h in self.dynamics._hidden_state])
        indoor_dry_bulb_temperature_norm, self.dynamics._hidden_state = self.dynamics(model_input_tensor.float(),
                                                                                      hidden_state)

        # update dry bulb temperature for current time step in model input
        ix = self.dynamics.input_observation_names.index('indoor_dry_bulb_temperature')
        self.dynamics._model_input[ix][-1] = indoor_dry_bulb_temperature_norm.item()

        # unnormalize temperature
        low_limit, high_limit = self.dynamics.input_normalization_minimum[ix], self.dynamics.input_normalization_maximum[ix]
        indoor_dry_bulb_temperature = indoor_dry_bulb_temperature_norm*(high_limit - low_limit) + low_limit
        
        # update temperature
        # this function is called after advancing to next timestep 
        # so the cooling demand update and this temperature update are set at the same time step
        self.energy_simulation.indoor_dry_bulb_temperature[self.time_step] = indoor_dry_bulb_temperature.item()

    def get_dynamics_input(self) -> np.ndarray:
        model_input = []

        for i, k in enumerate(self.dynamics.input_observation_names):
            if k == 'indoor_dry_bulb_temperature':
                # indoor temperature values are t = (t - lookback - 1) : t = (t - 1)
                #  i.e. use samples from previous time step to current time step
                model_input.append(self.dynamics._model_input[i][:-1])

            else:
                # other values are t = (t - lookback) : t = (t)
                #  i.e. use samples from previous time step to current time step
                model_input.append(self.dynamics._model_input[i][1:])

        model_input = np.array(model_input, dtype='float32')

        return model_input

    def _update_dynamics_input(self):
        """Updates and returns the input time series for the dynmaics prediction model.

        Updates the model input with the input variables for the current time step. 
        The variables in the input will have length of lookback + 1.
        """

        # get relevant observations for the current time step
        observations = self.observations(include_all=True, normalize=False, periodic_normalization=True)

        # append current time step observations to model input
        # leave out the oldest set of observations and keep only the previous n
        # where n is the lookback + 1 (to include current time step observations)
        self.dynamics._model_input = [
            l[-self.dynamics.lookback:] + [(observations[k] - min_) / (max_ - min_)]
            for l, k, min_, max_ in zip(
                self.dynamics._model_input,
                self.dynamics.input_observation_names,
                self.dynamics.input_normalization_minimum,
                self.dynamics.input_normalization_maximum
            )
        ]

    def update_cooling_demand(self, action: float):
        """Update space cooling demand for current time step.

        Sets the value of :py:attr:`citylearn.building.Building.energy_simulation.cooling_demand` for the current `time_step` to
        the ouput energy of the cooling device where the proportion of its nominal power made available is defined by `action`.
        If :py:attr:`citylearn.building.Building.energy_simulation.hvac_mode` at the next time step is = 0, i.e., off, or = 1, 
        i.e. cooling mode, the demand is set to 0.

        Parameters
        ----------
        action: float
            Proportion of cooling device nominal power that is made available.

        Notes
        -----
        Will only start controlling the heat pump when there are enough observations fo the LSTM lookback until then, maintains
        ideal load. This will imply that the agent does not learn anything in the initial timesteps that are less than the
        lookback. Taking this approach as a 'warm-up' because realistically, there will be no preceding observations to use in 
        lookback.
        """

        # only start controlling the heat pump when there are enough observations fo the LSTM lookback
        # until then, maintain ideal load. This will imply that the agent does not learn anything in the
        # initial timesteps that are less than the lookback. How does this affect learning longterm?
        # Taking this approach as a 'warm-up' because realistically, there will be no preceding observations
        # to use in lookback. Alternatively, one can use the rolled observation values at the end of the time series
        # but it complicates things and is not too realistic.

        if ('cooling_device' in self.active_actions or 'cooling_or_heating_device' in self.active_actions) and self.simulate_dynamics:
            if self.energy_simulation.hvac_mode[self.time_step] in [1, 3]:
                electric_power = action * self.cooling_device.nominal_power
                demand = self.cooling_device.get_max_output_power(
                    self.weather.outdoor_dry_bulb_temperature[self.time_step],
                    heating=False,
                    max_electric_power=electric_power
                )
            else:
                demand = 0.0

            self.energy_simulation.cooling_demand[self.time_step] = demand
        else:
            pass

    def update_heating_demand(self, action: float):
        """Update space heating demand for current time step.

        Sets the value of :py:attr:`citylearn.building.Building.energy_simulation.heating_demand` for the current `time_step` to
        the ouput energy of the heating device where the proportion of its nominal power made available is defined by `action`.
        If :py:attr:`citylearn.building.Building.energy_simulation.hvac_mode` at the next time step is = 0, i.e., off, or = 1, 
        i.e. cooling mode, the demand is set to 0.

        Parameters
        ----------
        action: float
            Proportion of heating device nominal power that is made available.

        Notes
        -----
        Will only start controlling the heat pump when there are enough observations fo the LSTM lookback until then, maintains
        ideal load. This will imply that the agent does not learn anything in the initial timesteps that are less than the
        lookback. Taking this approach as a 'warm-up' because realistically, there will be no preceding observations to use in 
        lookback.
        """
        
        if ('heating_device' in self.active_actions or 'cooling_or_heating_device' in self.active_actions) and self.simulate_dynamics:
            if self.energy_simulation.hvac_mode[self.time_step] in [2, 3]:
                electric_power = action * self.heating_device.nominal_power
                demand = self.heating_device.get_max_output_power(
                    self.weather.outdoor_dry_bulb_temperature[self.time_step],
                    heating=True,
                    max_electric_power=electric_power
                ) if isinstance(self.heating_device, HeatPump) else self.heating_device.get_max_output_power(
                    max_electric_power=electric_power)
            else:
                demand = 0.0

            self.energy_simulation.heating_demand[self.time_step] = demand

        else:
            pass

class OccupantInteractionBuilding(DynamicsBuilding):
    r"""Base class for temperature dynamic and occupant interaction building.

    Parameters
    ----------
    *args: Any
        Positional arguments in :py:class:`citylearn.building.Building`.
    occupant: Occupant
        Occupant thermostat interaction model.
    ignore_occupant: bool, default: False
        Wether to ignore occupant interaction.

    Other Parameters
    ----------------
    **kwargs : Any
        Other keyword arguments used to initialize :py:class:`citylearn.building.Building` super class.
    """

    def __init__(self, *args: Any, occupant: Occupant = None, ignore_occupant: bool = None, **kwargs: Any):
        """Intialize `OccupantInteractionBuilding`"""

        self.occupant = occupant
        self.ignore_occupant = False if ignore_occupant is None else ignore_occupant
        super().__init__(*args, **kwargs)
    
    @DynamicsBuilding.episode_tracker.setter
    def episode_tracker(self, episode_tracker: EpisodeTracker):
        DynamicsBuilding.episode_tracker.fset(self, episode_tracker)
        self.occupant.episode_tracker = episode_tracker

    @DynamicsBuilding.random_seed.setter
    def random_seed(self, seed: int):
        DynamicsBuilding.random_seed.fset(self, seed)
        self.occupant.random_seed = self.random_seed
    
    def apply_actions(self, **kwargs):
        super().apply_actions(**kwargs)

        if self.simulate_dynamics:
            self.update_set_points()
        
        else:
            pass

    def update_set_points(self):
        """Update building indoor temperature dry-bulb temperature, humidity, etc setpoint using occupant interaction model."""

        raise NotImplementedError
    
    def reset_dynamic_variables(self):
        """Resets data file variables that change during control to their initial values.
        
        Resets cooling demand, heating deamand and indoor temperature time series to their initial value 
        at the beginning of an episode.
        """

        start_ix = 0
        end_ix = self.episode_tracker.episode_time_steps
        self.energy_simulation.indoor_dry_bulb_temperature_cooling_set_point[start_ix:end_ix] = self.energy_simulation.indoor_dry_bulb_temperature_cooling_set_point_without_control.copy()[start_ix:end_ix]
        self.energy_simulation.indoor_dry_bulb_temperature_heating_set_point[start_ix:end_ix] = self.energy_simulation.indoor_dry_bulb_temperature_heating_set_point_without_control.copy()[start_ix:end_ix]

    def next_time_step(self):
        super().next_time_step()
        self.occupant.next_time_step()
        
    def reset(self):
        """Reset Building to initial state and resets `dynamics` and `occupant`."""

        super().reset()
        self.occupant.reset()

class LogisticRegressionOccupantInteractionBuilding(OccupantInteractionBuilding, LSTMDynamicsBuilding):
    def __init__(self, *args, occupant: LogisticRegressionOccupant = None, set_point_hold_time_steps: int = None, **kwargs):
        super().__init__(*args, occupant=occupant, **kwargs)
        self.occupant: LogisticRegressionOccupant
        self.__set_point_hold_time_step_counter = None
        self.set_point_hold_time_steps = set_point_hold_time_steps
        self.occupant_interaction_indoor_dry_bulb_temperature_set_point_delta_summary = []
        
    @property
    def set_point_hold_time_steps(self) -> int:
        return self.__set_point_hold_time_steps
    
    @set_point_hold_time_steps.setter
    def set_point_hold_time_steps(self, value: int):
        assert value is None or value >= 0, 'set_point_hold_time_steps must be >= 0'
        self.__set_point_hold_time_steps = np.inf if value is None else int(value)

    def update_set_points(self):
        current_cooling_set_point = self.energy_simulation.indoor_dry_bulb_temperature_cooling_set_point[self.time_step]
        previous_cooling_set_point = self.energy_simulation.indoor_dry_bulb_temperature_cooling_set_point[self.time_step - 1]
        current_heating_set_point = self.energy_simulation.indoor_dry_bulb_temperature_heating_set_point[self.time_step]
        previous_heating_set_point = self.energy_simulation.indoor_dry_bulb_temperature_heating_set_point[self.time_step - 1]
        hvac_mode = self.energy_simulation.hvac_mode[self.time_step]

        if hvac_mode == 1:
            current_set_point = current_cooling_set_point
            previous_set_point = previous_cooling_set_point

        elif hvac_mode == 2:
            current_set_point = current_heating_set_point
            previous_set_point = previous_heating_set_point

        elif hvac_mode == 3:
            raise NotImplementedError(
                'Setpoint update not implemented for auto hvac mode.' 
                ' Set hvac_mode in citylearn.Building.energy_simulation to 0 (off), 1 (cooling mode) or 2 (heating mode).'
            )
        
        else:
            pass

        current_temperature = self.energy_simulation.indoor_dry_bulb_temperature[self.time_step]
        previous_temperature = self.energy_simulation.indoor_dry_bulb_temperature[self.time_step - 1]
        interaction_input = current_temperature
        delta_input = [[current_set_point, previous_set_point, previous_temperature - previous_set_point]]
        model_input = (interaction_input, delta_input) 
        setpoint_delta = self.occupant.predict(x=model_input)
        self.occupant.parameters.occupant_interaction_indoor_dry_bulb_temperature_set_point_delta[self.time_step] = setpoint_delta

        if abs(setpoint_delta) > 0.0 and not self.ignore_occupant:
            if hvac_mode == 1:
                self.energy_simulation.indoor_dry_bulb_temperature_cooling_set_point[self.time_step:] = current_set_point + setpoint_delta

            elif hvac_mode == 2:
                self.energy_simulation.indoor_dry_bulb_temperature_heating_set_point[self.time_step:] = current_set_point + setpoint_delta
            
            else:
                pass

            self.__set_point_hold_time_step_counter = self.set_point_hold_time_steps

        elif self.__set_point_hold_time_step_counter is None:
            pass

        else:
            self.__set_point_hold_time_step_counter -= 1
        
        # revert back to default setpoint schedule if no occupant interaction in defined window
        if self.__set_point_hold_time_step_counter is not None and self.__set_point_hold_time_step_counter == 0:
            self.energy_simulation.indoor_dry_bulb_temperature_cooling_set_point[self.time_step + 1:] \
                = self.energy_simulation.indoor_dry_bulb_temperature_cooling_set_point_without_control[self.time_step + 1:]
            self.energy_simulation.indoor_dry_bulb_temperature_heating_set_point[self.time_step + 1:] \
                = self.energy_simulation.indoor_dry_bulb_temperature_heating_set_point_without_control[self.time_step + 1:]
            self.__set_point_hold_time_step_counter = None

        else:
            pass

    def _get_observations_data(self) -> Mapping[str, Union[float, int]]:
        return {
            **super()._get_observations_data(),
            **{
                k.lstrip('_'): self.occupant.parameters.__getattr__(k.lstrip('_'))[self.time_step] 
                for k, v in vars(self.occupant.parameters).items() if isinstance(v, np.ndarray)
            }
        }

    def _get_observation_space_limits_data(self) -> Mapping[str, List[Union[float, int]]]:
        data = {
            **super()._get_observation_space_limits_data(),
            **{k.lstrip('_'): self.occupant.parameters.__getattr__(
                k.lstrip('_'), 
                start_time_step=self.episode_tracker.simulation_start_time_step, 
                end_time_step=self.episode_tracker.simulation_end_time_step
            ) for k in vars(self.occupant.parameters)},
        }

        # hacky way to set the limits for the occupant setpoint change delta
        delta = max(list(self.occupant.delta_output_map.values()))
        data['occupant_interaction_indoor_dry_bulb_temperature_set_point_delta'][0] = delta
        data['occupant_interaction_indoor_dry_bulb_temperature_set_point_delta'][-1] = -delta

        return data

    def reset_data_sets(self):
        super().reset_data_sets()
        start_time_step = self.episode_tracker.episode_start_time_step
        end_time_step = self.episode_tracker.episode_end_time_step
        self.occupant.parameters.start_time_step = start_time_step
        self.occupant.parameters.end_time_step = end_time_step

    def reset_dynamic_variables(self):
        super().reset_dynamic_variables()
        start_ix = 0
        end_ix = self.episode_tracker.episode_time_steps
        delta_summary = np.unique(self.occupant.parameters.occupant_interaction_indoor_dry_bulb_temperature_set_point_delta[start_ix:end_ix], return_counts=True)
        self.occupant_interaction_indoor_dry_bulb_temperature_set_point_delta_summary.append([delta_summary[0].tolist(), delta_summary[1].tolist()])
        self.occupant.parameters.occupant_interaction_indoor_dry_bulb_temperature_set_point_delta[start_ix:end_ix] =\
            self.occupant.parameters.occupant_interaction_indoor_dry_bulb_temperature_set_point_delta_without_control.copy()[start_ix:end_ix]

    def reset(self):
        super().reset()
        self.__set_point_hold_time_step_counter = None
