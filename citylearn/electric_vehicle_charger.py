import inspect
from typing import List, Dict
import numpy as np
from citylearn.base import Environment
from citylearn.electric_vehicle import ElectricVehicle
from citylearn.data import ZERO_DIVISION_PLACEHOLDER
np.seterr(divide='ignore', invalid='ignore')

class Charger(Environment):
    def __init__(
            self, nominal_power: float, efficiency: float = None, charger_id: str = None, charger_type: int = None, max_charging_power: float = None,
            min_charging_power: float = None, max_discharging_power: float = None,  min_discharging_power: float = None, charge_efficiency_curve: Dict[float, float] = None,
            discharge_efficiency_curve: Dict[float, float] = None, connected_electric_vehicle: ElectricVehicle = None, incoming_electric_vehicle: ElectricVehicle = None,
            **kwargs
    ):
        r"""Initializes the `Electric Vehicle Charger` class with the given attributes.

        Parameters
        ----------
        charger_id: str
            Id through which the charger is uniquely identified in the system
        charger_type: int
            Either private (0) or public (1) charger
        max_charging_power : float, default 50
            Maximum charging power in kW.
        min_charging_power : float, default 0
            Minimum charging power in kW.
        max_discharging_power : float, default 50
            Maximum discharging power in kW.
        min_discharging_power : float, default 0
            Minimum discharging power in kW.
        charge_efficiency_curve : dict, default {3.6: 0.95, 7.2: 0.97, 22: 0.98, 50: 0.98}
            Efficiency curve for charging containing power levels and corresponding efficiency values.
        discharge_efficiency_curve : dict, default {3.6: 0.95, 7.2: 0.97, 22: 0.98, 50: 0.98}
            Efficiency curve for discharging containing power levels and corresponding efficiency values.

        Other Parameters
        ----------------
        **kwargs : dict
            Other keyword arguments used to initialize super classes.
        """

        self.nominal_power = nominal_power
        self.efficiency = efficiency
        self.charger_id = charger_id
        self.charger_type = charger_type
        self.max_charging_power = max_charging_power
        self.min_charging_power = min_charging_power
        self.max_discharging_power = max_discharging_power
        self.min_discharging_power = min_discharging_power
        self.charge_efficiency_curve = charge_efficiency_curve
        self.discharge_efficiency_curve = discharge_efficiency_curve
        self.connected_electric_vehicle = connected_electric_vehicle
        self.incoming_electric_vehicle = incoming_electric_vehicle

        arg_spec = inspect.getfullargspec(super().__init__)
        kwargs = {
            key: value for (key, value) in kwargs.items()
            if (key in arg_spec.args or (arg_spec.varkw is not None))
        }
        super().__init__(**kwargs)

    @property
    def charger_id(self) -> str:
        """ID of the charger."""

        return self.__charger_id

    @property
    def charger_type(self) -> int:
        """Type of the charger."""

        return self.__charger_type

    @property
    def max_charging_power(self) -> float:
        """Maximum charging power in kW."""

        return self.__max_charging_power

    @property
    def min_charging_power(self) -> float:
        """Minimum charging power in kW."""

        return self.__min_charging_power

    @property
    def max_discharging_power(self) -> float:
        """Maximum discharging power in kW."""

        return self.__max_discharging_power

    @property
    def min_discharging_power(self) -> float:
        """Minimum discharging power in kW."""

        return self.__min_discharging_power
    
    @property
    def charge_efficiency_curve(self) -> dict:
        """Efficiency curve for charging containing power levels and corresponding efficiency values."""

        return self.__charge_efficiency_curve

    @property
    def discharge_efficiency_curve(self) -> dict:
        """Efficiency curve for discharging containing power levels and corresponding efficiency values."""

        return self.__discharge_efficiency_curve

    @property
    def connected_electric_vehicle(self) -> ElectricVehicle:
        """Electric_Vehicle currently connected to charger"""

        return self.__connected_ev

    @property
    def incoming_electric_vehicle(self) -> ElectricVehicle:
        """Electric_Vehicle incoming to charger"""

        return self.__incoming_ev

    @property
    def efficiency(self) -> float:
        """Technical efficiency."""

        return self.__efficiency

    @property
    def nominal_power(self) -> float:
        r"""Nominal power."""

        return self.__nominal_power

    @property
    def past_connected_evs(self) -> List[ElectricVehicle]:
        r"""Each timestep with the list of Past connected Evs or None in the case no electric_vehicle was connected """

        return self.__past_connected_evs

    @property
    def past_charging_action_values(self) -> List[float]:
        r"""Actions given to charge/discharge in [kWh]. Different from the electricity consumption as in this an action can be given but no electric_vehicle being connect it will not consume such energy"""
       
        return self.__past_charging_action_values

    @property
    def electricity_consumption(self) -> List[float]:
        r"""Electricity consumption time series."""

        return self.__electricity_consumption

    @property
    def available_nominal_power(self) -> float:
        r"""Difference between `nominal_power` and `electricity_consumption` at current `time_step`."""

        return None if self.nominal_power is None else self.nominal_power - self.electricity_consumption[self.time_step]
    

    @charger_id.setter
    def charger_id(self, charger_id: str):
        self.__charger_id = charger_id

    @charger_type.setter
    def charger_type(self, charger_type: str):
        self.__charger_type = charger_type

    @max_charging_power.setter
    def max_charging_power(self, max_charging_power: float):
            if max_charging_power is None:
                self.__max_charging_power = 50.0
            else:
                self.__max_charging_power = max_charging_power

    @min_charging_power.setter
    def min_charging_power(self, min_charging_power: float):
            if min_charging_power is None:
                self.__min_charging_power = 0.0
            else:
                self.__min_charging_power = min_charging_power

    @max_discharging_power.setter
    def max_discharging_power(self, max_discharging_power: float):
            if max_discharging_power is None:
                self.__max_discharging_power = 50.0
            else:
                self.__max_discharging_power = max_discharging_power

    @min_discharging_power.setter
    def min_discharging_power(self, min_discharging_power: float):
            if min_discharging_power is None:
                self.__min_discharging_power = 0.0
            else:
                self.__min_discharging_power = min_discharging_power

    @charge_efficiency_curve.setter
    def charge_efficiency_curve(self, charge_efficiency_curve: List[List[float]]):
        if charge_efficiency_curve is None:
            charge_efficiency_curve = [[3.6, 0.95],[7.2, 0.97],[22, 0.98],[50, 0.98]]
        else:
            pass
        self.__charge_efficiency_curve = np.array(charge_efficiency_curve).T
    
    @discharge_efficiency_curve.setter
    def discharge_efficiency_curve(self, discharge_efficiency_curve: List[List[float]]):
        if discharge_efficiency_curve is None:
            discharge_efficiency_curve = [[3.6, 0.95],[7.2, 0.97],[22, 0.98],[50, 0.98]]
        else:
            pass
        self.__discharge_efficiency_curve = np.array(discharge_efficiency_curve).T

    @connected_electric_vehicle.setter
    def connected_electric_vehicle(self, electric_vehicle: ElectricVehicle):
            self.__connected_ev = electric_vehicle if electric_vehicle is None else None 

    @incoming_electric_vehicle.setter
    def incoming_electric_vehicle(self, electric_vehicle: ElectricVehicle):
            self.__incoming_ev = electric_vehicle if electric_vehicle is None else None 

    @efficiency.setter
    def efficiency(self, efficiency: float):
            if efficiency is None:
                self.__efficiency = 1.0
            else:
                assert efficiency > 0, 'efficiency must be > 0.'
                self.__efficiency = efficiency

    @nominal_power.setter
    def nominal_power(self, nominal_power: float):
            if nominal_power is None or nominal_power == 0:
                self.__nominal_power = ZERO_DIVISION_PLACEHOLDER
            else:
                assert nominal_power >= 0, 'nominal_power must be >= 0.'
                self.__nominal_power = nominal_power


    def plug_car(self, electric_vehicle: ElectricVehicle):
        """
        Connects a electric_vehicle to the charger.

        Parameters
        ----------
        electric_vehicle : object
            electric_vehicle instance to be connected to the charger.

        Raises
        ------
        ValueError
            If the charger has reached its maximum connected electric_vehicle' capacity.
        """
        self.__past_connected_evs[self.time_step] = electric_vehicle
        self.connected_electric_vehicle = electric_vehicle

    def unplug_car(self):
        """
        Disconnects a electric_vehicle from the charger.

        Parameters
        ----------
        electric_vehicle : object
            electric_vehicle instance to be disconnected from the charger.
        """
        self.connected_electric_vehicle = None

    def associate_incoming_car(self, electric_vehicle: ElectricVehicle):
        """
        Associates incoming electric_vehicle to the charger.

        Parameters
        ----------
        electric_vehicle : object
            electric_vehicle instance to be connected to the charger.

        Raises
        ------
        ValueError
            If the charger has reached its maximum associated electric_vehicle' capacity.
        """
        self.incoming_electric_vehicle = electric_vehicle

        # else:
        #    raise ValueError("Charger has reached its maximum associated electric_vehicle capacity")

    def disassociate_incoming_car(self):
        """
        Disassociates incoming electric_vehicle from the charger.

        Parameters
        ----------
        electric_vehicle : object
            electric_vehicle instance to be disconnected from the charger.
        """
        self.incoming_electric_vehicle = None

    def update_connected_electric_vehicle_soc(self, action_value: float):
        self.__past_charging_action_values[self.time_step] = action_value #ToDo
        if self.connected_electric_vehicle and action_value != 0:
            electric_vehicle = self.connected_electric_vehicle
            if action_value > 0:
                energy = action_value * self.max_charging_power
            else:
                energy = action_value * self.max_discharging_power

            charging = energy >= 0

            if charging:
                # make sure we do not charge beyond the maximum capacity
                energy = min(energy, electric_vehicle.battery.capacity - electric_vehicle.battery.soc[self.time_step])
            else:
                # make sure we do not discharge beyond the minimum level (assuming it's zero)
                max_discharge = - (electric_vehicle.battery.soc[self.time_step] - electric_vehicle.min_battery_soc/100 * electric_vehicle.battery.capacity)
                energy = max(energy, max_discharge)

            energy_kwh = energy * self.efficiency

            # Here we call the electric_vehicle's battery's charge method directly, passing the energy (positive for charging,
            # negative for discharging)
            electric_vehicle.battery.charge(energy_kwh)
            self.__electricity_consumption[self.time_step] = electric_vehicle.battery.electricity_consumption[-1]
        else:
            self.__electricity_consumption[self.time_step] = 0

    def next_time_step(self):
        r"""Advance to next `time_step` and set `electricity_consumption` at new `time_step` to 0.0."""

        self.__electricity_consumption.append(0.0)
        self.__past_connected_evs.append(None)
        self.__past_charging_action_values.append(0.0)
        self.connected_electric_vehicle = None
        self.incoming_electric_vehicle = None
        super().next_time_step()

    def reset(self):
        """
        Resets the Charger to its initial state by disconnecting all electric_vehicles.
        """
        super().reset()
        self.connected_electric_vehicle = None
        self.incoming_electric_vehicle = None
        self.__electricity_consumption = [0.0]
        self.__past_connected_evs = [None]
        self.__past_charging_action_values = [0.0]

    def __str__(self):
       return (
            f"Charger ID: {self.charger_id}\n"
            f"electricity consumption: {self.electricity_consumption} kW\n"
            f"past_connected_evs: {self.past_connected_evs} kW\n"
            f"past_charging_action_values: {self.past_charging_action_values} kW\n"
            f"Currently Connected electric_vehicle: {self.connected_electric_vehicle}\n"
            f"Incoming electric_vehicle: {self.incoming_electric_vehicle}\n"
       )
