- [ ] Building timeseries.
- [ ] Heat pump `t_target_heating` in `building_attributes.json` to be adequate for space heating.
- [ ] Resize DHW tank to meet SH peak hourly demand as done with cold water tank.
- [ ] Rename DHW storage to HW storage in code and `buildings_state_action_space.json`?

During heating, use heat pump to heat to `t_target_heating` and supplement with electric heater to meet heating load at timestemp `t`.