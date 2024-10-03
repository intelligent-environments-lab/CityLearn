WITH u AS (
    -- weighted conditioned zone variables
    SELECT
        r.TimeIndex,
        r.ReportDataDictionaryIndex,
        'weighted_variable' AS label,
        r.Value
    FROM weighted_variable r
    LEFT JOIN ReportDataDictionary d ON d.ReportDataDictionaryIndex = r.ReportDataDictionaryIndex
    WHERE d.Name IN ('Zone Air Temperature', 'Zone Air Relative Humidity')

    UNION ALL

    -- setpoint
    SELECT
        r.TimeIndex,
        r.ReportDataDictionaryIndex,
        'setpoint' AS label,
        r.Value
    FROM ReportData r
    LEFT JOIN ReportDataDictionary d ON d.ReportDataDictionaryIndex = r.ReportDataDictionaryIndex
    LEFT JOIN Zones z ON z.ZoneName = d.KeyValue
    WHERE d.Name IN ('Zone Thermostat Cooling Setpoint Temperature', 'Zone Thermostat Heating Setpoint Temperature')

    UNION ALL

    --  thermal load
    SELECT
        r.TimeIndex,
        r.ReportDataDictionaryIndex,
        CASE WHEN r.Value > 0 THEN 'heating_load' ELSE 'cooling_load' END AS label,
        ABS(r.Value) AS Value
    FROM ReportData r
    LEFT JOIN ReportDataDictionary d ON d.ReportDataDictionaryIndex = r.ReportDataDictionaryIndex
    WHERE 
        d.Name = 'Other Equipment Convective Heating Rate' AND
        (d.KeyValue LIKE '%HEATING_LOAD' OR d.KeyValue LIKE '%COOLING_LOAD')

    UNION ALL

    -- other variables
    SELECT
        r.TimeIndex,
        r.ReportDataDictionaryIndex,
        'other' AS label,
        r.Value
    FROM ReportData r
    LEFT JOIN ReportDataDictionary d ON d.ReportDataDictionaryIndex = r.ReportDataDictionaryIndex
    WHERE 
        d.Name IN (
            'Water Use Equipment Heating Rate',
            'Zone Lights Electricity Rate',
            'Zone Electric Equipment Electricity Rate',
            'Zone People Occupant Count'
        )
), p AS (
    SELECT
        u.TimeIndex,
        SUM(CASE WHEN d.Name = 'Zone Air Temperature' THEN Value END) AS indoor_dry_bulb_temperature,
        SUM(CASE WHEN d.Name = 'Zone Air Relative Humidity' THEN Value END) AS indoor_relative_humidity,
        SUM(CASE WHEN d.Name IN ('Zone Lights Electricity Rate', 'Zone Electric Equipment Electricity Rate') THEN Value/(1000.0) END) AS non_shiftable_load,
        SUM(CASE WHEN d.Name = 'Water Use Equipment Heating Rate' THEN ABS(Value)/(1000.0) END) AS dhw_demand,
        SUM(CASE WHEN d.Name = 'Other Equipment Convective Heating Rate' AND u.label = 'cooling_load' THEN ABS(Value)/(1000.0) END) AS cooling_demand,
        SUM(CASE WHEN d.Name = 'Other Equipment Convective Heating Rate' AND u.label = 'heating_load' THEN ABS(Value)/(1000.0) END) AS heating_demand,
        SUM(CASE WHEN d.Name = 'Zone People Occupant Count' THEN Value END) AS occupant_count,
        MAX(CASE WHEN d.Name = 'Zone Thermostat Cooling Setpoint Temperature' THEN Value END) AS indoor_dry_bulb_temperature_cooling_set_point,
        MAX(CASE WHEN d.Name = 'Zone Thermostat Heating Setpoint Temperature' THEN Value END) AS indoor_dry_bulb_temperature_heating_set_point
    FROM u
    LEFT JOIN ReportDataDictionary d ON d.ReportDataDictionaryIndex = u.ReportDataDictionaryIndex
    GROUP BY u.TimeIndex
)

SELECT
    t.Month AS "month",
    t.Hour AS "hour",
    CASE
        WHEN t.DayType = 'Monday' THEN 1
            WHEN t.DayType = 'Tuesday' THEN 2
                WHEN t.DayType = 'Wednesday' THEN 3
                    WHEN t.DayType = 'Thursday' THEN 4
                        WHEN t.DayType = 'Friday' THEN 5
                            WHEN t.DayType = 'Saturday' THEN 6
                                WHEN t.DayType = 'Sunday' THEN 7
                                    WHEN t.DayType = 'Holiday' THEN 8
                                        ELSE NULL
                                            END AS day_type,
    t.Dst AS daylight_savings_status,
    p.indoor_dry_bulb_temperature,
    p.indoor_relative_humidity,
    p.non_shiftable_load,
    p.dhw_demand,
    CASE 
        WHEN COALESCE(p.cooling_demand, 0.0) > COALESCE(p.heating_demand, 0.0) 
            THEN COALESCE(p.cooling_demand, 0.0) ELSE 0.0 END AS cooling_demand,
    CASE 
        WHEN COALESCE(p.heating_demand, 0.0) > COALESCE(p.cooling_demand, 0.0) 
            THEN COALESCE(p.heating_demand, 0.0) ELSE 0.0 END AS heating_demand,
    0.0 AS solar_generation,
    p.occupant_count,
    p.indoor_dry_bulb_temperature_cooling_set_point,
    p.indoor_dry_bulb_temperature_heating_set_point,
    3 AS hvac_mode
FROM p
LEFT JOIN Time t ON t.TimeIndex = p.TimeIndex
WHERE t.DayType NOT IN ('SummerDesignDay', 'WinterDesignDay')
ORDER BY t.TimeIndex
;