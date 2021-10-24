WITH zone AS (
    -- get zone floor area proportion of total zone floor area
    SELECT
        z.ZoneName,
        z.Multiplier,
        z.Volume,
        z.FloorArea,
        (z.FloorArea*z.Multiplier)/t.total_floor_area AS floor_area_proportion
    FROM Zones z
    CROSS JOIN (
        SELECT
            SUM(FloorArea*Multiplier) AS total_floor_area
        FROM Zones
    ) t
), unioned_variables AS (
    -- weighted_cooling_setpoint_difference
    SELECT
        r.TimeIndex,
        r.ReportDataDictionaryIndex,
        ABS(s.value - r.Value)*z.floor_area_proportion AS value
    FROM ReportData r
    INNER JOIN ReportDataDictionary d ON d.ReportDataDictionaryIndex = r.ReportDataDictionaryIndex
    INNER JOIN zone z ON z.ZoneName = d.KeyValue
    INNER JOIN (
        SELECT
            r.TimeIndex,
            d.KeyValue,
            r.Value AS value
        FROM ReportData r
        INNER JOIN ReportDataDictionary d ON d.ReportDataDictionaryIndex = r.ReportDataDictionaryIndex
        WHERE
            d.Name IN ('Zone Air Temperature')
    ) s ON 
        s.TimeIndex = r.TimeIndex
        AND s.KeyValue = d.KeyValue
    WHERE
        d.Name IN ('Zone Thermostat Cooling Setpoint Temperature')

    UNION

    -- other_weighted_average_variable
    SELECT
        r.TimeIndex,
        r.ReportDataDictionaryIndex,
        r.Value*z.floor_area_proportion AS value
    FROM ReportData r
    INNER JOIN ReportDataDictionary d ON d.ReportDataDictionaryIndex = r.ReportDataDictionaryIndex
    INNER JOIN zone z ON z.ZoneName = d.KeyValue
    WHERE
        d.Name IN ('Zone Air Temperature', 'Zone Air Relative Humidity')
    
    UNION

    -- load
    SELECT
        r.TimeIndex,
        r.ReportDataDictionaryIndex,
        r.Value AS value
    FROM ReportData r
    INNER JOIN ReportDataDictionary d ON d.ReportDataDictionaryIndex = r.ReportDataDictionaryIndex
    WHERE
        d.Name IN ('Zone Ideal Loads Zone Total Cooling Energy', 'Zone Ideal Loads Zone Total Heating Energy', 'Electric Equipment Electricity Energy')
), aggregate AS (
    -- sum the varaibles per timestamp
    SELECT
        u.TimeIndex,
        d.Name,
        SUM(u.value) AS value
    FROM unioned_variables u
    INNER JOIN ReportDataDictionary d ON d.ReportDataDictionaryIndex = u.ReportDataDictionaryIndex
    GROUP BY
        u.TimeIndex,
        d.Name
), aggregate_pivot AS (
    -- pivot table to match CityLearn input format
    SELECT
        a.TimeIndex,
        SUM(CASE WHEN a.Name = 'Zone Air Temperature' THEN a.value END) AS "Indoor Temperature (C)",
        SUM(CASE WHEN a.Name = 'Zone Thermostat Cooling Setpoint Temperature' THEN a.value END) AS "Average Unmet Cooling Setpoint Difference (C)",
        SUM(CASE WHEN a.Name = 'Zone Air Relative Humidity' THEN a.value END) AS "Indoor Relative Humidity (%)",
        SUM(CASE WHEN a.Name = 'Electric Equipment Electricity Energy' THEN (a.value/3600)/1000 END) AS "Equipment Electric Power (kWh)",
        SUM(CASE WHEN a.Name = 'Zone Ideal Loads Zone Total Heating Energy' THEN (a.value/3600)/1000 END) AS "Heating Load (kWh)",
        SUM(CASE WHEN a.Name = 'Zone Ideal Loads Zone Total Cooling Energy' THEN (a.value/3600)/1000 END) AS "Cooling Load (kWh)"
    FROM aggregate a
    GROUP BY
        a.TimeIndex
)

-- define time-related columns
SELECT
    t.Month,
    t.Hour,
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
    END AS "Day Type",
    t.Dst AS "Daylight Savings Status",
    a."Indoor Temperature (C)",
    a."Average Unmet Cooling Setpoint Difference (C)",
    a."Indoor Relative Humidity (%)",
    a."Equipment Electric Power (kWh)",
    a."Heating Load (kWh)",
    a."Cooling Load (kWh)"
FROM aggregate_pivot a
INNER JOIN Time t ON t.TimeIndex = a.TimeIndex
WHERE
    t.DayType NOT IN ('SummerDesignDay', 'WinterDesignDay')
ORDER BY
    t.TimeIndex