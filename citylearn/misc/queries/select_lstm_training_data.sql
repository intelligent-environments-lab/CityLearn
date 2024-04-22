WITH u AS (
    -- site variables
    SELECT
        r.TimeIndex,
        r.ReportDataDictionaryIndex,
        'site_variable' AS label,
        r.Value
    FROM ReportData r
    LEFT JOIN ReportDataDictionary d ON d.ReportDataDictionaryIndex = r.ReportDataDictionaryIndex
    WHERE d.Name IN (
        'Site Direct Solar Radiation Rate per Area', 
        'Site Diffuse Solar Radiation Rate per Area', 
        'Site Outdoor Air Drybulb Temperature'
    )

    UNION ALL

    -- weighted conditioned zone variables
    SELECT
        r.TimeIndex,
        r.ReportDataDictionaryIndex,
        'weighted_variable' AS label,
        r.Value
    FROM weighted_variable r
    LEFT JOIN ReportDataDictionary d ON d.ReportDataDictionaryIndex = r.ReportDataDictionaryIndex
    WHERE d.Name IN ('Zone Air Temperature')

    UNION ALL

    -- thermal load variables
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
        'occupant_count' AS label,
        r.Value
    FROM ReportData r
    LEFT JOIN ReportDataDictionary d ON d.ReportDataDictionaryIndex = r.ReportDataDictionaryIndex
    WHERE 
        d.Name = 'Zone People Occupant Count'
), p AS (
    SELECT
        u.TimeIndex,
        MAX(CASE WHEN d.Name = 'Site Direct Solar Radiation Rate per Area' THEN Value END) AS direct_solar_radiation,
        MAX(CASE WHEN d.Name = 'Site Diffuse Solar Radiation Rate per Area' THEN Value END) AS diffuse_solar_radiation,
        MAX(CASE WHEN d.Name = 'Site Outdoor Air Drybulb Temperature' THEN Value END) AS outdoor_air_temperature,
        SUM(CASE WHEN d.Name = 'Zone Air Temperature' THEN Value END) AS average_indoor_air_temperature,
        SUM(CASE WHEN d.Name = 'Zone People Occupant Count' THEN Value END) AS occupant_count,
        SUM(CASE WHEN u.label = 'cooling_load' THEN Value/1000.0 END) AS cooling_load,
        SUM(CASE WHEN u.label = 'heating_load' THEN Value/1000.0 END) AS heating_load
    FROM u
    LEFT JOIN ReportDataDictionary d ON d.ReportDataDictionaryIndex = u.ReportDataDictionaryIndex
    GROUP BY u.TimeIndex
)
SELECT
    t.TimeIndex AS timestep,
    t.Month AS month,
    t.Day AS day,
    t.DayType AS day_name,
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
    END AS day_of_week,
    t.Hour AS hour,
    t.Minute AS minute,
    p.direct_solar_radiation,
    p.diffuse_solar_radiation,
    p.outdoor_air_temperature,
    p.average_indoor_air_temperature,
    p.occupant_count,
    COALESCE(p.cooling_load, 0) AS cooling_load,
    COALESCE(p.heating_load, 0) AS heating_load
FROM p
LEFT JOIN Time t ON t.TimeIndex = p.TimeIndex
WHERE t.DayType NOT IN ('SummerDesignDay', 'WinterDesignDay')
;