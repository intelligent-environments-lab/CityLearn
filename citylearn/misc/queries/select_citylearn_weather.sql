SELECT
    MAX(CASE WHEN d.Name = 'Site Outdoor Air Drybulb Temperature' THEN r.value END) AS outdoor_dry_bulb_temperature,
    MAX(CASE WHEN d.Name = 'Site Outdoor Air Relative Humidity' THEN r.value END) AS outdoor_relative_humidity,
    MAX(CASE WHEN d.Name = 'Site Diffuse Solar Radiation Rate per Area' THEN r.value END) AS diffuse_solar_irradiance,
    MAX(CASE WHEN d.Name = 'Site Direct Solar Radiation Rate per Area' THEN r.value END) AS direct_solar_irradiance
FROM ReportData r
LEFT JOIN ReportDataDictionary d ON d.ReportDataDictionaryIndex = r.ReportDataDictionaryIndex
LEFT JOIN Time t ON t.TimeIndex = r.TimeIndex
WHERE t.DayType NOT IN ('SummerDesignDay', 'WinterDesignDay')
GROUP BY t.TimeIndex
ORDER BY t.TimeIndex
;