SELECT
    r.TimeIndex AS timestep,
    z.ZoneIndex AS zone_index,
    z.ZoneName AS zone_name,
    'cooling_load' AS load,
    r.Value AS value
FROM ReportData r
INNER JOIN ReportDataDictionary d ON d.ReportDataDictionaryIndex = r.ReportDataDictionaryIndex
LEFT JOIN Zones z ON z.ZoneName = REPLACE(d.KeyValue, ' IDEAL LOADS AIR SYSTEM', '')
WHERE d.Name = 'Zone Ideal Loads Zone Sensible Cooling Rate' AND z.ZoneName IN (<cooled_zone_names>)

UNION ALL

SELECT
    r.TimeIndex AS timestep,
    z.ZoneIndex AS zone_index,
    z.ZoneName AS zone_name,
    'heating_load' AS load,
    r.Value AS value
FROM ReportData r
INNER JOIN ReportDataDictionary d ON d.ReportDataDictionaryIndex = r.ReportDataDictionaryIndex
LEFT JOIN Zones z ON z.ZoneName = REPLACE(d.KeyValue, ' IDEAL LOADS AIR SYSTEM', '')
WHERE d.Name = 'Zone Ideal Loads Zone Sensible Heating Rate' AND z.ZoneName IN (<heated_zone_names>)
;