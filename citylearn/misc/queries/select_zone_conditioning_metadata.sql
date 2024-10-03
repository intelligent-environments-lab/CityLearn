WITH zone_conditioning AS (
    SELECT
        z.ZoneIndex,
        SUM(CASE WHEN s.Name = 'Zone Thermostat Cooling Setpoint Temperature' AND s.average_setpoint > 0 THEN 1 ELSE 0 END) AS is_cooled,
        SUM(CASE WHEN s.Name = 'Zone Thermostat Heating Setpoint Temperature' AND s.average_setpoint > 0 THEN 1 ELSE 0 END) AS is_heated,
        MAX(CASE WHEN s.Name = 'Zone Thermostat Cooling Setpoint Temperature' AND s.average_setpoint > 0 THEN s.average_setpoint 
            ELSE NULL END) AS average_cooling_setpoint,
        MAX(CASE WHEN s.Name = 'Zone Thermostat Heating Setpoint Temperature' AND s.average_setpoint > 0 THEN s.average_setpoint 
            ELSE NULL END) AS average_heating_setpoint
    FROM (
        SELECT
            d.KeyValue,
            d.Name,
            AVG(r."value") AS average_setpoint
        FROM ReportData r
        INNER JOIN ReportDataDictionary d ON d.ReportDataDictionaryIndex = r.ReportDataDictionaryIndex
        WHERE d.Name IN ('Zone Thermostat Cooling Setpoint Temperature', 'Zone Thermostat Heating Setpoint Temperature')
        GROUP BY d.KeyValue, d.Name
    ) s
    LEFT JOIN Zones z ON z.ZoneName = s.KeyValue
    GROUP BY
        z.ZoneName,
        z.ZoneIndex
)
-- get zone floor area proportion of total zone floor area
SELECT
    z.ZoneName AS zone_name,
    z.ZoneIndex AS zone_index,
    z.Multiplier AS multiplier,
    z.Volume AS volume,
    z.FloorArea AS floor_area,
    (z.FloorArea*z.Multiplier)/t.total_floor_area AS total_floor_area_proportion,
    CASE WHEN c.is_cooled != 0 OR c.is_heated != 0 THEN (z.FloorArea*z.Multiplier)/t.conditioned_floor_area ELSE 0 END AS conditioned_floor_area_proportion,
    c.is_cooled,
    c.is_heated,
    c.average_cooling_setpoint,
    c.average_heating_setpoint
FROM Zones z
CROSS JOIN (
    SELECT
        SUM(z.FloorArea*z.Multiplier) AS total_floor_area,
        SUM(CASE WHEN c.is_cooled != 0 OR c.is_heated != 0 THEN z.FloorArea*z.Multiplier ELSE 0 END) AS conditioned_floor_area
    FROM Zones z
    LEFT JOIN zone_conditioning c ON c.ZoneIndex = z.ZoneIndex
) t
LEFT JOIN zone_conditioning c ON c.ZoneIndex = z.ZoneIndex
;