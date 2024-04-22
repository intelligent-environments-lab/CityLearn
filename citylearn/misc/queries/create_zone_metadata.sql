DROP TABLE IF EXISTS zone_metadata;
CREATE TABLE IF NOT EXISTS zone_metadata (
    zone_index INTEGER PRIMARY KEY,
    zone_name TEXT,
    multiplier REAL,
    volume REAL,
    floor_area REAL,
    total_floor_area_proportion REAL,
    conditioned_floor_area_proportion REAL,
    is_cooled INTEGER,
    is_heated INTEGER,
    average_cooling_setpoint REAL,
    average_heating_setpoint REAL
);

DROP VIEW IF EXISTS weighted_variable;
CREATE VIEW weighted_variable AS
    SELECT
        r.TimeIndex,
        r.ReportDataDictionaryIndex,
        r.Value*z.conditioned_floor_area_proportion AS Value
    FROM ReportData r
    LEFT JOIN ReportDataDictionary d ON d.ReportDataDictionaryIndex = r.ReportDataDictionaryIndex
    INNER JOIN (SELECT * FROM zone_metadata WHERE is_cooled + is_heated >= 1) z ON z.zone_name = d.KeyValue
    WHERE d.Name IN ('Zone Air Temperature', 'Zone Air Relative Humidity')
;