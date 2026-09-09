import pandas as pd

from scripts.generate_annual_rec_suite import TIMEZONE, _local_at


def test_local_at_preserves_wall_clock_after_dst_transitions():
    spring_day = pd.Timestamp("2023-03-26", tz=TIMEZONE)
    autumn_day = pd.Timestamp("2023-10-29", tz=TIMEZONE)

    spring = _local_at(spring_day, 8 * 60)
    autumn = _local_at(autumn_day, 8 * 60)

    assert (spring.hour, spring.minute) == (8, 0)
    assert spring.utcoffset() == pd.Timedelta(hours=1)
    assert (autumn.hour, autumn.minute) == (8, 0)
    assert autumn.utcoffset() == pd.Timedelta(hours=0)


def test_local_at_resolves_missing_and_repeated_civil_hours_deterministically():
    spring_day = pd.Timestamp("2023-03-26", tz=TIMEZONE)
    autumn_day = pd.Timestamp("2023-10-29", tz=TIMEZONE)

    missing = _local_at(spring_day, 90)
    repeated = _local_at(autumn_day, 90)

    assert (missing.hour, missing.minute) == (2, 30)
    assert missing.utcoffset() == pd.Timedelta(hours=1)
    assert (repeated.hour, repeated.minute) == (1, 30)
    assert repeated.utcoffset() == pd.Timedelta(hours=1)
