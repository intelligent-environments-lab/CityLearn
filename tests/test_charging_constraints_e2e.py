"""
Run from tests folder: python3 test_charging_constraints_e2e.py

This exercises the charging-constraint demo dataset end-to-end with the baseline
EV reference controller, mirroring tests/test_evs.py. It ensures the new schema
parses, agents can train for one episode, and KPI evaluation succeeds.
"""
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
PARENT = ROOT_DIR.as_posix()
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)

from citylearn.agents.rbc import BasicElectricVehicleRBC_ReferenceController as Agent
from citylearn.citylearn import CityLearnEnv

DATASET_PATH = ROOT_DIR / 'data/datasets/citylearn_charging_constraints_demo/schema.json'
RENDER_DIR = Path('tests/tmp/render_charging_constraints')


def main():
    env = CityLearnEnv(
        str(DATASET_PATH),
        central_agent=True,
        render=False,
        episode_time_steps=96,
    )

    model = Agent(env)
    model.learn(episodes=1, logging_level=1)

    kpis = model.env.evaluate()
    kpis = kpis.pivot(index='cost_function', columns='name', values='value').round(3)
    kpis = kpis.dropna(how='all')
    print(kpis)

    print('Charging constraint KPI sample:')
    print(kpis.head())


if __name__ == '__main__':
    main()
