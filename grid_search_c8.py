import itertools
import sys
import pandas as pd
from utilities import read_json, write_data

GRID_SEARCH_FILEPATH = 'grid_search_c8.json'
WRITE_FREQUENCY = 4000
PYTHON_EXECUTION = f'{sys.executable} -m citylearn_cli --write_sqlite --write_frequency {WRITE_FREQUENCY} --write_pickle single'
TACC_LAUNCHER_JOB_FILEPATH = 'tacc_launcher_job_c8'


def main():
    set_grid()

def set_grid():
    grid_search = read_json(GRID_SEARCH_FILEPATH)
    param_names = list(grid_search.keys())
    param_values = list(grid_search.values())
    param_values_grid = list(itertools.product(*param_values))
    grid = pd.DataFrame(param_values_grid,columns=param_names)
    grid = grid.drop_duplicates()
    grid['--simulation_id'] = grid.reset_index().index.map(lambda x: f'simulation_{x + 1}')
    grid.to_csv(f'{GRID_SEARCH_FILEPATH.split(".")[0]}.csv',index=False)
    script = [[
        f'{key if key.startswith("--") else ""} {value if not isinstance(value,bool) else ""}'.strip() for key, value in record.items()
        if (value is not None and not isinstance(value,bool)) or (isinstance(value,bool) and value)
    ] for record in grid.to_dict(orient='records')]
    script = [PYTHON_EXECUTION + ' ' + ' '.join(c) for c in script] 
    script.append('') # append blank line
    script = '\n'.join(script)
    write_data(script,TACC_LAUNCHER_JOB_FILEPATH)
    print('Number of simulations to run:',grid.shape[0])

if __name__ == '__main__':
    main()