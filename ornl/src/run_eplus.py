"""multiprocessing runs
using generators instead of a list
when you are running a 100 files you have to use generators"""

import json
import os
from eppy.modeleditor import IDF
from eppy.runner.run_functions import runIDFs

def make_eplaunch_options(idf):
    """Make options for run, so that it runs like EPLaunch on Windows"""
    idfversion = idf.idfobjects['version'][0].Version_Identifier.split('.')
    idfversion.extend([0] * (3 - len(idfversion)))
    idfversionstr = '-'.join([str(item) for item in idfversion])
    fname = idf.idfname
    options = {
        'ep_version':idfversionstr, # runIDFs needs the version number
        'output_prefix':os.path.basename(fname).split('.')[0],
        'output_suffix':'C',
        'output_directory':'data/energy_plus_output/test/',
        'readvars':False,
        'expandobjects':True
        }
    return options

def main():
    iddfile = '/Applications/EnergyPlus-9-4-0/PreProcess/IDFVersionUpdater/V9-4-0-Energy+.idd'
    IDF.setiddname(iddfile)
    epwfile = 'data/weather/USA_TX_Austin-Camp.Mabry.722544_TMY3.epw'

    # with open('data/ornl_single_family_ids.json') as f:
    #     ids = json.load(f)
    
    # filenames = [f'ornl/idf/TX_Austin/{id}.idf' for id in ids][0:1]
    filenames = ['data/energy_plus_output/test/test.idf']
    idfs = (IDF(filename, epwfile) for filename in filenames)
    runs = ((idf, make_eplaunch_options(idf) ) for idf in idfs)
    num_CPUs = 2
    runIDFs(runs, num_CPUs)

if __name__ == '__main__':
    main()