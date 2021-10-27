import argparse
import inspect
import os
import sys
from citylearn_madmeub.ornl import CityLearnIDF
from citylearn_madmeub.utilities import read_json

def simulate(filepath=None):
    if filepath is None:
        filepath = os.path.join(os.path.dirname(__file__),'../data/idf/selected.json')
    else:
        pass
    
    for selected_idf in read_json(filepath):
        selected_idf['idf_filepath'] = os.path.join(
            os.path.join(os.path.dirname(__file__),'../data/idf/counties/'),
            selected_idf['idf_filepath']
        )
        selected_idf['id'] = __build_id(selected_idf)
        args = (selected_idf['idf_filepath'],selected_idf['climate_zone'],selected_idf['building_type'])
        kwargs = {"id":selected_idf['id'],"random_state":int(selected_idf['id'].split('_')[-1])}
        clidf = CityLearnIDF(*args,**kwargs)
        clidf.preprocess()
        clidf.simulate()
        clidf.save()
        break

def upload(filepath=None,root_citylearn_directory=None,root_output_directory=None):
    if filepath is None:
        filepath = os.path.join(os.path.dirname(__file__),'../data/idf/selected.json')
    else:
        pass

    if root_citylearn_directory is None:
        root_citylearn_directory = os.path.join(os.path.dirname(__file__),'../../')
    else:
        pass

    if root_output_directory is None:
        root_output_directory = os.path.join(os.path.dirname(__file__),CityLearnIDF.settings()['root_output_directory'])
    else:
        pass

    for selected_idf in read_json(filepath):
        output_id = __build_id(selected_idf)
        timeseries_source_filepath = os.path.join(os.path.join(root_output_directory,output_id),f'{output_id}_timeseries.csv')
        attributes_source_filepath = os.path.join(os.path.join(root_output_directory,output_id),f'{output_id}_attributes.json')
        state_action_space_source_filepath = os.path.join(os.path.join(root_output_directory,output_id),f'{output_id}_timeseries.json')


def __build_id(selected_idf):
    return f'{selected_idf["climate_zone"]}_{"_".join(selected_idf["idf_filepath"].split("/")[-2:])}'[0:-4]

def main():
    parser = argparse.ArgumentParser(prog='citylearn_madmeub',description='Add buildings from the Model America – data and models of every U.S. building dataset to the CityLearn environment.')
    parser.add_argument('--version', action='version', version='%(prog)s 0.0.1')
    parser.add_argument('-f','--filepath',type=str,default=None,dest='filepath',help='Filepath to selected IDFs JSON.')
    subparsers = parser.add_subparsers(title='subcommands',required=True,dest='subcommands')
    
    # simulate
    sp_simulate = subparsers.add_parser('simulate',description='Simulate selected IDFs.')
    sp_simulate.set_defaults(func=simulate)

    # # upload
    sp_upload = subparsers.add_parser('upload',description='Upload simulation output to CityLearn environment.')
    sp_upload.set_defaults(func=upload)
    sp_upload.add_argument('-c','--root_citylearn_directory',type=str,default=None,dest='root_citylearn_directory',help='CityLearn repository root directory.')
    sp_upload.add_argument('-o','--root_output_directory',type=str,default=None,dest='root_output_directory',help='Simulation output root directory.')

    args = parser.parse_args()
    arg_spec = inspect.getfullargspec(args.func)
    args_for_func = {k:getattr(args,k,None) for k in arg_spec.args}
    args.func(**args_for_func)

if __name__ == '__main__':
    sys.exit(main())