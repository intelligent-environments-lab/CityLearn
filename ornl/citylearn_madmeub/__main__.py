import argparse
import inspect
import os
from shutil import copy
import sys
from citylearn_madmeub.ornl import CityLearnIDF
from citylearn_madmeub.utilities import read_json, write_json

def simulate(idf_directory,filepath=None):
    filepath = os.path.join(os.path.dirname(__file__),'../data/selected.json') if filepath is None else filepath
    selected_idfs = read_json(filepath)[0:2]
    print('Simulating ...')
    
    for i, selected_idf in enumerate(selected_idfs):
        print(f'{i+1}/{len(selected_idfs)} {selected_idf}')
        selected_idf['idf_filepath'] = os.path.join(idf_directory,selected_idf['idf_filepath'])
        selected_idf['id'] = __build_id(selected_idf)
        args = (selected_idf['idf_filepath'],selected_idf['climate_zone'],selected_idf['building_type'])
        kwargs = {"id":selected_idf['id'],"random_state":int(selected_idf['id'].split('_')[-2])}
        clidf = CityLearnIDF(*args,**kwargs)
        clidf.preprocess()
        clidf.simulate()
        clidf.save()

def upload(filepath=None,root_citylearn_directory=None,root_output_directory=None):
    filepath = os.path.join(os.path.dirname(__file__),'../data/selected.json') if filepath is None else filepath
    root_citylearn_directory = os.path.join(os.path.dirname(__file__),'../../') if root_citylearn_directory is None else root_citylearn_directory
    root_output_directory = os.path.join(os.path.dirname(__file__),CityLearnIDF.settings()['root_output_directory']) if root_output_directory is None else root_output_directory
    citylearn_building_state_action_space_filepath = os.path.join(root_citylearn_directory,'buildings_state_action_space.json')
    citylearn_building_state_action_space = read_json(citylearn_building_state_action_space_filepath)
    citylearn_building_attributes = {}
    selected_idfs = read_json(filepath)
    print('Uploading ...')

    for i, selected_idf in enumerate(selected_idfs):
        output_id = __build_id(selected_idf)
        output_directory = os.path.join(root_output_directory,output_id)

        try:
            assert os.path.isdir(output_directory)
        except AssertionError:
            continue
        
        print(f'{i+1}/{len(selected_idfs)} {selected_idf}')
        building_key = f'Building_{output_id}'
        timeseries_source_filepath = os.path.join(os.path.join(root_output_directory,output_id),f'{output_id}_timeseries.csv')
        timeseries_destination_filepath = os.path.join(root_citylearn_directory,f'data/Climate_Zone_{selected_idf["climate_zone"]}/{building_key}.csv')
        _ = copy(timeseries_source_filepath,timeseries_destination_filepath)
        state_action_space_source_filepath = os.path.join(os.path.join(root_output_directory,output_id),f'{output_id}_state_action_space.json')
        citylearn_building_state_action_space[building_key] = read_json(state_action_space_source_filepath)
        attributes_source_filepath = os.path.join(os.path.join(root_output_directory,output_id),f'{output_id}_attributes.json')
        attributes_destination_filepath = os.path.join(root_citylearn_directory,f'data/Climate_Zone_{selected_idf["climate_zone"]}/building_attributes.json')

        if attributes_destination_filepath not in citylearn_building_attributes.keys():
            citylearn_building_attributes[attributes_destination_filepath] = read_json(attributes_destination_filepath)
        else:
            pass
         
        citylearn_building_attributes[attributes_destination_filepath][building_key] = read_json(attributes_source_filepath)

    write_json(citylearn_building_state_action_space_filepath,citylearn_building_state_action_space)
    
    for filepath, building_attributes in citylearn_building_attributes.items():
        write_json(filepath,building_attributes)

def __build_id(selected_idf):
    return f'madmeub_{"_".join(selected_idf["idf_filepath"][0:-4].split("/")[-2:])}_CZ{selected_idf["climate_zone"]}'

def main():
    parser = argparse.ArgumentParser(prog='citylearn_madmeub',description='Add buildings from the Model America – data and models of every U.S. building dataset to the CityLearn environment.')
    parser.add_argument('--version', action='version', version='%(prog)s 0.0.1')
    parser.add_argument('-f','--filepath',type=str,default=None,dest='filepath',help='Filepath to selected IDFs JSON.')
    subparsers = parser.add_subparsers(title='subcommands',required=True,dest='subcommands')
    
    # simulate
    sp_simulate = subparsers.add_parser('simulate',description='Simulate selected IDFs.')
    sp_simulate.set_defaults(func=simulate)
    sp_simulate.add_argument('idf_directory',help='Path to directory containing raw IDF files to simulate.')

    # upload
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