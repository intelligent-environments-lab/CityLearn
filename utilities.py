import simplejson as json
import os
from datetime import datetime

def read_json(filepath):
    """Return json document as dictionary.
    
    Parameters
    ----------
    filepath : str
       pathname of JSON document.
       
    Returns
    -------
    dict
        JSON document converted to dictionary.
    """

    with open(filepath) as f:
        json_file = json.load(f)
    return json_file

def write_json(filepath,dictionary,**kwargs):
    """Writes dictionary to json file. Returns boolen value of operation success. 
    
    Parameters
    ----------
    filepath : str
        pathname of JSON document.
    dictionary: dict
        dictionary to convert to JSON.
    """

    def default(obj):
        if isinstance(obj,datetime):
            return obj.__str__()
            
    
    try:
        kwargs = {
            'ignore_nan':True,
            'sort_keys':False,
            'default':default,
            'indent':2,
            **kwargs
        }
        with open(filepath,'w') as f:
            json.dump(dictionary,f,**kwargs)
    except Exception as e:
        pass

def write_data(data,filepath,mode='w'):
    with open(filepath,mode) as f:
        f.write(data)

def get_data_from_path(filepath):
    if os.path.exists(filepath):
        with open(filepath,mode='r') as f:
            data = f.read()
    else:
        data = filepath

    return data

def unnest_dict(nested_dict,seperator=',',prefix=None):
    master_unnest = {}

    for key, value in nested_dict.items():
        key = f'{"" if prefix is None else str(prefix)+seperator}{key}'
        if isinstance(value,dict):
            child_unnest = unnest_dict(value,prefix=key)
            master_unnest = {**master_unnest,**child_unnest}

        else:
            master_unnest[key] = value

    return master_unnest