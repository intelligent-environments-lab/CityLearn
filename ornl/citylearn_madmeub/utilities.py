"""A module containing utility functions that are used by other modules.

Methods
-------
get_read_json(filepath)
    Return json document as dictionary.
def make_directory(directory)
    Make nested directories.
"""

from datetime import datetime
import os
import numpy as np
import simplejson as json

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
            
    with open(filepath,'w') as f:
        kwargs = {
            **{'ignore_nan':True,'sort_keys':False,'default':default,'indent':2},
            **kwargs
        }
        json.dump(dictionary,f,**kwargs)

def make_directory(directory):
    """Make nested directories.

    Parameters
    ----------
    directory : str
        directory to make. Example is: 'directory/to/make'.
    """

    os.makedirs(directory,exist_ok=True)

def verbosity_printer(verbosity,priority,text):
    """Print output if text priority is alteast equal to verbosity level.

    Parameters
    ----------
    verbosity : int
        output verbosity level.
    priority : int
        text priority level.
    text : str
        text to be printed
    """
    
    if verbosity >= priority:
        print(text)
    else:
        pass

# Source: https://github.com/hmallen/numpyencoder/blob/master/numpyencoder/numpyencoder.py
class NumpyEncoder(json.JSONEncoder):
    """ Custom encoder for numpy data types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):

            return int(obj)

        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        
        elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
            return {'real': obj.real, 'imag': obj.imag}
        
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
    
        elif isinstance(obj, (np.bool_)):
            return bool(obj)

        elif isinstance(obj, (np.void)): 
            return None

        return json.JSONEncoder.default(self, obj)