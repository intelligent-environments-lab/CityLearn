import simplejson as json

def read_json(filepath: str, **kwargs):
    """Return json document as dictionary.
    
    Parameters
    ----------
    filepath : str
       pathname of JSON document.

    Other Parameters
    ----------------
    **kwargs : dict
        Other infrequently used keyword arguments to be parsed to `simplejson.load`.
       
    Returns
    -------
    dict
        JSON document converted to dictionary.
    """

    with open(filepath) as f:
        json_file = json.load(f,**kwargs)

    return json_file

def write_json(filepath: str, dictionary: dict, **kwargs):
    """Write dictionary to json file. Returns boolen value of operation success. 
    
    Parameters
    ----------
    filepath : str
        pathname of JSON document.
    dictionary: dict
        dictionary to convert to JSON.

    Other Parameters
    ----------------
    **kwargs : dict
        Other infrequently used keyword arguments to be parsed to `simplejson.dump`.
    """

    kwargs = {'ignore_nan':True,'sort_keys':False,'default':str,'indent':2,**kwargs}
    with open(filepath,'w') as f:
        json.dump(dictionary,f,**kwargs)