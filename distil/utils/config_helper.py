import json
import os

def read_config_file(filename):

    """
    Loads and returns a configuration from the supplied filename / path.
    
    Parameters
    ----------
    filename: string
        The name/path of the config file to load.
    Returns
    ----------
    config: object
        The resulting configuration laoded from the JSON file
    """
    
    print(filename.split('.')[-1])
    if filename.split('.')[-1] not in ['json']:
        raise IOError('Only json type are supported now!')
	
    if not os.path.exists(filename):
        raise FileNotFoundError('Config file does not exist!')
        
    with open(filename, 'r') as f:
        config = json.load(f)

    return config	