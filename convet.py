import h5py
import json
import numpy as np


def h5_to_dict(name, obj):
    """Convert HDF5 object to a dictionary."""
    if isinstance(obj, h5py.Group):
        result = {}
        for key, item in obj.items():
            result[key] = h5_to_dict(key, item)
        return result
    elif isinstance(obj, h5py.Dataset):
        return obj[()]
    else:
        raise TypeError(f"Unsupported HDF5 object type: {type(obj)}")

def convert_h5_to_json(h5_filename, json_filename):
    with h5py.File(h5_filename, 'r') as h5_file:
        data = h5_to_dict('', h5_file)
        with open(json_filename, 'w') as json_file:
            json.dump(data, json_file, indent=4, default=str)

# Example usage
h5_filename = 'model_weights.h5'
json_filename = 'models.json'
convert_h5_to_json(h5_filename, json_filename)
