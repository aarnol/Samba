 
from model.samba_ele_to_hemo import SambaEleToHemo 
from model.samba_hemo_to_ele import SambaHemoToEle 

# List of model classes
MODEL_LIST = [
    SambaEleToHemo, SambaHemoToEle,    
]

# Create a dictionary with both original and lowercase model names as keys
MODEL_DICT = {x.__name__: x for x in MODEL_LIST}
MODEL_DICT.update({x.__name__.lower(): x for x in MODEL_LIST})

def str2model(model_name):
    """ Convert string to model class object, handling both cases. """
    try:
        model = MODEL_DICT[model_name]
    except KeyError: 
        raise NotImplementedError(f"{model_name} not implemented. Check model registry")
    return model