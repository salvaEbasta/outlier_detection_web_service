""" 
    __init__:
    
    args, _, _, values = inspect.getargvalues(inspect.currentframe())
    values.pop("self")

    for arg, val in values.items():
        setattr(self, arg, val) 
"""