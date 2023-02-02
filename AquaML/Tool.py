

# concatenate the dictionary 
def concatenate_dictionary(*args):
    result = {}
    
    for dict_ in args:
        for key, value in dict_.items():
            if key in result:
                raise ValueError("key {} is already in result".format(key))
            else:
                result[key] = value
    
    return result