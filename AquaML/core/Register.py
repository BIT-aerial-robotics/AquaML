from abc import ABC, abstractmethod

class RegisterBase(ABC):

    def __init__(self,
                 ):
        
        ########################################
        # API
        ########################################
        self._args_dict = {} # provide args for other aqua, plugin, buffer, etc.

    
    def get_obj(self, obj_name: str):

        if obj_name is None:

            return None
        
        obj = getattr(self, obj_name, None)

        if obj is None:
            raise NotImplementedError
        
        return obj