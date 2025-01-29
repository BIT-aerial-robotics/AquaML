import abc

class BaseRLAlgo(abc.ABC):
    
    def __init__(self):
        
        pass
    
    @abc.abstractmethod
    def _getAction(self, state: dict)-> dict:
        '''
        Get the action from the state.
        
        :param state: The state needed to get the action.
        :type state: dict{str: tensor or numpy.array} or tensor or numpy.array.
        
        :return: The action to take.
        :rtype: dict{str: tensor or numpy.array}.
        '''
        raise NotImplementedError