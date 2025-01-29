import abc

class BaseEnv(abc.ABC):
    '''
    Base class for all environments.
    The step function must contain the reset function.
    It means that the step function can automatically reset the environment when the episode ends.
    '''
    
    @abc.abstractmethod
    def reset(self)-> tuple:
        '''
        Reset the environment to the initial state.
        
        :return: The initial state.
        :rtype: dict{str: tensor or numpy.array} or tensor or numpy.array.
        '''
        raise NotImplementedError
    
    @abc.abstractmethod
    def step(self, action)-> tuple:
        '''
        Take an action and return the next state, reward, done, and info.
        
        :param action: The action to take. 
        :type action: tensor or dict{str: tensor}.
        
        :return: The next state, reward, done, and info.
        :rtype: dict{str: tensor or numpy.array}, tensor or numpy.array, tensor or numpy.array, dict{str: tensor or numpy.array}.
        '''
        raise NotImplementedError
    
    
    
    