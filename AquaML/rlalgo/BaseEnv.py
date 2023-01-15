import abc

# TODO: 重新规定环境的接口
class BaseEnv(abc.ABC):
    """
    The base class of environment.
    
    All the environment should inherit this class.
    
    """
    
    @abc.abstractmethod
    def reset(self):
        """
        Reset the environment.
        
        return: 
        observation (dict): observation of environment.
        """
    
    @abc.abstractmethod
    def step(self, action):
        """
        Step the environment.
        
        Args:
            action (optional): action of environment.
        Return: 
        observation (dict): observation of environment.
        reward(dict): reward of environment.
        done (bool): done flag of environment.
        info (dict or None): info of environment.
        """
    