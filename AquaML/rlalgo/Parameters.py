import abc

class BaseParameter(abc.ABC):
    
    def __init__(self,epoch_length:int, n_epochs:int, batch_size:int,update_interval:int=0):
        """
        Parameters of environment.
        
        All the algorithm parameters should inherit this class.

        Args:
            epoch_length (int): The length of epoch.
            n_epochs (int): Times of optimizing the network.
            batch_size (int): batch size.
            update_interval (int): update interval.
        """
        
        

class SAC_parameter(BaseParameter):
    
    def __init__(self, epoch_length: int, n_epochs: int, batch_size: int,
                 update_interval: int, 
                 discount: float, tau: float):
        """
        Parameters of SAC algorithm.
        
        Reference:
        ----------
        [1] Haarnoja, Tuomas, et al. "Soft actor-critic: Off-policy maximum 
        entropy deep reinforcement learning with a stochastic actor." 
        arXiv preprint arXiv:1801.01290 (2018).

        Args:
            epoch_length (int): length of epoch.
            n_epochs (int): times of optimizing the network.
            batch_size (int): batch size.
            update_interval (int): update interval.
            discount (float): discount factor.
            tau (float): Soft value function target update weight.
  
            
        """
        super().__init__(epoch_length, n_epochs, batch_size, update_interval)
        
        self.discount = discount
        self.tau = tau
        # self.learning_rate = learning_rate
        # self.update_interval = update_interval

class SAC2_parameter(BaseParameter):
    
    def __init__(self, epoch_length: int, n_epochs: int,
                 batch_size: int, 
                 discount: float, alpha: float,tau:float,
                 update_interval: int = 0):
        """
        Parameters of SAC2 algorithm.
        
        Reference:
        ----------
        [1] Haarnoja, Tuomas, et al. "Soft actor-critic algorithms and applications." 
        arXiv preprint arXiv:1812.05905 (2018).

        Args:
            epoch_length (int): length of epoch.
            n_epochs (int): times of optimizing the network.
            batch_size (int): batch size.
            discount (float): discount factor.
            alpha (float): temperature parameter.
            tau (float): Soft q function target update weight.
            update_interval (int, optional): update interval. Defaults to 0.
        """
        super().__init__(epoch_length, n_epochs, batch_size, update_interval)
        
        self.discount = discount
        self.alpha = alpha
        self.tau = tau
        self.update_interval = update_interval
    