import abc

from AquaML.core.BaseParameter import BaseParameter


class BaseAgentParameter(BaseParameter, abc.ABC):
    def __init__(self,
        rollout_steps:int,
        epochs:int,
        batch_size:int,
        explore_policy='Default',
    ):
        """
        2.1版本中，优化参数设置，更加人性化，更加易于理解。
        2.1将全面支持不定长MDP。

        Args:
            rollout_steps (int): 单个线程的rollout步数。
            epochs (int): 算法的训练轮数。
            batch_size (int): 算法的训练批次大小。
        """
        super().__init__()
        self.rollout_steps = rollout_steps
        self.epochs = epochs
        self.batch_size = batch_size
        self.explore_policy = explore_policy

    def point_adjust_parameter(self, names: list):
        """
        For higher level algorithm, you can adjust the parameter in this function.
        """

        self.adjust_parameters = names

        # check the parameter name whether is in the class
        for name in names:
            if not hasattr(self, name):
                raise AttributeError(f'{name} is not in {self.__class__.__name__} class')
            
    def set_adjust_parameter_value(self, parm_dict: dict):
        for key, value in parm_dict.items():
            setattr(self, key, value)


class DDPGAgentParameter(BaseAgentParameter):

    def __init__(self, 
                 rollout_steps: int, 
                 epochs: int, 
                 batch_size: int = 128,
                 tau: float = 0.001,
                 explore_policy='Default',
                 ):
        super().__init__(
            rollout_steps=rollout_steps,
            epochs=epochs,
            batch_size=batch_size,
            explore_policy=explore_policy,
        )

        self.tau = tau


class PPOAgentParameter(BaseAgentParameter):

    def __init__(self, 
                 rollout_steps: int, 
                 epochs: int, 
                 batch_size: int, 
                 explore_policy='Default',
                 log_std_init_value: float = -0.5,
                 ):
        super().__init__(
            rollout_steps=rollout_steps,
            epochs=epochs,
            batch_size=batch_size,
            explore_policy=explore_policy,
        )

        self.log_std_init_value = log_std_init_value