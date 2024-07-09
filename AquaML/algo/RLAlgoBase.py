from AquaML.algo.AlgoBase import AlgoBase
import abc
from AquaML.param.ParamBase import RLParmBase
from AquaML.core.DataInfo import DataInfo

class RLAlgoBase(AlgoBase):
    """
    RL算法的基类。
    """

    def __init__(self,
                 hyper_params:RLParmBase,
                 model_dict,
                 ):
        """
        RL算法的基类。
        """
        super(RLAlgoBase, self).__init__(hyper_params, model_dict)
        
        ############################
        # 通用接口部分
        ############################
        
        # 算法产生的额外的数据信息，比如在PPO算法中，产生随机动作会有log_prob信息
        self._action_info = DataInfo() 

    
    @abc.abstractmethod
    def init(self):
        """
        通用初始化部分，通用的初始化部分。
        """
        # TODO：未来优化这一部分代码。
        pass
    
    def set_get_action(self, testing=False):
        """
        设置动作函数。
        
        Args:
            action_fn (function): 动作函数。
        """
        if testing:
            self._get_action_fn = self._test_action
        else:
            self._get_action_fn = self._train_action
            
    def get_action(self, state):
        """
        获取动作。
        
        Args:
            state (np.ndarray): 状态。
        
        Returns:
            dict: 动作。
            mu: 动作的均值,返回不加探索的噪声。
        """
        return self._get_action_fn(state)
            
        
    
    @abc.abstractmethod
    def _train_action(self, state):
        """
        获取动作。
        
        Args:
            state (np.ndarray): 状态。
        
        Returns:
            dict: 动作。
            mu: 动作的均值,返回不加探索的噪声。
        """
        pass
    
    def _test_action(self, state):
        """
        测试动作。
        
        Args:
            state (np.ndarray): 状态。
        
        Returns:
            np.ndarray: 动作。
        """
        
        action_dict, mu = self._train_action(state)
        
        action_dict['action'] = mu
        
        return action_dict, mu
    
    @property
    def action_info(self)->DataInfo:
        """
        动作信息。
        """
        return self._action_info